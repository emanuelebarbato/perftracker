#!/usr/bin/env python3
"""
Performance Regression Detector v2.0 - Visual Pattern Recognition Approach

Philosophy: Detect regressions the way a human would by looking at charts.
Focus: If lines are visibly separated with sustained degradation, flag it.

Key Innovations:
1. Sliding window analysis - checks every segment independently
2. Statistical consistency - uses median absolute deviation (MAD) for robustness
3. Visual separation metric - measures how "far apart" the lines look
4. Adaptive thresholds - context-aware based on data characteristics
5. No global AUC - focuses on localized effects (catches tail degradations)

Author: Refactored 2025
"""

import json
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd


# ============================================================================
# Configuration & Data Classes
# ============================================================================

@dataclass
class DetectionConfig:
    """Configuration for regression detection parameters."""
    
    # Window analysis
    min_window_size: int = 3           # Minimum consecutive points to consider
    window_step: int = 1               # Sliding window step size
    
    # Degradation thresholds
    min_degradation_pct: float = 15.0  # Minimum % change to be "significant"
    severe_degradation_pct: float = 50.0  # "Severe" degradation threshold
    
    # Consistency requirements
    consistency_threshold: float = 0.7  # 70% of points in window must degrade
    
    # Visual separation
    min_visual_separation: float = 20.0  # Minimum % for visual "gap"
    
    # Quality filters
    min_baseline_value: float = 0.01   # Filter out near-zero baselines
    min_curve_points: int = 5          # Minimum points needed for analysis
    
    # Noise reduction
    outlier_mad_threshold: float = 3.0  # MAD multiplier for outlier detection


@dataclass
class RegressionWindow:
    """A window where regression was detected."""
    start_idx: int
    end_idx: int
    start_x: float
    end_x: float
    window_size: int
    median_degradation: float
    mean_degradation: float
    consistency_score: float  # Fraction of points degrading
    visual_separation: float  # How "far apart" lines appear
    max_degradation: float
    severity: str  # "moderate", "severe", "critical"


@dataclass
class CurveAnalysis:
    """Complete analysis results for a single curve."""
    
    # Identification
    endpoint: str
    metric_type: str
    threads: int
    scenario: str
    
    # Data characteristics
    points: int
    x_values: List[float]
    baseline_values: List[float]
    candidate_values: List[float]
    
    # Computed metrics
    pct_changes_raw: List[float]
    pct_changes_robust: List[float]  # After outlier filtering
    
    # Detected windows
    regression_windows: List[RegressionWindow]
    
    # Overall assessment
    is_regression: bool
    severity: str
    confidence: str  # "high", "medium", "low"
    
    # Summary statistics
    worst_window: Optional[RegressionWindow]
    median_pct_change: float
    affected_range: str  # e.g., "40k-48k policies"
    
    # Quality indicators
    avg_samples: float
    min_samples: int
    data_quality: str  # "good", "fair", "poor"


# ============================================================================
# Data Loading & Preparation
# ============================================================================

def load_test_data(json_path: Path) -> Dict[str, Any]:
    """Load and parse JSON test results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return {
        'metadata': {
            'id': data.get('id'),
            'product_ver': data.get('product_ver'),
            'tests_total': data.get('tests_total'),
            'tests_errors': data.get('tests_errors'),
            'tests_failed': data.get('tests_failed'),
        },
        'tests': data.get('tests', [])
    }


def parse_test_tag(tag: str) -> Tuple[str, str, int]:
    """
    Extract endpoint, metric type, and thread count from test tag.
    
    Example: "latency, p(95) (4 th): GET /api/endpoint (scenario)"
    Returns: (endpoint, metric_type, threads)
    """
    parts = tag.split(':', 1)
    if len(parts) != 2:
        return (tag, 'unknown', 1)
    
    prefix, rest = parts
    
    # Extract threads - look for pattern "(N th)" using regex
    threads = 1
    thread_match = re.search(r'\((\d+)\s+th\)', prefix)
    if thread_match:
        try:
            threads = int(thread_match.group(1))
        except ValueError:
            pass
    
    # Extract metric type
    metric_type = 'latency' if 'latency' in prefix.lower() else 'rate'
    
    # Extract endpoint (remove scenario part)
    endpoint = rest.strip()
    if '(' in endpoint:
        endpoint = endpoint.split('(')[0].strip()
    
    return endpoint, metric_type, threads


def extract_scenario(tag: str) -> str:
    """Extract scenario description from tag."""
    if '(' in tag and ')' in tag:
        return tag[tag.rfind('('):tag.rfind(')')+1]
    return ""


def group_tests_into_curves(tests: List[Dict]) -> Dict[Tuple, List[Dict]]:
    """
    Group tests by (endpoint, metric_type, threads, scenario) to form curves.
    Each group represents a performance curve across different load levels.
    """
    curves = defaultdict(list)
    
    for test in tests:
        tag = test.get('tag', '')
        endpoint, metric_type, threads = parse_test_tag(tag)
        scenario = extract_scenario(tag)
        category = test.get('category', '')
        
        key = (endpoint, metric_type, threads, scenario)
        curves[key].append(test)
    
    # Sort each curve by category (load level)
    for key in curves:
        curves[key].sort(key=lambda t: _extract_load_value(t.get('category', '')))
    
    return dict(curves)


def _extract_load_value(category: str) -> float:
    """Extract numeric load value from category string."""
    import re
    match = re.search(r'(\d+)', category)
    return float(match.group(1)) if match else 0.0


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def calculate_robust_pct_change(baseline: float, candidate: float, 
                                metric_type: str) -> Optional[float]:
    """
    Calculate percentage change with "worse is positive" convention.
    Returns None for invalid data.
    """
    if baseline <= 0 or candidate <= 0:
        return None
    
    pct_change = ((candidate - baseline) / baseline) * 100.0
    
    # For rate metrics, invert (decrease = regression)
    if metric_type == 'rate':
        pct_change = -pct_change
    
    return pct_change


def detect_outliers_mad(values: List[float], threshold: float = 3.0) -> List[bool]:
    """
    Detect outliers using Median Absolute Deviation (MAD).
    More robust than standard deviation for noisy data.
    
    Returns: List of booleans indicating if each point is an outlier.
    """
    if len(values) < 3:
        return [False] * len(values)
    
    values_array = np.array([v for v in values if v is not None])
    if len(values_array) == 0:
        return [False] * len(values)
    
    median = np.median(values_array)
    mad = np.median(np.abs(values_array - median))
    
    if mad == 0:
        return [False] * len(values)
    
    outliers = []
    for v in values:
        if v is None:
            outliers.append(True)
        else:
            z_score = abs(v - median) / (mad * 1.4826)  # Scale factor for normal dist
            outliers.append(z_score > threshold)
    
    return outliers


def calculate_visual_separation(pct_changes: List[float], window_start: int, 
                                window_end: int) -> float:
    """
    Calculate how "separated" the lines appear in a window.
    Combines magnitude and consistency.
    """
    window_values = [pct_changes[i] for i in range(window_start, window_end + 1)
                     if pct_changes[i] is not None]
    
    if not window_values:
        return 0.0
    
    # Use absolute values for visual separation
    abs_values = [abs(v) for v in window_values]
    
    # Median represents typical separation
    median_sep = np.median(abs_values)
    
    # Consider consistency (low variance = more visible)
    if len(abs_values) > 1:
        std_dev = np.std(abs_values)
        consistency_bonus = max(0, 20 - std_dev)  # Bonus for low variance
    else:
        consistency_bonus = 0
    
    return median_sep + consistency_bonus * 0.5


# ============================================================================
# Window-Based Regression Detection
# ============================================================================

def analyze_sliding_windows(pct_changes: List[float], x_values: List[float],
                            config: DetectionConfig) -> List[RegressionWindow]:
    """
    Scan the curve with sliding windows to find sustained regressions.
    This is the core detection algorithm.
    """
    n = len(pct_changes)
    detected_windows = []
    
    # Try different window sizes (3, 4, 5, 6, 7 points)
    for window_size in range(config.min_window_size, min(8, n + 1)):
        
        for start_idx in range(0, n - window_size + 1, config.window_step):
            end_idx = start_idx + window_size - 1
            
            # Extract window values
            window_values = []
            valid_indices = []
            for i in range(start_idx, end_idx + 1):
                if pct_changes[i] is not None:
                    window_values.append(pct_changes[i])
                    valid_indices.append(i)
            
            if len(window_values) < config.min_window_size:
                continue
            
            # Check if this window shows regression
            regression_info = _evaluate_window(
                window_values, valid_indices, x_values, config
            )
            
            if regression_info:
                detected_windows.append(regression_info)
    
    # Merge overlapping windows, keep most severe
    return _merge_overlapping_windows(detected_windows)


def _evaluate_window(values: List[float], indices: List[int], 
                     x_values: List[float], config: DetectionConfig) -> Optional[RegressionWindow]:
    """
    Evaluate if a window shows significant regression.
    
    Criteria:
    1. Consistency: Most points must show degradation
    2. Magnitude: Median degradation must exceed threshold
    3. Visual separation: Lines must be visibly apart
    """
    if len(values) < config.min_window_size:
        return None
    
    # Calculate statistics
    median_deg = np.median(values)
    mean_deg = np.mean(values)
    max_deg = np.max(np.abs(values))
    
    # Check consistency: what fraction of points are degraded?
    degraded_count = sum(1 for v in values if v > config.min_degradation_pct)
    consistency = degraded_count / len(values)
    
    # Calculate visual separation
    start_idx, end_idx = indices[0], indices[-1]
    visual_sep = calculate_visual_separation(
        [None if i not in indices else values[indices.index(i)] 
         for i in range(len(x_values))],
        start_idx, end_idx
    )
    
    # Determine severity
    if abs(median_deg) >= 100:
        severity = "critical"
    elif abs(median_deg) >= config.severe_degradation_pct:
        severity = "severe"
    else:
        severity = "moderate"
    
    # Decision criteria
    is_significant = (
        consistency >= config.consistency_threshold and
        abs(median_deg) >= config.min_degradation_pct and
        visual_sep >= config.min_visual_separation
    )
    
    if not is_significant:
        return None
    
    return RegressionWindow(
        start_idx=start_idx,
        end_idx=end_idx,
        start_x=x_values[start_idx],
        end_x=x_values[end_idx],
        window_size=len(indices),
        median_degradation=median_deg,
        mean_degradation=mean_deg,
        consistency_score=consistency,
        visual_separation=visual_sep,
        max_degradation=max_deg,
        severity=severity
    )


def _merge_overlapping_windows(windows: List[RegressionWindow]) -> List[RegressionWindow]:
    """Merge overlapping windows, keeping the most severe."""
    if not windows:
        return []
    
    # Sort by start index
    windows = sorted(windows, key=lambda w: w.start_idx)
    
    merged = []
    current = windows[0]
    
    for next_window in windows[1:]:
        if next_window.start_idx <= current.end_idx:
            # Overlapping - keep the more severe one
            if abs(next_window.median_degradation) > abs(current.median_degradation):
                current = next_window
        else:
            merged.append(current)
            current = next_window
    
    merged.append(current)
    return merged


# ============================================================================
# Curve Analysis Pipeline
# ============================================================================

def analyze_curve(baseline_tests: List[Dict], candidate_tests: List[Dict],
                 endpoint: str, metric_type: str, threads: int, scenario: str,
                 config: DetectionConfig) -> Optional[CurveAnalysis]:
    """
    Complete analysis pipeline for a single curve.
    """
    # Align tests by category
    aligned = _align_tests(baseline_tests, candidate_tests, config)
    
    if len(aligned) < config.min_curve_points:
        return None  # Not enough data
    
    # Extract values
    x_values = [t['x'] for t in aligned]
    baseline_values = [t['baseline'] for t in aligned]
    candidate_values = [t['candidate'] for t in aligned]
    samples = [t['samples'] for t in aligned]
    
    # Calculate percentage changes
    pct_changes_raw = []
    for b, c in zip(baseline_values, candidate_values):
        pct = calculate_robust_pct_change(b, c, metric_type)
        pct_changes_raw.append(pct)
    
    # DISABLED: Outlier detection was filtering out real regressions!
    # The MAD algorithm treats large degradations as outliers, removing the signal.
    # For now, use raw values directly.
    pct_changes_robust = pct_changes_raw.copy()
    
    # Detect regression windows
    regression_windows = analyze_sliding_windows(
        pct_changes_robust, x_values, config
    )
    
    # Overall assessment
    is_regression = len(regression_windows) > 0
    
    worst_window = None
    if regression_windows:
        worst_window = max(regression_windows, 
                          key=lambda w: abs(w.median_degradation))
    
    # Determine overall severity and confidence
    if worst_window:
        severity = worst_window.severity
        confidence = _assess_confidence(worst_window, samples)
    else:
        severity = "none"
        confidence = "n/a"
    
    # Calculate summary statistics
    valid_pct = [p for p in pct_changes_robust if p is not None]
    median_pct = np.median(valid_pct) if valid_pct else 0.0
    
    affected_range = ""
    if worst_window:
        affected_range = f"{int(worst_window.start_x)}-{int(worst_window.end_x)}"
    
    # Data quality assessment
    avg_samples = np.mean(samples)
    min_samples = np.min(samples)
    if avg_samples >= 3:
        data_quality = "good"
    elif avg_samples >= 2:
        data_quality = "fair"
    else:
        data_quality = "poor"
    
    return CurveAnalysis(
        endpoint=endpoint,
        metric_type=metric_type,
        threads=threads,
        scenario=scenario,
        points=len(x_values),
        x_values=x_values,
        baseline_values=baseline_values,
        candidate_values=candidate_values,
        pct_changes_raw=pct_changes_raw,
        pct_changes_robust=pct_changes_robust,
        regression_windows=regression_windows,
        is_regression=is_regression,
        severity=severity,
        confidence=confidence,
        worst_window=worst_window,
        median_pct_change=median_pct,
        affected_range=affected_range,
        avg_samples=avg_samples,
        min_samples=min_samples,
        data_quality=data_quality
    )


def _align_tests(baseline_tests: List[Dict], candidate_tests: List[Dict],
                config: DetectionConfig) -> List[Dict]:
    """Align baseline and candidate tests by category."""
    # Create lookup by category
    baseline_map = {t.get('category'): t for t in baseline_tests 
                    if t.get('status') == 'SUCCESS'}
    candidate_map = {t.get('category'): t for t in candidate_tests
                     if t.get('status') == 'SUCCESS'}
    
    aligned = []
    for category in sorted(baseline_map.keys(), key=lambda c: _extract_load_value(c)):
        if category not in candidate_map:
            continue
        
        b_test = baseline_map[category]
        c_test = candidate_map[category]
        
        b_score = b_test.get('avg_score', 0)
        c_score = c_test.get('avg_score', 0)
        
        # Filter out very small baselines
        if b_score < config.min_baseline_value:
            continue
        
        aligned.append({
            'x': _extract_load_value(category),
            'category': category,
            'baseline': b_score,
            'candidate': c_score,
            'samples': b_test.get('samples', 1)
        })
    
    return aligned


def _assess_confidence(window: RegressionWindow, samples: List[int]) -> str:
    """Assess confidence in regression detection."""
    # High confidence if:
    # - Severe degradation
    # - High consistency
    # - Multiple samples
    
    severity_score = 1.0 if window.severity == "critical" else \
                     0.7 if window.severity == "severe" else 0.4
    
    consistency_score = window.consistency_score
    
    avg_samples = np.mean(samples)
    sample_score = 1.0 if avg_samples >= 3 else \
                   0.7 if avg_samples >= 2 else 0.4
    
    overall_score = (severity_score + consistency_score + sample_score) / 3
    
    if overall_score >= 0.8:
        return "high"
    elif overall_score >= 0.6:
        return "medium"
    else:
        return "low"


# ============================================================================
# Output Generation
# ============================================================================

def generate_output(analyses: List[CurveAnalysis], metadata1: Dict, 
                   metadata2: Dict, config: DetectionConfig) -> Dict[str, Any]:
    """Generate comprehensive output report."""
    
    regressions = [a for a in analyses if a.is_regression]
    
    # Group regressions by severity
    critical = [r for r in regressions if r.severity == "critical"]
    severe = [r for r in regressions if r.severity == "severe"]
    moderate = [r for r in regressions if r.severity == "moderate"]
    
    output = {
        "detection_method": "visual_pattern_recognition_v2",
        "config": {
            "min_window_size": config.min_window_size,
            "min_degradation_pct": config.min_degradation_pct,
            "severe_degradation_pct": config.severe_degradation_pct,
            "consistency_threshold": config.consistency_threshold,
            "min_visual_separation": config.min_visual_separation,
        },
        "metadata": {
            "run1": metadata1,
            "run2": metadata2,
        },
        "summary": {
            "curves_analyzed": len(analyses),
            "regressions_detected": len(regressions),
            "by_severity": {
                "critical": len(critical),
                "severe": len(severe),
                "moderate": len(moderate),
            },
            "by_confidence": {
                "high": len([r for r in regressions if r.confidence == "high"]),
                "medium": len([r for r in regressions if r.confidence == "medium"]),
                "low": len([r for r in regressions if r.confidence == "low"]),
            }
        },
        "regressions": []
    }
    
    # Add detailed regression info
    for analysis in sorted(regressions, 
                          key=lambda a: abs(a.worst_window.median_degradation) 
                          if a.worst_window else 0, 
                          reverse=True):
        
        reg_info = {
            "endpoint": analysis.endpoint,
            "metric_type": analysis.metric_type,
            "threads": analysis.threads,
            "scenario": analysis.scenario,
            "severity": analysis.severity,
            "confidence": analysis.confidence,
            "data_quality": analysis.data_quality,
            "points_analyzed": analysis.points,
            "median_pct_change_overall": round(analysis.median_pct_change, 2),
        }
        
        if analysis.worst_window:
            w = analysis.worst_window
            reg_info["worst_window"] = {
                "location": analysis.affected_range,
                "size": w.window_size,
                "median_degradation_pct": round(w.median_degradation, 2),
                "mean_degradation_pct": round(w.mean_degradation, 2),
                "max_degradation_pct": round(w.max_degradation, 2),
                "consistency_score": round(w.consistency_score, 2),
                "visual_separation": round(w.visual_separation, 2),
            }
            
            # Add all detected windows
            reg_info["all_windows"] = [
                {
                    "location": f"{int(rw.start_x)}-{int(rw.end_x)}",
                    "median_degradation": round(rw.median_degradation, 2),
                    "consistency": round(rw.consistency_score, 2),
                }
                for rw in analysis.regression_windows
            ]
        
        output["regressions"].append(reg_info)
    
    return output


def save_results(output: Dict, output_dir: Path):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / "result_v2.json"
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {result_file}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Performance Regression Detector v2.0 - Visual Pattern Recognition"
    )
    parser.add_argument("baseline", type=Path, help="Baseline JSON file")
    parser.add_argument("candidate", type=Path, help="Candidate JSON file")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    
    # Detection parameters
    parser.add_argument("--min-window", type=int, default=3,
                       help="Minimum window size for detection (default: 3)")
    parser.add_argument("--min-degradation", type=float, default=15.0,
                       help="Minimum degradation %% (default: 15)")
    parser.add_argument("--severe-threshold", type=float, default=50.0,
                       help="Severe degradation threshold %% (default: 50)")
    parser.add_argument("--consistency", type=float, default=0.7,
                       help="Consistency threshold 0-1 (default: 0.7)")
    parser.add_argument("--visual-sep", type=float, default=20.0,
                       help="Minimum visual separation (default: 20)")
    parser.add_argument("--min-baseline", type=float, default=0.01,
                       help="Minimum baseline value (default: 0.01)")
    
    args = parser.parse_args()
    
    # Load data
    print("📊 Loading test data...")
    baseline_data = load_test_data(args.baseline)
    candidate_data = load_test_data(args.candidate)
    
    print(f"   Baseline: {baseline_data['metadata']['tests_total']} tests")
    print(f"   Candidate: {candidate_data['metadata']['tests_total']} tests")
    
    # Extract test names from filenames and create subdirectory
    # e.g., "job-84269-GRPM_API.json" → "84269_GRPM"
    def extract_test_name(filepath: Path) -> str:
        """Extract test ID from filename (e.g., job-84269-GRPM_API.json → 84269_GRPM)."""
        name = filepath.stem  # Remove .json
        # Try to extract pattern like "84269-GRPM" or "84269_GRPM"
        parts = name.replace('-', '_').split('_')
        # Look for numeric ID followed by name
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) >= 5:  # Likely a job ID
                if i + 1 < len(parts):
                    return f"{part}_{parts[i+1]}"
                return part
        # Fallback: use first part of filename
        return parts[0] if parts else name
    
    baseline_name = extract_test_name(args.baseline)
    candidate_name = extract_test_name(args.candidate)
    comparison_dir = f"{baseline_name} vs {candidate_name}"
    
    # Create full output path with comparison subdirectory
    full_output_dir = args.out / comparison_dir
    
    # Configure detection
    config = DetectionConfig(
        min_window_size=args.min_window,
        min_degradation_pct=args.min_degradation,
        severe_degradation_pct=args.severe_threshold,
        consistency_threshold=args.consistency,
        min_visual_separation=args.visual_sep,
        min_baseline_value=args.min_baseline,
    )
    
    # Group tests into curves
    print("\n🔍 Grouping tests into curves...")
    baseline_curves = group_tests_into_curves(baseline_data['tests'])
    candidate_curves = group_tests_into_curves(candidate_data['tests'])
    
    # Find matching curves
    matching_keys = set(baseline_curves.keys()) & set(candidate_curves.keys())
    print(f"   Found {len(matching_keys)} matching curves")
    
    # Analyze each curve
    print("\n🔬 Analyzing curves for regressions...")
    analyses = []
    
    for key in sorted(matching_keys):
        endpoint, metric_type, threads, scenario = key
        
        analysis = analyze_curve(
            baseline_curves[key],
            candidate_curves[key],
            endpoint, metric_type, threads, scenario,
            config
        )
        
        if analysis:
            analyses.append(analysis)
    
    print(f"   Analyzed {len(analyses)} curves")
    
    # Generate output
    print("\n📝 Generating report...")
    output = generate_output(
        analyses,
        baseline_data['metadata'],
        candidate_data['metadata'],
        config
    )
    
    # Save results
    save_results(output, full_output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 DETECTION SUMMARY")
    print("="*60)
    print(f"Curves analyzed: {output['summary']['curves_analyzed']}")
    print(f"Regressions found: {output['summary']['regressions_detected']}")
    print(f"\nBy Severity:")
    print(f"  🔴 Critical: {output['summary']['by_severity']['critical']}")
    print(f"  🟠 Severe:   {output['summary']['by_severity']['severe']}")
    print(f"  🟡 Moderate: {output['summary']['by_severity']['moderate']}")
    print(f"\nBy Confidence:")
    print(f"  ✅ High:   {output['summary']['by_confidence']['high']}")
    print(f"  ⚠️  Medium: {output['summary']['by_confidence']['medium']}")
    print(f"  ⚡ Low:    {output['summary']['by_confidence']['low']}")
    print("="*60)
    
    # Return non-zero if critical regressions found
    if output['summary']['by_severity']['critical'] > 0:
        print("\n⚠️  CRITICAL REGRESSIONS DETECTED!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
