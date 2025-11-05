#!/usr/bin/env python3
"""
LLM-powered performance regression analysis.
Uses local Ollama to analyze performance test JSONs and generate reports.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List


def call_ollama(prompt: str, model: str = "qwen2.5-coder:14b") -> str:
    """
    Call local Ollama LLM with a prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        model: Model to use (default: qwen2.5-coder:14b)
        
    Returns:
        LLM response as string
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr}")
            
        return result.stdout.strip()
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("LLM took too long to respond (>5 minutes)")
    except FileNotFoundError:
        raise RuntimeError(
            "Ollama not found. Please install it: https://ollama.ai"
        )


def load_test_data(baseline_path: str, candidate_path: str) -> tuple:
    """Load baseline and candidate test JSONs."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(candidate_path) as f:
        candidate = json.load(f)
    return baseline, candidate


def prepare_analysis_prompt(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    focus_endpoints: List[str] = None
) -> str:
    """
    Prepare a prompt for the LLM to analyze performance data.
    
    Args:
        baseline: Baseline test results JSON
        candidate: Candidate test results JSON
        focus_endpoints: Optional list of endpoints to focus on
        
    Returns:
        Formatted prompt string
    """
    
    # Extract metadata
    baseline_meta = {
        "id": baseline.get("job", {}).get("id"),
        "version": baseline.get("job", {}).get("productVersion"),
        "tests_total": len(baseline.get("tests", []))
    }
    
    candidate_meta = {
        "id": candidate.get("job", {}).get("id"),
        "version": candidate.get("job", {}).get("productVersion"),
        "tests_total": len(candidate.get("tests", []))
    }
    
    # Sample some test data (to keep prompt size manageable)
    baseline_tests = baseline.get("tests", [])[:50]
    candidate_tests = candidate.get("tests", [])[:50]
    
    # Build the prompt
    prompt = f"""You are a performance analysis expert. Analyze these two performance test runs and identify regressions.

## Task
Compare baseline vs candidate performance tests and identify:
1. **Critical regressions** (>100% degradation)
2. **Severe regressions** (50-100% degradation)  
3. **Moderate regressions** (15-50% degradation)

Focus on:
- Sustained degradation patterns (multiple consecutive points)
- Endpoints with consistent performance drops
- High-impact APIs (latency spikes or throughput drops)

## Baseline Run
- Job ID: {baseline_meta['id']}
- Version: {baseline_meta['version']}
- Total tests: {baseline_meta['tests_total']}

## Candidate Run
- Job ID: {candidate_meta['id']}
- Version: {candidate_meta['version']}
- Total tests: {candidate_meta['tests_total']}

## Test Data Sample

### Baseline Tests (sample):
```json
{json.dumps(baseline_tests[:10], indent=2)}
```

### Candidate Tests (sample):
```json
{json.dumps(candidate_tests[:10], indent=2)}
```

## Instructions

1. **Compare corresponding tests** by matching:
   - Same endpoint (tag)
   - Same category (load level)
   - Same metric type (latency/rate)
   
2. **Calculate degradation**:
   - For latency: (candidate - baseline) / baseline * 100
   - For rate: -(candidate - baseline) / baseline * 100 (inverted)
   
3. **Identify patterns**:
   - Look for 3+ consecutive degraded points
   - Require >70% of points in a window to be degraded
   - Flag if degradation >15%

4. **Output Format**:
```markdown
# Performance Regression Report

## Summary
- Total regressions found: X
- Critical: X | Severe: X | Moderate: X

## Critical Regressions (>100% degradation)

### 1. [Endpoint Name]
- **Metric**: latency/rate
- **Threads**: X
- **Worst window**: XYZ load range
- **Degradation**: X%
- **Impact**: [Brief description]

## Severe Regressions (50-100% degradation)
[Same format]

## Moderate Regressions (15-50% degradation)
[Same format]

## Recommendations
[Your analysis and suggestions]
```

Please analyze and generate the report.
"""
    
    return prompt


def analyze_with_llm(
    baseline_path: str,
    candidate_path: str,
    model: str = "qwen2.5-coder:14b",
    output_file: str = None
) -> str:
    """
    Run LLM-powered performance analysis.
    
    Args:
        baseline_path: Path to baseline JSON
        candidate_path: Path to candidate JSON
        model: Ollama model to use
        output_file: Optional output file path
        
    Returns:
        Analysis report as string
    """
    print("📊 Loading test data...")
    baseline, candidate = load_test_data(baseline_path, candidate_path)
    
    print("🤖 Preparing analysis prompt...")
    prompt = prepare_analysis_prompt(baseline, candidate)
    
    print(f"🧠 Running LLM analysis with {model}...")
    print("   (This may take 2-5 minutes depending on your hardware)")
    
    report = call_ollama(prompt, model)
    
    if output_file:
        print(f"💾 Saving report to {output_file}...")
        with open(output_file, 'w') as f:
            f.write(report)
    
    return report


def compare_with_script_results(
    llm_report: str,
    script_results_path: str
) -> str:
    """
    Compare LLM findings with the Python script results.
    
    Args:
        llm_report: LLM-generated report
        script_results_path: Path to result_v2.json from our script
        
    Returns:
        Comparison analysis
    """
    with open(script_results_path) as f:
        script_results = json.load(f)
    
    prompt = f"""Compare these two performance regression analyses:

## Analysis 1: Rule-based Python Script
```json
{json.dumps(script_results['summary'], indent=2)}
```

Total regressions: {script_results['summary']['regressions_detected']}
- Critical: {script_results['summary']['by_severity']['critical']}
- Severe: {script_results['summary']['by_severity']['severe']}
- Moderate: {script_results['summary']['by_severity']['moderate']}

## Analysis 2: LLM Analysis
{llm_report}

## Task
1. Identify agreements (regressions both found)
2. Identify disagreements (only one found)
3. Explain possible reasons for differences
4. Which analysis would you trust more and why?

Provide a concise comparison report.
"""
    
    return call_ollama(prompt)


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python llm_analysis.py <baseline.json> <candidate.json> [model]")
        print()
        print("Examples:")
        print("  python llm_analysis.py job-82221-GRPM_API.json job-82443-GRPM_API.json")
        print("  python llm_analysis.py job-82221-GRPM_API.json job-82443-GRPM_API.json qwen2.5-coder:7b")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    candidate_path = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5-coder:14b"
    
    # Generate output filename
    baseline_id = Path(baseline_path).stem.split('-')[1]
    candidate_id = Path(candidate_path).stem.split('-')[1]
    output_file = f"reports/llm_analysis/{baseline_id}_vs_{candidate_id}_llm_report.md"
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run analysis
        report = analyze_with_llm(
            baseline_path,
            candidate_path,
            model=model,
            output_file=output_file
        )
        
        print("\n" + "="*80)
        print("✅ LLM ANALYSIS COMPLETE")
        print("="*80)
        print()
        print(report)
        print()
        print(f"📄 Full report saved to: {output_file}")
        
        # Optional: Compare with script results if available
        script_results = f"reports/v2_analysis/{baseline_id}_GRPM vs {candidate_id}_GRPM/result_v2.json"
        if Path(script_results).exists():
            print()
            print("🔍 Comparing with rule-based script results...")
            comparison = compare_with_script_results(report, script_results)
            
            comparison_file = output_file.replace('.md', '_comparison.md')
            with open(comparison_file, 'w') as f:
                f.write(comparison)
            
            print(f"📊 Comparison saved to: {comparison_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
