#!/usr/bin/env python3
"""
All-in-One Performance Analysis & AI Report Generator

This script combines both steps:
1. Runs perfreg_v2.py to detect regressions
2. Generates AI report from the results

Usage:
    python llm/analyze_and_report.py baseline.json candidate.json
    
Benefits:
- One command does everything
- Automatic pipeline
- Both JSON results and AI report
- No intermediate steps needed

Author: 2025
"""

import json
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import requests


# Import the OllamaClient from llm_report_generator
class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9
    ) -> str:
        """Generate text using Ollama API."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": 4000
            }
        }
        
        if stream:
            return self._generate_streaming(url, payload)
        else:
            return self._generate_blocking(url, payload)
    
    def _generate_streaming(self, url: str, payload: Dict) -> str:
        """Generate with streaming output."""
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    text = chunk["response"]
                    print(text, end="", flush=True)
                    full_response += text
                    
                if chunk.get("done"):
                    break
        
        print()
        return full_response
    
    def _generate_blocking(self, url: str, payload: Dict) -> str:
        """Generate without streaming."""
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    
    def list_models(self) -> list:
        """List available models."""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        return [model["name"] for model in response.json()["models"]]


def run_detection(
    baseline_path: str,
    candidate_path: str,
    output_dir: str = "reports/v2_analysis",
    **kwargs
) -> Path:
    """
    Run perfreg_v2.py to detect regressions.
    
    Args:
        baseline_path: Path to baseline JSON
        candidate_path: Path to candidate JSON
        output_dir: Output directory for results
        **kwargs: Additional arguments for perfreg_v2.py
        
    Returns:
        Path to result_v2.json file
    """
    print("=" * 80)
    print("📊 STEP 1: Running Regression Detection")
    print("=" * 80)
    
    # Build command - use same Python interpreter as current script
    cmd = [
        sys.executable,  # Use the same Python that's running this script
        "scripts/perfreg_v2.py",
        baseline_path,
        candidate_path,
        "--out", output_dir
    ]
    
    # Add optional parameters
    if kwargs.get('min_window'):
        cmd.extend(["--min-window", str(kwargs['min_window'])])
    if kwargs.get('min_degradation'):
        cmd.extend(["--min-degradation", str(kwargs['min_degradation'])])
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run the detection script (Python 3.6 compatible)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Show output
    print(result.stdout)
    
    # Exit code 1 is OK (means critical regressions found)
    # Only fail on other errors or if stderr has real errors
    if result.returncode not in [0, 1]:
        print("❌ Detection script failed!")
        print(result.stderr)
        sys.exit(1)
    
    # Find the result file
    output_path = Path(output_dir)
    result_files = list(output_path.glob("*/result_v2.json"))
    
    if not result_files:
        # Try direct path
        direct_path = output_path / "result_v2.json"
        if direct_path.exists():
            return direct_path
        print("❌ Could not find result_v2.json!")
        sys.exit(1)
    
    # Return the most recent one
    result_file = sorted(result_files, key=lambda p: p.stat().st_mtime)[-1]
    
    print(f"✅ Detection complete!")
    print(f"📄 Results saved to: {result_file}")
    print()
    
    return result_file


def create_report_prompt(script_results: Dict[str, Any]) -> str:
    """Create analysis prompt for LLM."""
    summary = script_results['summary']
    regressions = script_results['regressions']
    config = script_results.get('config', {})
    metadata = script_results.get('metadata', {})
    
    run1 = metadata.get('run1', {})
    run2 = metadata.get('run2', {})
    
    prompt = f"""You are a senior performance engineering expert analyzing regression detection results.

## Context
A performance regression detection script analyzed two test runs:
- **Baseline (Run 1)**: Job {run1.get('id', 'N/A')}, version {run1.get('product_ver', 'N/A')} - {run1.get('tests_total', 'N/A')} tests
- **Candidate (Run 2)**: Job {run2.get('id', 'N/A')}, version {run2.get('product_ver', 'N/A')} - {run2.get('tests_total', 'N/A')} tests

## Detection Methodology
The script used **visual pattern recognition** with sliding windows:
- **Method**: {script_results['detection_method']}
- **Min window size**: {config.get('min_window_size', 3)} consecutive points
- **Degradation threshold**: ≥{config.get('min_degradation_pct', 15)}% change
- **Consistency requirement**: {int(config.get('consistency_threshold', 0.7) * 100)}% of points must degrade

## Results Summary
- **Curves analyzed**: {summary['curves_analyzed']}
- **Regressions detected**: {summary['regressions_detected']}
  - 🔴 Critical (≥100%): {summary['by_severity']['critical']}
  - 🟠 Severe (50-99%): {summary['by_severity']['severe']}
  - 🟡 Moderate (15-49%): {summary['by_severity']['moderate']}

**Confidence distribution**:
- High: {summary['by_confidence']['high']}
- Medium: {summary['by_confidence']['medium']}
- Low: {summary['by_confidence']['low']}

## Detected Regressions
```json
{json.dumps(regressions[:15], indent=2)}
```

{f"... and {len(regressions) - 15} more regressions" if len(regressions) > 15 else ""}

## Your Task
Generate a comprehensive, executive-level performance report with:

### 1. Executive Summary (3-4 sentences)
- Overall assessment
- Most critical finding
- Bottom-line recommendation (BLOCK / INVESTIGATE / ACCEPTABLE)

### 2. Critical Issues (if any)
For each critical regression:
- **Endpoint**: What's affected
- **Impact**: Specific degradation % and load range
- **User Impact**: How users will experience this
- **Root Causes**: Top 2-3 hypotheses
- **Priority**: Immediate / High / Medium

### 3. Severe Issues (if any)
Brief summary of severe regressions

### 4. Risk Assessment
- **Production Impact**: 🔴 HIGH / 🟠 MEDIUM / 🟢 LOW
- **User Experience**: Description
- **Release Recommendation**: ✅ APPROVE / ⚠️ APPROVE WITH MONITORING / ❌ BLOCK
- **Investigation Urgency**: Immediate / This Sprint / Monitor

### 5. Prioritized Action Items
Numbered list of what to do (in priority order)

### 6. Technical Deep-Dive
For the worst 3 regressions:
- Load range analysis
- Consistency assessment
- Pattern description
- Investigation hints

### 7. Patterns & Trends
- Common issues across regressions
- Affected components/metrics
- Architectural insights

### 8. Recommendations
- Short-term mitigation
- Long-term improvements
- Monitoring suggestions

Output in professional markdown format. Be specific and actionable.

Begin your analysis:
"""
    
    return prompt


def generate_ai_report(
    result_file: Path,
    model: str = "qwen2.5-coder:14b",
    temperature: float = 0.2
) -> Path:
    """
    Generate AI report from detection results.
    
    Args:
        result_file: Path to result_v2.json
        model: Ollama model to use
        temperature: LLM temperature
        
    Returns:
        Path to generated report
    """
    print("=" * 80)
    print("🤖 STEP 2: Generating AI Report")
    print("=" * 80)
    
    print(f"📊 Loading results from: {result_file}")
    with open(result_file) as f:
        script_results = json.load(f)
    
    # Initialize Ollama client
    client = OllamaClient()
    
    # Check model availability
    print(f"🤖 Checking for model '{model}'...")
    try:
        available_models = client.list_models()
        if model not in available_models:
            print(f"⚠️  Model '{model}' not found. Available models:")
            for m in available_models[:5]:
                print(f"   - {m}")
            print(f"\n💡 To install: ollama pull {model}")
            sys.exit(1)
    except Exception as e:
        print(f"⚠️  Could not check models: {e}")
        print("   Proceeding anyway...")
    
    print("📝 Creating analysis prompt...")
    prompt = create_report_prompt(script_results)
    
    print(f"\n🧠 Generating report with {model}...")
    print(f"   Temperature: {temperature}")
    print("=" * 80)
    
    report = client.generate(
        model=model,
        prompt=prompt,
        stream=True,
        temperature=temperature
    )
    
    print("=" * 80)
    
    # Save report next to result_v2.json
    report_file = result_file.parent / "llm_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ AI report generated!")
    print(f"📄 Saved to: {report_file}")
    
    return report_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="All-in-one performance analysis and AI report generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python llm/analyze_and_report.py baseline.json candidate.json
  
  # With custom output directory
  python llm/analyze_and_report.py baseline.json candidate.json --output reports/my_analysis
  
  # Use faster model
  python llm/analyze_and_report.py baseline.json candidate.json --model qwen2.5-coder:7b
  
  # Adjust detection sensitivity
  python llm/analyze_and_report.py baseline.json candidate.json --min-degradation 20

Workflow:
  1. Runs perfreg_v2.py to detect regressions (1-5 seconds)
  2. Generates AI report from results (1-2 minutes)
  3. Outputs both JSON and markdown reports
        """
    )
    
    parser.add_argument(
        "baseline",
        help="Path to baseline test JSON"
    )
    parser.add_argument(
        "candidate",
        help="Path to candidate test JSON"
    )
    parser.add_argument(
        "--output",
        default="reports/v2_analysis",
        help="Output directory (default: reports/v2_analysis)"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:14b",
        help="Ollama model for AI report (default: qwen2.5-coder:14b)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature 0.0-1.0 (default: 0.2)"
    )
    parser.add_argument(
        "--min-degradation",
        type=float,
        help="Minimum degradation threshold for detection (default: 15)"
    )
    parser.add_argument(
        "--min-window",
        type=int,
        help="Minimum window size for detection (default: 3)"
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip detection, only generate report from existing results"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.skip_detection:
        if not Path(args.baseline).exists():
            print(f"❌ Error: Baseline file not found: {args.baseline}")
            sys.exit(1)
        if not Path(args.candidate).exists():
            print(f"❌ Error: Candidate file not found: {args.candidate}")
            sys.exit(1)
    
    try:
        # Step 1: Run detection (unless skipped)
        if args.skip_detection:
            print("⏭️  Skipping detection, looking for existing results...")
            output_path = Path(args.output)
            result_files = list(output_path.glob("*/result_v2.json"))
            if not result_files:
                direct_path = output_path / "result_v2.json"
                if not direct_path.exists():
                    print("❌ No existing results found!")
                    sys.exit(1)
                result_file = direct_path
            else:
                result_file = sorted(result_files, key=lambda p: p.stat().st_mtime)[-1]
            print(f"✅ Found results: {result_file}\n")
        else:
            result_file = run_detection(
                args.baseline,
                args.candidate,
                args.output,
                min_window=args.min_window,
                min_degradation=args.min_degradation
            )
        
        # Step 2: Generate AI report
        report_file = generate_ai_report(
            result_file,
            model=args.model,
            temperature=args.temperature
        )
        
        # Summary
        print("\n" + "=" * 80)
        print("🎉 COMPLETE ANALYSIS FINISHED!")
        print("=" * 80)
        print(f"\n📊 Detection Results (JSON): {result_file}")
        print(f"📄 AI Report (Markdown):     {report_file}")
        print(f"\n💡 View report: cat {report_file}")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to Ollama.", file=sys.stderr)
        print("💡 Make sure Ollama is running:", file=sys.stderr)
        print("   ollama serve", file=sys.stderr)
        print("   OR open the Ollama app", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
