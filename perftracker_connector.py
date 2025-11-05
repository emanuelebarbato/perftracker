#!/usr/bin/env python3
"""
PerfTracker Connector - Download jobs and run AI analysis

No authentication needed! Your PerfTracker instance is open access.

Usage:
    python3 perftracker_connector.py 82221 82443
    python3 perftracker_connector.py 82221 82443 --project 2
"""

import requests
import json
import sys
import subprocess
from pathlib import Path
import argparse
import urllib3
import uuid as uuid_lib

# Disable SSL warnings for internal corporate servers
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PerfTrackerConnector:
    """Simple connector to PerfTracker for downloading jobs and running analysis."""
    
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        # No authentication needed!
    
    def find_latest_comparable_jobs(self, project_id, product_name):
        """
        Find the latest 2 comparable jobs for a given product.
        
        Algorithm:
        1. Get all jobs for this product
        2. Filter out "local" runs
        3. Group by hardware (env_node)
        4. For each hardware, find latest 2 with different versions
        5. Return the pair with most recent candidate
        
        Returns: (baseline_id, candidate_id, hardware_name)
        """
        print(f"\n🔍 Auto-discovering latest comparable jobs for {product_name}...")
        print("-" * 70)
        
        # Query API for jobs matching this product
        url = f"{self.base_url}/api/v1.0/{project_id}/job/"
        params = {
            'search[value]': product_name,
            'length': 200,  # Get enough jobs to find matches
            'start': 0
        }
        
        try:
            response = self.session.get(url, params=params, verify=False, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ Failed to query jobs: {e}")
            return None, None, None
        
        all_jobs = data.get('data', [])
        print(f"Found {len(all_jobs)} total jobs for {product_name}")
        
        # Filter out local runs
        real_jobs = [
            job for job in all_jobs 
            if job.get('env_node') 
            and len(job['env_node']) > 0
            and 'local' not in job['env_node'][0].get('name', '').lower()
        ]
        print(f"After filtering local runs: {len(real_jobs)} jobs")
        
        if len(real_jobs) < 2:
            print(f"❌ Not enough non-local jobs found")
            return None, None, None
        
        # Group by hardware
        from collections import defaultdict
        hardware_groups = defaultdict(list)
        
        for job in real_jobs:
            hw_name = job['env_node'][0].get('display_name', job['env_node'][0].get('name'))
            hardware_groups[hw_name].append(job)
        
        print(f"\nFound jobs on {len(hardware_groups)} different hardware:")
        for hw, jobs in hardware_groups.items():
            print(f"  - {hw}: {len(jobs)} jobs")
        
        # For each hardware, find latest 2 with different versions
        best_pair = None
        best_candidate_date = None
        
        for hw_name, jobs in hardware_groups.items():
            if len(jobs) < 2:
                continue
            
            # Sort by end date (latest first)
            jobs_sorted = sorted(jobs, key=lambda j: j['end'], reverse=True)
            
            # Find latest 2 with different versions
            candidate = jobs_sorted[0]
            baseline = None
            
            for job in jobs_sorted[1:]:
                if job['product_ver'] != candidate['product_ver']:
                    baseline = job
                    break
            
            if baseline:
                # Keep track of the most recent pair overall
                if best_candidate_date is None or candidate['end'] > best_candidate_date:
                    best_pair = (baseline, candidate, hw_name)
                    best_candidate_date = candidate['end']
        
        if not best_pair:
            print(f"\n❌ Could not find 2 jobs with different versions on same hardware")
            return None, None, None
        
        baseline, candidate, hw_name = best_pair
        
        print(f"\n✅ Selected jobs on {hw_name}:")
        print(f"   Baseline:  Job {baseline['id']} - {baseline['product_ver']} ({baseline['end'][:10]})")
        print(f"   Candidate: Job {candidate['id']} - {candidate['product_ver']} ({candidate['end'][:10]})")
        
        return baseline['id'], candidate['id'], hw_name
    
    def get_job_json(self, project_id, job_id):
        """Get job data in the same format as your current JSON files."""
        # Use the web UI's JSON download endpoint, not the REST API
        # This gives us the complete job data with all tests
        url = f"{self.base_url}/{project_id}/job/{job_id}?as_json=1"
        print(f"📥 Fetching job {job_id} from {url}...")
        
        try:
            # Disable SSL verification for internal corporate servers
            response = self.session.get(url, timeout=30, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch job {job_id}: {e}")
            raise
        
        job_data = response.json()
        print(f"✅ Fetched job {job_id}: {job_data.get('title', 'Unknown')} "
              f"({job_data.get('tests_total', 0)} tests)")
        
        return job_data
    
    def upload_report(self, project_id, job_uuid, report_path, description="AI-generated performance regression analysis"):
        """Upload AI report as artifact linked to a job."""
        if not Path(report_path).exists():
            print(f"❌ Report file not found: {report_path}")
            return None
        
        # Generate UUID for this artifact
        artifact_uuid = str(uuid_lib.uuid4())
        
        # Upload URL
        url = f"{self.base_url}/api/v1.0/{project_id}/artifact/{artifact_uuid}"
        
        print(f"📤 Uploading report to PerfTracker...")
        print(f"   Artifact UUID: {artifact_uuid}")
        print(f"   Linked to job: {job_uuid}")
        
        # Read report file
        with open(report_path, 'rb') as f:
            file_content = f.read()
        
        filename = Path(report_path).name
        
        # Prepare multipart form data
        files = {
            'file': (filename, file_content, 'text/markdown')
        }
        
        data = {
            'filename': filename,
            'description': description,
            'links': json.dumps([job_uuid]),  # Link to job
            'ttl_days': '180',  # Keep for 180 days
            'inline': 'true',   # Show inline in browser
            'compression': 'false'
        }
        
        try:
            response = self.session.post(url, files=files, data=data, verify=False, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ Report uploaded successfully!")
            print(f"   View at: {self.base_url}/1/artifact_content/{artifact_uuid}")
            
            return artifact_uuid
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to upload report: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text[:500]}")
            return None
    
    def download_and_analyze(self, project_id, baseline_id, candidate_id, output_dir="data", upload_report=False, product_name=None):
        """Download jobs and run AI analysis automatically."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("🚀 PERFTRACKER AUTOMATED ANALYSIS")
        print("=" * 80)
        
        # Download baseline job
        print(f"\n📥 Step 1: Downloading baseline job {baseline_id}...")
        try:
            baseline_data = self.get_job_json(project_id, baseline_id)
            baseline_file = output_path / f"job-{baseline_id}.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            print(f"💾 Saved to: {baseline_file}")
            # Get product name from job if not provided
            if not product_name:
                product_name = baseline_data.get('product_name', 'unknown')
        except Exception as e:
            print(f"❌ Failed to download baseline job: {e}")
            return None
        
        # Download candidate job
        print(f"\n📥 Step 2: Downloading candidate job {candidate_id}...")
        try:
            candidate_data = self.get_job_json(project_id, candidate_id)
            candidate_uuid = candidate_data.get('uuid')  # Save UUID for upload linking
            candidate_file = output_path / f"job-{candidate_id}.json"
            with open(candidate_file, 'w') as f:
                json.dump(candidate_data, f, indent=2)
            print(f"💾 Saved to: {candidate_file}")
        except Exception as e:
            print(f"❌ Failed to download candidate job: {e}")
            return None
        
        # Run AI analysis
        print(f"\n🤖 Step 3: Running AI analysis...")
        print("=" * 80)
        
        # Check if venv Python exists
        venv_python = Path(".venv311/bin/python")
        if venv_python.exists():
            python_cmd = str(venv_python)
        else:
            python_cmd = "python3"
        
        # Set output directory based on product name
        # Sanitize product name for filesystem (replace spaces and special chars)
        if product_name:
            safe_product_name = product_name.replace(' ', '_').replace('/', '_')
            reports_dir = f"reports/{safe_product_name}"
        else:
            reports_dir = "reports/v2_analysis"
        
        cmd = [
            python_cmd,
            "llm/analyze_and_report.py",
            str(baseline_file),
            str(candidate_file),
            "--output", reports_dir
        ]
        
        print(f"Running: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd)
        
        if result.returncode not in [0, 1]:  # 0 or 1 are OK
            print(f"\n❌ Analysis failed with exit code {result.returncode}")
            return None
        
        # Find the generated report
        print("\n" + "=" * 80)
        print("📊 Step 4: Locating generated report...")
        
        report_path = Path(reports_dir)
        reports = list(report_path.glob("*/llm_report.md"))
        
        if reports:
            latest_report = sorted(reports, key=lambda p: p.stat().st_mtime)[-1]
            result_json = latest_report.parent / "result_v2.json"
            
            print(f"✅ Report generated successfully!")
            print(f"\n📄 Files created:")
            print(f"   - Detection results: {result_json}")
            print(f"   - AI report:         {latest_report}")
            
            # Show summary
            if result_json.exists():
                with open(result_json) as f:
                    data = json.load(f)
                    summary = data.get('summary', {})
                    print(f"\n📊 Summary:")
                    print(f"   - Curves analyzed:     {summary.get('curves_analyzed', 'N/A')}")
                    print(f"   - Regressions found:   {summary.get('regressions_detected', 'N/A')}")
                    by_severity = summary.get('by_severity', {})
                    print(f"     • Critical: {by_severity.get('critical', 0)}")
                    print(f"     • Severe:   {by_severity.get('severe', 0)}")
                    print(f"     • Moderate: {by_severity.get('moderate', 0)}")
            
            print("\n💡 View report:")
            print(f"   cat {latest_report}")
            print(f"   # or")
            print(f"   code {latest_report}")
            
            # Upload report if requested
            if upload_report and candidate_uuid:
                print("\n" + "=" * 80)
                print("📤 Step 5: Uploading report to PerfTracker...")
                artifact_uuid = self.upload_report(
                    project_id,
                    candidate_uuid,
                    latest_report,
                    f"AI Regression Analysis: Job {baseline_id} vs {candidate_id}"
                )
                if artifact_uuid:
                    print(f"\n🔗 Report linked to candidate job {candidate_id}")
                    print(f"   View job: {self.base_url}/1/job/{candidate_id}")
            
            return latest_report
        else:
            print("⚠️  No report found (analysis may have failed)")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Download jobs from PerfTracker and run AI analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discovery: Latest GRPM jobs (recommended)
  python3 perftracker_connector.py --product GRPM --upload
  
  # Auto-discovery: Latest function-dispatcher jobs
  python3 perftracker_connector.py --product function-dispatcher --upload
  
  # Manual mode: Specify exact job IDs
  python3 perftracker_connector.py 82221 82443 --upload
  
  # Specify different project
  python3 perftracker_connector.py --product GRPM --project 2
  
  # Use different PerfTracker server
  python3 perftracker_connector.py --product GRPM --url https://other-pt.example.com

Note: No authentication required! Your PerfTracker instance is open access.
        """
    )
    
    parser.add_argument(
        "baseline_id",
        nargs='?',
        type=int,
        help="Baseline job ID (e.g., 82221) - optional if using --product"
    )
    parser.add_argument(
        "candidate_id",
        nargs='?',
        type=int,
        help="Candidate job ID (e.g., 82443) - optional if using --product"
    )
    parser.add_argument(
        "--product",
        type=str,
        choices=["GRPM", "function-dispatcher"],
        help="Product name for auto-discovery (GRPM or function-dispatcher)"
    )
    parser.add_argument(
        "--project",
        type=int,
        default=1,
        help="Project ID in PerfTracker (default: 1)"
    )
    parser.add_argument(
        "--url",
        default="https://pt.perf.corp.acronis.com",
        help="PerfTracker base URL (default: https://pt.perf.corp.acronis.com)"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save downloaded JSONs (default: data)"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload AI report back to PerfTracker (links to candidate job)"
    )
    
    args = parser.parse_args()
    
    # Determine baseline and candidate IDs
    product_name = None
    if args.product:
        # Auto-discovery mode
        connector = PerfTrackerConnector(args.url)
        baseline_id, candidate_id, hw_name = connector.find_latest_comparable_jobs(
            args.project,
            args.product
        )
        
        if not baseline_id or not candidate_id:
            print(f"\n❌ Could not find comparable jobs for {args.product}")
            sys.exit(1)
        
        product_name = args.product  # Use the product name from auto-discovery
    elif args.baseline_id and args.candidate_id:
        # Manual mode
        baseline_id = args.baseline_id
        candidate_id = args.candidate_id
        connector = PerfTrackerConnector(args.url)
        # product_name will be detected from job data
    else:
        parser.error("Either specify --product for auto-discovery OR provide baseline_id and candidate_id")
    
    try:
        report = connector.download_and_analyze(
            args.project,
            baseline_id,
            candidate_id,
            args.output_dir,
            upload_report=args.upload,
            product_name=product_name
        )
        
        if report:
            print("\n" + "=" * 80)
            print("🎉 COMPLETE! All steps finished successfully!")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("❌ FAILED! Check errors above.")
            print("=" * 80)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
