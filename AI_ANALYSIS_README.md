# PerfTracker AI Analysis Tool

Automated performance regression detection and AI-powered analysis for PerfTracker jobs.

## 🎯 Overview

This tool provides **fully automated** performance regression analysis for PerfTracker test runs:

1. **Auto-Discovery** - Automatically finds the latest 2 comparable jobs for a given product
2. **Regression Detection** - Analyzes performance curves using statistical methods
3. **AI Report Generation** - Creates comprehensive reports using LLM analysis
4. **Upload to PerfTracker** - Optionally uploads reports as artifacts

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+ with virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate  # On Mac/Linux
# .venv311\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama with qwen2.5-coder model
ollama pull qwen2.5-coder:14b
```

### Basic Usage

#### Auto-Discovery Mode (Recommended)

```bash
# Analyze latest GRPM jobs
python3 perftracker_connector.py --product GRPM

# Analyze latest function-dispatcher jobs
python3 perftracker_connector.py --product function-dispatcher

# With upload to PerfTracker
python3 perftracker_connector.py --product GRPM --upload
```

#### Manual Mode

```bash
# Specify exact job IDs
python3 perftracker_connector.py 85817 86303

# With upload
python3 perftracker_connector.py 85817 86303 --upload
```

## 📊 What It Does

### 1. Auto-Discovery Algorithm

```
1. Query PerfTracker API for all jobs matching product name
2. Filter out "local-run" jobs
3. Group by hardware (bg-f1, bg-g9, etc.)
4. For each hardware group:
   - Sort by date (newest first)
   - Find latest 2 with different versions
5. Select the hardware with most recent candidate job
```

**Example Output:**
```
🔍 Auto-discovering latest comparable jobs for GRPM...
Found 77 total jobs for GRPM
After filtering local runs: 58 jobs

Found jobs on 8 different hardware:
  - bg-f1: 3 jobs
  - bg-g9: 7 jobs
  - bg-e6: 12 jobs
  - bg-g14: 14 jobs
  - bg-g10: 17 jobs

✅ Selected jobs on bg-f1:
   Baseline:  Job 85817 - 1.64-4996 (2025-06-23)
   Candidate: Job 86303 - 1.66-5101 (2025-09-06)
```

### 2. Download Jobs

Downloads complete job data with all test results (thousands of tests) from PerfTracker.

### 3. Regression Detection

Runs statistical analysis to detect performance regressions:
- Sliding window analysis
- Median degradation calculation
- Consistency scoring
- Severity classification (Critical/Severe/Moderate)
- Confidence levels (High/Medium/Low)

### 4. AI Report Generation

Uses LLM (qwen2.5-coder:14b) to generate comprehensive reports:
- Executive summary with BLOCK/APPROVE recommendation
- Critical issues ranked by severity
- Root cause analysis
- Risk assessment
- Prioritized action items
- Technical deep-dive

### 5. Upload to PerfTracker (Optional)

Uploads the AI-generated report as an artifact linked to the candidate job for team visibility.

## 📁 Output Structure

```
reports/
├── GRPM/
│   └── 85817 vs 86303/
│       ├── result_v2.json      # Detection results (raw data)
│       └── llm_report.md        # AI-generated report
│
└── function-dispatcher/
    └── 86333 vs 86440/
        ├── result_v2.json
        └── llm_report.md

data/
├── job-85817.json              # Downloaded baseline job
└── job-86303.json              # Downloaded candidate job
```

## 🔧 Configuration

### Command-Line Options

```
positional arguments:
  baseline_id           Baseline job ID (optional with --product)
  candidate_id          Candidate job ID (optional with --product)

options:
  --product {GRPM,function-dispatcher}
                        Product name for auto-discovery
  --project PROJECT     Project ID in PerfTracker (default: 1)
  --url URL             PerfTracker base URL
  --output-dir DIR      Directory for downloaded JSONs (default: data)
  --upload              Upload AI report to PerfTracker
```

### Supported Products

Currently configured for:
- **GRPM** - Policy management API
- **function-dispatcher** - Function dispatcher API

To add more products, update the `choices` in `perftracker_connector.py`:

```python
parser.add_argument(
    "--product",
    choices=["GRPM", "function-dispatcher", "your-product"],
    ...
)
```

## 🔍 How Auto-Discovery Works

### Selection Criteria

1. ✅ **Same Product** - Matches product name (e.g., "GRPM")
2. ✅ **Same Hardware** - Jobs must run on same hardware (e.g., bg-f1)
3. ✅ **Different Versions** - Ensures meaningful comparison (1.64-4996 vs 1.66-5101)
4. ✅ **Latest 2 Runs** - Selects most recent pair
5. ✅ **Excludes Local Runs** - Filters out developer test runs

### Hardware Examples

The tool automatically groups jobs by hardware:
- `bg-f1` - Specific test server
- `bg-g9` - Another test server
- `bg-e6` - Yet another server
- etc.

Each hardware may have different performance characteristics, so comparisons are only made within the same hardware group.

## 📈 Example Workflow

### Complete Auto-Analysis

```bash
$ python3 perftracker_connector.py --product GRPM --upload

# Output:
🔍 Auto-discovering latest comparable jobs for GRPM...
✅ Selected jobs on bg-f1:
   Baseline:  Job 85817 - 1.64-4996
   Candidate: Job 86303 - 1.66-5101

📥 Downloading baseline job 85817... ✅ (8238 tests)
📥 Downloading candidate job 86303... ✅ (8216 tests)
🔍 Running regression detection...
🤖 Generating AI report...

📊 Summary:
   - Curves analyzed:     152
   - Regressions found:   135
     • Critical: 29
     • Severe:   43
     • Moderate: 63

📤 Uploading report to PerfTracker... ✅
   View at: https://pt.perf.corp.acronis.com/1/artifact_content/xxx

🎉 COMPLETE!
```

## 📝 Report Example

The AI generates detailed markdown reports like:

```markdown
# Performance Regression Report

## Executive Summary
**Bottom-line recommendation**: BLOCK the release

## Critical Issues

### Endpoint: PUT /api/policy_management/v2/drafts/{id}/settings
- **Impact**: Degradation of 393.03% during load range 43000-46000
- **User Impact**: Severe delays and potential timeouts
- **Root Causes**:
  - Inefficient resource allocation
  - Potential memory leaks
- **Priority**: Immediate

## Risk Assessment
- **Production Impact**: 🔴 HIGH
- **Release Recommendation**: ❌ BLOCK

## Prioritized Action Items
1. Identify root cause of PUT endpoint regression
2. Optimize database queries
3. ...
```

## 🔄 CI/CD Integration

### Cron Job Example

```bash
# Daily GRPM analysis at 2 AM
0 2 * * * cd /path/to/project && \
  ./.venv311/bin/python perftracker_connector.py --product GRPM --upload
```

### Jenkins/GitLab CI

```yaml
performance-analysis:
  script:
    - python3 perftracker_connector.py --product GRPM --upload
  only:
    - schedules
```

## 🐛 Troubleshooting

### SSL Certificate Errors

The tool disables SSL verification for internal corporate servers. If you see SSL errors, this is expected and handled.

### No Comparable Jobs Found

```
❌ Could not find 2 jobs with different versions on same hardware
```

**Solutions:**
1. Check if there are enough test runs in PerfTracker
2. Verify product name matches exactly (case-sensitive)
3. Ensure at least 2 runs exist on same hardware with different versions

### LLM Model Not Found

```
❌ Model 'qwen2.5-coder:14b' not found
```

**Solution:**
```bash
ollama pull qwen2.5-coder:14b
```

## 📚 File Structure

### Core Files

```
perftracker_connector.py    # Main orchestrator
├── PerfTrackerConnector    # API client
│   ├── find_latest_comparable_jobs()  # Auto-discovery
│   ├── get_job_json()                  # Download jobs
│   ├── upload_report()                 # Upload to PT
│   └── download_and_analyze()          # Full pipeline

llm/
├── analyze_and_report.py   # Analysis pipeline
├── llm_analysis.py         # LLM report generator

scripts/
└── perfreg_v2.py           # Regression detection engine
```

## 🔐 Authentication

**No authentication required!** Your PerfTracker instance allows:
- ✅ **GET** (read) - Open access
- ✅ **POST** (write) - Open access for artifacts

This is configured by the commented-out `@login_required` decorators in PerfTracker.

## 🎯 Key Features

### ✅ Fully Automated
- No manual job selection
- No manual analysis
- No manual report writing

### ✅ Smart Discovery
- Filters local runs
- Groups by hardware
- Ensures version differences

### ✅ Production-Ready
- SSL handling
- Error recovery
- Progress reporting

### ✅ Team Collaboration
- Uploads reports to PerfTracker
- Links to candidate job
- Markdown format for easy reading

## 📞 Support

For issues or questions:
1. Check `--help` for command options
2. Review logs for error messages
3. Verify PerfTracker API access

## 🚀 Future Enhancements

Potential improvements:
- [ ] Email notifications on critical regressions
- [ ] Slack/Teams integration
- [ ] Multiple product support in one run
- [ ] Historical trend analysis
- [ ] Custom severity thresholds
- [ ] Web dashboard

---

**Happy Testing!** 🎉
