Ptracker Hardware Validation Analysis
A comprehensive Python toolkit for analyzing hardware validation logs using transition-focused pattern detection, anomaly identification, and performance monitoring.
Overview
This project provides AI-powered analysis tools for ptracker hardware validation logs, focusing on state transitions rather than raw register values. It identifies anomalies by comparing failing test patterns against reference models built from passing tests.
Key Features
•	Transition-Focused Analysis: Analyzes state changes (0x0→0x1→0x0) instead of raw register values
•	Performance Monitoring: Comprehensive timing analysis with CSV export
•	Anomaly Detection: Machine learning-based detection using Isolation Forest
•	False Positive Filtering: Advanced filtering to reduce noise in analysis results
•	Reference Model Building: Automated baseline creation from passing test logs
•	Comprehensive Logging: Structured logging with performance tracking
•	Divergence Detection: Identifies where failing tests deviate from expected patterns

Project Structure
ptracker-analysis/
├── src/
│   ├── transition_analysis.py      # Main transition-focused analysis script
│   ├── feature_extractor.py        # Optimized feature extraction (10x faster)
│   ├── reference_model.py          # Reference model building and anomaly detection
│   ├── enhanced_logging.py         # Comprehensive logging utilities
│   └── analysis_components/        # Additional analysis modules
├── tests/                          # Test files and validation scripts
├── docs/                          # Documentation and examples
├── requirements.txt               # Python dependencies
└── README.md                     # This file

Installation
Prerequisites
•	Python 3.8 or higher
•	Required packages listed in requirements.txt
|
Setup
1.	Clone the repository:
bash
git clone <repository-url>
cd ptracker-analysis
2.	Install dependencies:
bash
pip install -r requirements.txt
3.	Verify installation:
bash
python src/transition_analysis.py --help
Usage
Basic Transition Analysis
Run transition-focused analysis on your log files:
bash
python src/transition_analysis.py \
    --passing-dir ./logs/passing \
    --failing-dir ./logs/failing \
    --output-dir ./results \
    --log-level INFO

Parameters
Required Parameter	Description	
--passing-dir	Directory containing passing test logs 
--failing-dir	Directory containing failing test logs  
Optional Parameter	Description
--output-dir	Output directory for analysis results	No	transition_output
--log-level	Logging verbosity (DEBUG, INFO, WARNING, ERROR)	No	INFO
--max-files	Limit number of files for testing	No	All files
--save-model – add this knob in order to save the reference model extracted during the run ​
--reference-model - Path to saved reference model (where to save the new reference model/ load the existing one)​
--load-model – run with an existing  reference model extracted during previous runs​
--generate-debug-report - Enable comprehensive debug report generation with detailed register sequence analysis and root cause suggestions ​

Example Analysis Workflow
python
from src.enhanced_logging import setup_logger
from src.reference_model import PtrackerReferenceModel
from src.feature_extractor import OptimizedPtrackerFeatureExtractor

# Setup enhanced logging
logger = setup_logger("analysis", "./logs", enable_performance=True)

# Build reference model from passing tests
reference_model = PtrackerReferenceModel(logger)
reference_data = reference_model.load_reference_logs("./passing_logs")
reference_model.build_reference_model(reference_data)

# Extract features from failing tests
extractor = OptimizedPtrackerFeatureExtractor(logger, reference_model)
failing_features = extractor.extract_features(failing_test_data)

# Detect anomalies
anomalies = reference_model.detect_anomalies(failing_features)
Output Files
The analysis generates several output files:
Analysis Results
•	transition_summary_YYYYMMDD_HHMMSS.json - Summary statistics and metadata
•	transition_hints_YYYYMMDD_HHMMSS.txt - Human-readable debugging hints
•	analysis_metadata_YYYYMMDD_HHMMSS.json - Detailed analysis metadata
Performance Analysis
•	timing_analysis_YYYYMMDD_HHMMSS.csv - Phase-by-phase timing breakdown
•	transition_analysis_YYYYMMDD_HHMMSS.log - Detailed execution log
Results Format
Transition Summary Example:
json
{
  "timestamp": "20241210_143022",
  "analysis_type": "transition_focused", 
  "total_failing_tests": 45,
  "tests_with_transitions": 42,
  "timing_analysis": {
    "total_time_seconds": 23.45,
    "phase_timings": {
      "extract_passing_transitions": 8.2,
      "build_reference_model": 2.1,
      "extract_failing_transitions": 9.8,
      "analyze_failing_transitions": 3.35
    }
  }
}
Transition Hints Example:
TRANSITION PATTERN #1
   Register: sst_manager.hwp_enabled
   Occurs in: 12 tests
   Severity: 0.950

   Expected transitions: 0x0→0x1
   Actual transitions:   0x0→0x1→0x0

   Issue: Has 1 extra transitions
   Category: extra_transitions

Architecture
Core Components
1. TransitionLogProcessor (transition_analysis.py)
•	Parses ptracker logs and extracts state transitions
•	Builds reference transition models from passing tests
•	Compares failing test transitions against reference patterns
•	Generates transition-focused error analysis
2. OptimizedPtrackerFeatureExtractor (feature_extractor.py)
•	High-performance feature extraction (10x faster than previous versions)
•	Focuses on essential register sequence features
•	Filters out meaningless constant-value sequences
•	Includes false positive pattern detection
3. PtrackerReferenceModel (reference_model.py)
•	Builds baseline models from passing test data
•	Uses Isolation Forest for anomaly detection
•	Supports temporal pattern analysis
•	Provides model persistence and loading
4. Enhanced Logging (enhanced_logging.py)
•	Structured logging with performance tracking
•	Phase-based timing analysis
•	Color-coded console output
•	Comprehensive analysis reporting

Analysis Phases
1.	Log Parsing: Extract register access patterns from compressed logs
2.	Transition Extraction: Convert register sequences to state transitions
3.	Reference Model Building: Create baseline patterns from passing tests
4.	Anomaly Detection: Compare failing tests against reference model
5.	Pattern Analysis: Categorize and prioritize detected anomalies
Transition Types
The system categorizes different types of transition anomalies:
Type	Description	Severity
identical_transitions	Transitions match exactly (filtered as false positive)	None
missing_all_transitions	No state changes detected	Critical
incomplete_transitions	State machine started but stopped early	High
extra_transitions	State machine continued beyond expected endpoint	Medium
different_transition_path	Completely different state sequence	Critical
unexpected_transitions	Transitions not seen in reference data	High
Performance Optimization
The system includes several performance optimizations:
•	Simplified Feature Extraction: Removed complex temporal and FSM analysis
•	False Positive Filtering: Eliminates meaningless constant-value sequences
•	Memory-Efficient Processing: Processes files individually to manage memory
•	Timing Analysis: Comprehensive performance monitoring with CSV export

Configuration
False Positive Patterns
The system automatically filters known false positive patterns:
python
false_positive_patterns = [
    'sst_manager.hwp_enabled',
    'sst_manager.configs_done', 
    'initialized_flags',
    'target_values'
]

Minimum Sequence Requirements
•	Minimum sequence length: 3 register accesses
•	Minimum change frequency: 5% of values must change
•	Minimum unique values: Must have more than 1 unique value
Dependencies
Core requirements:
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
psutil>=5.8.0
Troubleshooting
Common Issues
No transitions found in logs:
•	Verify log file format matches expected ptracker patterns
•	Check that files contain actual register state changes
•	Ensure minimum sequence length requirements are met
Performance issues:
•	Use --max-files parameter to limit file count during testing
•	Check memory usage with larger datasets
•	Review timing analysis CSV for bottlenecks
False positives in results:
•	Review and update false positive pattern filters
•	Adjust minimum change frequency thresholds
•	Examine transition hints for pattern validation

Debug Mode
Enable detailed debugging:
bash
python src/transition_analysis.py \
    --passing-dir ./passing \
    --failing-dir ./failing \
    --log-level DEBUG \
    --output-dir ./debug_results
Contributing
When contributing to this project:
1.	Maintain the transition-focused analysis approach
2.	Ensure performance optimizations are preserved
3.	Add comprehensive logging for new features
4.	Include timing analysis for new processing phases
5.	Update false positive filters as needed
License
This project is licensed under the MIT License - see the LICENSE file for details.
Support
For support and questions:
•	Check the generated transition hints files for debugging guidance
•	Review the performance analysis CSV files for optimization opportunities
•	Examine the detailed log files for execution details
•	Open an issue with sample log files and error messages
Roadmap
•	Support for additional ptracker log formats
•	Real-time log analysis capabilities
•	Web-based analysis dashboard
•	Enhanced visualization of transition patterns
•	Integration with hardware test frameworks

