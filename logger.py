"""
Enhanced logging utilities for AI Debug Assist Hardware Validation Analysis
Provides comprehensive logging with structured output, performance tracking, and report integration
"""
import os
import logging
import sys
from datetime import datetime
from pathlib import Path
import json
import time
from contextlib import contextmanager


class PtrackerAnalysisLogger:
    """
    Enhanced logger specifically designed for ptracker hardware validation analysis
    Provides structured logging, performance tracking, and analysis-specific features
    """
    
    def __init__(self, name, output_dir=None, level=logging.INFO, enable_performance=True):
        """
        Initialize the enhanced ptracker analysis logger
        
        Parameters:
        name (str): Logger name
        output_dir (str): Directory for log files
        level (int): Logging level
        enable_performance (bool): Enable performance tracking
        """
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else None
        self.level = level
        self.enable_performance = enable_performance
        
        # Create the main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers = []
        
        # Performance tracking
        self.start_time = time.time()
        self.phase_times = {}
        self.current_phase = None
        
        # Analysis metrics
        self.analysis_stats = {
            'files_processed': 0,
            'tests_analyzed': 0,
            'anomalies_detected': 0,
            'features_extracted': 0,
            'errors_encountered': 0
        }
        
        # Setup logging handlers
        self._setup_handlers()
        
        # Log initialization
        self.logger.info(f"Initialized {name} logger with level {logging.getLevelName(level)}")
        if self.output_dir:
            self.logger.info(f"Log files will be saved to: {self.output_dir}")
    
    def _setup_handlers(self):
        """Setup console and file handlers with appropriate formatters"""
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler with color support
        console_handler = EnhancedConsoleHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if output directory is specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Main log file
            log_file = self.output_dir / f"{self.name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # File gets more detail
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Performance log file (if enabled)
            if self.enable_performance:
                perf_log_file = self.output_dir / f"{self.name}_performance_{timestamp}.log"
                self.perf_handler = logging.FileHandler(perf_log_file, encoding='utf-8')
                self.perf_handler.setLevel(logging.INFO)
                perf_formatter = logging.Formatter(
                    '%(asctime)s - PERF - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                self.perf_handler.setFormatter(perf_formatter)
    
    def info(self, message, *args, **kwargs):
        """Log info message with optional formatting"""
        self.logger.info(message, *args, **kwargs)
    
    def debug(self, message, *args, **kwargs):
        """Log debug message with optional formatting"""
        self.logger.debug(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """Log warning message with optional formatting"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """Log error message and increment error counter"""
        self.logger.error(message, *args, **kwargs)
        self.analysis_stats['errors_encountered'] += 1
    
    def critical(self, message, *args, **kwargs):
        """Log critical message and increment error counter"""
        self.logger.critical(message, *args, **kwargs)
        self.analysis_stats['errors_encountered'] += 1
    
    @contextmanager
    def phase(self, phase_name):
        """Context manager for tracking analysis phases with timing"""
        self.start_phase(phase_name)
        try:
            yield self
        finally:
            self.end_phase()
    
    def start_phase(self, phase_name):
        """Start a new analysis phase"""
        if self.current_phase:
            self.end_phase()
        
        self.current_phase = phase_name
        self.phase_times[phase_name] = {'start': time.time()}
        
        self.info(f"🚀 Starting phase: {phase_name}")
        if self.enable_performance:
            self._log_performance(f"PHASE_START: {phase_name}")
    
    def end_phase(self):
        """End the current analysis phase"""
        if not self.current_phase:
            return
        
        end_time = time.time()
        start_time = self.phase_times[self.current_phase]['start']
        duration = end_time - start_time
        
        self.phase_times[self.current_phase]['end'] = end_time
        self.phase_times[self.current_phase]['duration'] = duration
        
        self.info(f"✅ Completed phase: {self.current_phase} (Duration: {duration:.2f}s)")
        if self.enable_performance:
            self._log_performance(f"PHASE_END: {self.current_phase} - Duration: {duration:.2f}s")
        
        self.current_phase = None
    
    def log_file_processing(self, filename, status="processing"):
        """Log file processing status"""
        if status == "processing":
            self.debug(f"📄 Processing file: {filename}")
        elif status == "completed":
            self.debug(f"✅ Completed file: {filename}")
            self.analysis_stats['files_processed'] += 1
        elif status == "failed":
            self.error(f"❌ Failed to process file: {filename}")
        elif status == "skipped":
            self.debug(f"⏭️  Skipped file: {filename}")
    
    def log_test_analysis(self, test_id, status="analyzing"):
        """Log test analysis status"""
        if status == "analyzing":
            self.debug(f"🔍 Analyzing test: {test_id}")
        elif status == "completed":
            self.debug(f"✅ Completed test analysis: {test_id}")
            self.analysis_stats['tests_analyzed'] += 1
        elif status == "anomaly_detected":
            self.info(f"⚠️  Anomaly detected in test: {test_id}")
            self.analysis_stats['anomalies_detected'] += 1
        elif status == "failed":
            self.error(f"❌ Failed to analyze test: {test_id}")
    
    def log_feature_extraction(self, feature_type, count=1):
        """Log feature extraction progress"""
        self.debug(f"🔧 Extracted {feature_type} features (count: {count})")
        self.analysis_stats['features_extracted'] += count
    
    def log_anomaly_detection(self, anomaly_type, severity=None, test_id=None):
        """Log anomaly detection with details"""
        severity_emoji = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': '💡'
        }
        
        emoji = severity_emoji.get(severity, '⚠️')
        severity_text = f" (severity: {severity})" if severity else ""
        test_text = f" in test {test_id}" if test_id else ""
        
        self.info(f"{emoji} {anomaly_type} anomaly detected{severity_text}{test_text}")
        self.analysis_stats['anomalies_detected'] += 1
    
    def log_progress(self, current, total, operation="Processing"):
        """Log progress for long-running operations"""
        percentage = (current / total) * 100 if total > 0 else 0
        self.info(f"📊 {operation}: {current}/{total} ({percentage:.1f}%)")
    
    def log_memory_usage(self, operation=""):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.debug(f"💾 Memory usage{' (' + operation + ')' if operation else ''}: {memory_mb:.1f} MB")
        except ImportError:
            self.debug("💾 Memory monitoring requires psutil package")
    
    def log_performance_summary(self):
        """Log a summary of performance metrics"""
        total_time = time.time() - self.start_time
        
        self.info("="*60)
        self.info("📈 PERFORMANCE SUMMARY")
        self.info("="*60)
        self.info(f"Total execution time: {total_time:.2f} seconds")
        
        # Phase timing breakdown
        if self.phase_times:
            self.info("\nPhase timing breakdown:")
            for phase, timing in self.phase_times.items():
                if 'duration' in timing:
                    percentage = (timing['duration'] / total_time) * 100
                    self.info(f"  • {phase}: {timing['duration']:.2f}s ({percentage:.1f}%)")
        
        # Analysis statistics
        self.info(f"\nAnalysis statistics:")
        for stat_name, value in self.analysis_stats.items():
            display_name = stat_name.replace('_', ' ').title()
            self.info(f"  • {display_name}: {value}")
        
        # Performance metrics
        if self.analysis_stats['tests_analyzed'] > 0:
            tests_per_second = self.analysis_stats['tests_analyzed'] / total_time
            self.info(f"  • Analysis rate: {tests_per_second:.2f} tests/second")
        
        self.info("="*60)
    
    def log_analysis_summary(self, summary_data):
        """Log structured analysis summary"""
        self.info("="*60)
        self.info("📋 ANALYSIS SUMMARY")
        self.info("="*60)
        
        if isinstance(summary_data, dict):
            for key, value in summary_data.items():
                if isinstance(value, dict):
                    self.info(f"{key.replace('_', ' ').title()}:")
                    for sub_key, sub_value in value.items():
                        self.info(f"  • {sub_key.replace('_', ' ').title()}: {sub_value}")
                else:
                    self.info(f"{key.replace('_', ' ').title()}: {value}")
        
        self.info("="*60)
    
    def save_analysis_metadata(self, metadata):
        """Save analysis metadata to JSON file"""
        if not self.output_dir:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_file = self.output_dir / f"analysis_metadata_{timestamp}.json"
            
            # Combine performance data with metadata
            combined_metadata = {
                'analysis_metadata': metadata,
                'performance_metrics': {
                    'total_execution_time': time.time() - self.start_time,
                    'phase_timings': self.phase_times,
                    'analysis_statistics': self.analysis_stats,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(combined_metadata, f, indent=2, default=str)
            
            self.info(f"📊 Analysis metadata saved to: {metadata_file}")
            
        except Exception as e:
            self.error(f"Failed to save analysis metadata: {e}")
    
    def _log_performance(self, message):
        """Internal method to log performance data"""
        if self.enable_performance and hasattr(self, 'perf_handler'):
            # Create a separate logger for performance data
            perf_logger = logging.getLogger(f"{self.name}_perf")
            perf_logger.setLevel(logging.INFO)
            if not perf_logger.handlers:
                perf_logger.addHandler(self.perf_handler)
            perf_logger.info(message)
    
    def create_section_separator(self, title, width=60):
        """Create a formatted section separator for logs"""
        separator = "="*width
        self.info(separator)
        self.info(f"{title.center(width)}")
        self.info(separator)
    
    def create_subsection_separator(self, title, width=40):
        """Create a formatted subsection separator for logs"""
        separator = "-"*width
        self.info(separator)
        self.info(f"{title}")
        self.info(separator)


class EnhancedConsoleHandler(logging.StreamHandler):
    """Enhanced console handler with color support and formatting"""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[0m',       # Default
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m'   # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)
        self.use_colors = self._supports_color()
    
    def _supports_color(self):
        """Check if the terminal supports color output"""
        return (
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
            os.environ.get('TERM') != 'dumb' and
            sys.platform != 'win32'  # Disable on Windows for simplicity
        )
    
    def format(self, record):
        """Format log record with colors if supported"""
        formatted = super().format(record)
        
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            formatted = f"{color}{formatted}{self.RESET}"
        
        return formatted


def setup_logger(name, output_dir=None, level=logging.INFO, enable_performance=True):
    """
    Factory function to create an enhanced ptracker analysis logger
    
    Parameters:
    name (str): Logger name
    output_dir (str): Directory for log files
    level (int): Logging level
    enable_performance (bool): Enable performance tracking
    
    Returns:
    PtrackerAnalysisLogger: Configured enhanced logger
    """
    return PtrackerAnalysisLogger(
        name=name,
        output_dir=output_dir,
        level=level,
        enable_performance=enable_performance
    )


# Backward compatibility functions
def setup_basic_logger(name, output_dir=None, level=logging.INFO):
    """
    Basic logger setup for backward compatibility
    Returns the underlying logging.Logger object
    """
    enhanced_logger = setup_logger(name, output_dir, level, enable_performance=False)
    return enhanced_logger.logger


# Example usage and testing
if __name__ == "__main__":
    # Demo the enhanced logger
    logger = setup_logger("demo_analysis", "./demo_logs", logging.INFO)
    
    # Demo various logging features
    logger.info("Starting demo analysis")
    
    with logger.phase("Demo Phase 1"):
        logger.log_file_processing("test_file_1.log", "processing")
        time.sleep(0.1)  # Simulate work
        logger.log_file_processing("test_file_1.log", "completed")
        
        logger.log_test_analysis("test_001", "analyzing")
        logger.log_feature_extraction("register_values", 25)
        logger.log_anomaly_detection("register_values", "high", "test_001")
        logger.log_test_analysis("test_001", "completed")
    
    with logger.phase("Demo Phase 2"):
        logger.log_progress(50, 100, "Processing tests")
        logger.log_memory_usage("after feature extraction")
        time.sleep(0.1)  # Simulate work
    
    # Demo summary logging
    summary_data = {
        "total_tests": 10,
        "anomalies_detected": 3,
        "feature_counts": {
            "register_values": 8,
            "temporal_patterns": 6
        }
    }
    
    logger.log_analysis_summary(summary_data)
    logger.log_performance_summary()
    
    # Save metadata
    metadata = {"analysis_type": "demo", "version": "1.0"}
    logger.save_analysis_metadata(metadata)