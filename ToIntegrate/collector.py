import os
import logging
import gzip
import re

class LogCollector:
    def __init__(self, regression_directory, logger=None):
        """
        Initialize the log collector
        Parameters:
        regression_directory (str): Path to the directory containing test directories
        logger (logging.Logger): Logger instance
        """
        self.regression_directory = regression_directory
        self.logger = logger or logging.getLogger(__name__)
        self.tests = {}
        self.passing_tests = {}
        self.failing_tests = {}
    
    def collect_logs(self):
        """
        Collect logs from the regression directory.
        Each test has its own directory with log files.
        Returns:
        dict, dict, dict: All tests, passing tests, and failing tests.
        """
        self.logger.info(f"Collecting logs from regression directory: {self.regression_directory}")
        # Reset collections
        self.tests = {}
        self.passing_tests = {}
        self.failing_tests = {}
        
        # Get all test directories
        try:
            test_dirs = [d for d in os.listdir(self.regression_directory)
                         if os.path.isdir(os.path.join(self.regression_directory, d))]
        except FileNotFoundError:
            self.logger.error(f"Regression directory not found: {self.regression_directory}")
            return self.tests, self.passing_tests, self.failing_tests
            
        self.logger.info(f"Found {len(test_dirs)} test directories")
        
        # Process each test directory
        for test_dir in test_dirs:
            test_id = test_dir
            test_path = os.path.join(self.regression_directory, test_dir)
            ptracker_log_path = os.path.join(test_path, "ptracker.log.gz")
            
            # Check if ptracker log file exists
            if not os.path.exists(ptracker_log_path):
                self.logger.warning(f"ptracker.log.gz not found for test {test_id}: {ptracker_log_path}")
                continue
                
            # Determine test status based solely on exit code
            status = self._determine_test_status(test_path)
            
            # Store test information
            self.tests[test_id] = {
                'id': test_id,
                'path': test_path,
                'log_path': ptracker_log_path,
                'status': status
            }
            
            # Add test to the appropriate collection
            if status == 'PASS':
                self.passing_tests[test_id] = self.tests[test_id]
            else:
                # Now any status other than PASS is considered as a failure.
                self.failing_tests[test_id] = self.tests[test_id]
                
        self.logger.info(
            f"Collected {len(self.passing_tests)} passing tests and {len(self.failing_tests)} failing tests")
        return self.tests, self.passing_tests, self.failing_tests
    
    def _determine_test_status(self, test_dir):
        """
        Determine if a test passed or failed based on the exit status in logbook.log.gz.
        Rule: If exit status is 0, test PASSES. Any other exit status means test FAILS.
        
        Parameters:
        test_dir (str): Path to the test directory.
        
        Returns:
        str: 'PASS' if exit status is 0, 'FAIL' for any other exit status or if the exit status cannot be determined.
        """
        logbook_path = os.path.join(test_dir, "logbook.log.gz")
        
        if not os.path.exists(logbook_path):
            self.logger.warning(f"logbook.log.gz not found in test directory: {test_dir}")
            return 'FAIL'  # Consider missing logbook as a failure.
        
        exit_status_pattern = re.compile(r'exit status:\s*(\d+)', re.IGNORECASE)
        
        try:
            with gzip.open(logbook_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    match = exit_status_pattern.search(line)
                    if match:
                        try:
                            exit_status = int(match.group(1))
                            # Explicitly check if exit status is 0 (PASS) or anything else (FAIL)
                            if exit_status == 0:
                                return 'PASS'
                            else:
                                self.logger.info(f"Test failed with exit status: {exit_status}")
                                return 'FAIL'
                        except ValueError:
                            self.logger.warning(f"Found exit status text but couldn't parse integer: {line}")
                            return 'FAIL'
            
            # If no exit status is found, default to marking the test as failed.
            self.logger.warning(f"No exit status found in logbook.log.gz for test: {test_dir}")
            return 'FAIL'
            
        except gzip.BadGzipFile:
            self.logger.error(f"Invalid gzip file: {logbook_path}")
            return 'FAIL'
        except Exception as e:
            self.logger.error(f"Error determining test status for {test_dir}: {str(e)}")
            return 'FAIL'