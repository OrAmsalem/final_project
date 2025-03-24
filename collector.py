"""
Log collection functionality
"""
import os
import logging


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
        Collect logs from the regression directory
        Each test has its own directory with log files

        Returns:
        dict, dict, dict: All tests, passing tests, and failing tests
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

            # Determine test status
            status = self._determine_test_status(test_path)

            # Store test information
            self.tests[test_id] = {
                'id': test_id,
                'path': test_path,
                'log_path': ptracker_log_path,
                'status': status
            }

            # Add to passing or failing collections based on status
            if status == 'PASS':
                self.passing_tests[test_id] = self.tests[test_id]
            elif status == 'FAIL':
                self.failing_tests[test_id] = self.tests[test_id]

        self.logger.info(
            f"Collected {len(self.passing_tests)} passing tests and {len(self.failing_tests)} failing tests")
        return self.tests, self.passing_tests, self.failing_tests

    def _determine_test_status(self, test_dir):
        """
        Determine if a test passed or failed based on the presence of uvm_error in jestr.log

        Parameters:
        test_dir (str): Path to the test directory

        Returns:
        str: 'PASS', 'FAIL', or 'UNKNOWN'
        """
        try:
            # Look for jestr.log in the test directory
            jestr_log_path = os.path.join(test_dir, "jestr.log")

            if not os.path.exists(jestr_log_path):
                self.logger.warning(f"jestr.log not found in test directory: {test_dir}")
                return 'UNKNOWN'

            # Check for uvm_error in jestr.log
            with open(jestr_log_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if "uvm_error" in line.lower():
                        return 'FAIL'

            # If no uvm_error found, the test passed
            return 'PASS'

        except Exception as e:
            self.logger.error(f"Error determining test status for {test_dir}: {e}")
            return 'UNKNOWN'