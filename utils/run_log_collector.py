import logging
import argparse
from collector import LogCollector

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Collect and analyze test logs.')
    parser.add_argument('regression_directory', type=str, 
                        help='Path to the directory containing test directories')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create an instance of LogCollector using the provided directory
    collector = LogCollector(args.regression_directory, logger)
    
    # Collect logs
    tests, passing_tests, failing_tests = collector.collect_logs()
    
    # Print summary
    print(f"Total tests: {len(tests)}")
    print(f"Passing tests: {len(passing_tests)}")
    print(f"Failing tests: {len(failing_tests)}")

if __name__ == "__main__":
    main()