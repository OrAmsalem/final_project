import os
import shutil
import sys
import gzip

def copy_ptracker_logs(source_dir, destination_dir):
    # Create directories for passing and failing tests
    passing_dir = os.path.join(destination_dir, 'passing_tests')
    failing_dir = os.path.join(destination_dir, 'failing_tests')
    os.makedirs(passing_dir, exist_ok=True)
    os.makedirs(failing_dir, exist_ok=True)

    # Iterate over each item in the source directory
    for test_dir in os.listdir(source_dir):
        test_dir_path = os.path.join(source_dir, test_dir)

        # Check if the item is a directory
        if os.path.isdir(test_dir_path):
            ptracker_log_path = os.path.join(test_dir_path, 'ptracker.log.gz')
            logbook_path = os.path.join(test_dir_path, 'logbook.log.gz')

            # Check if the ptracker.log.gz and logbook.log.gz files exist in the test directory
            if os.path.exists(ptracker_log_path) and os.path.exists(logbook_path):
                # Determine if the test is passing or failing by reading the exit code from logbook.log.gz
                with gzip.open(logbook_path, 'rt') as logbook_file:
                    logbook_content = logbook_file.readlines()
                    # Assuming the exit code is in the last few lines
                    exit_code = None
                    for line in reversed(logbook_content):
                        if 'exit status' in line.lower():
                            try:
                                exit_code = int(line.split()[-1])
                                break
                            except ValueError:
                                print(f"Invalid exit status format in {logbook_path}. Skipping.")
                                continue

                if exit_code is not None:
                    if exit_code == 0:
                        destination_path = os.path.join(passing_dir, f"{test_dir}_ptracker.log.gz")
                    else:
                        destination_path = os.path.join(failing_dir, f"{test_dir}_ptracker.log.gz")

                    # Copy the file to the appropriate destination directory
                    shutil.copy(ptracker_log_path, destination_path)
                    print(f"Copied {ptracker_log_path} to {destination_path}")
                else:
                    print(f"No valid exit code found in {logbook_path}. Skipping.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_directory> <destination_directory>")
        sys.exit(1)

    source_directory = sys.argv[1]
    destination_directory = sys.argv[2]

    copy_ptracker_logs(source_directory, destination_directory)