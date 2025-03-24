"""
Report generation functionality for AI Debug Assist.
Creates detailed debug reports for failing tests, providing hints for debugging.
"""
import os
import json
import logging
import datetime
import csv
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, select_autoescape


class ReportGenerator:
    def __init__(self, output_dir, template_dir=None, logger=None):
        """
        Initialize the Report Generator

        Parameters:
        output_dir (str): Directory to store generated reports
        template_dir (str): Directory containing report templates
        logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set template directory
        if template_dir:
            self.template_dir = template_dir
        else:
            # Use default templates directory
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.template_dir = os.path.join(module_dir, 'templates')

            # Create templates directory if it doesn't exist
            os.makedirs(self.template_dir, exist_ok=True)

            # Create default template if it doesn't exist
            self._create_default_templates()

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Create reports directory structure
        self.reports_dir = os.path.join(output_dir, 'reports')
        self.summary_dir = os.path.join(output_dir, 'summary')
        self.charts_dir = os.path.join(output_dir, 'charts')

        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)

    def generate_reports(self, tests, anomalies):
        """
        Generate debug-assist reports for failing tests

        Parameters:
        tests (dict): Dictionary of tests with extracted features
        anomalies (dict): Dictionary of detected anomalies

        Returns:
        dict: Paths to generated reports
        """
        self.logger.info("Generating debug-assist reports")

        # Generate timestamp for reports
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate individual test reports
        report_paths = {}
        for test_id, test_data in tests.items():
            # Skip if no anomalies for this test
            if test_id not in anomalies:
                continue

            # Get test anomaly data
            anomaly_data = anomalies[test_id]

            # Skip if no anomalies detected
            if not anomaly_data.get('anomalies', []):
                continue

            # Generate report for this test
            report_path = self._generate_test_report(test_id, test_data, anomaly_data, timestamp)
            if report_path:
                report_paths[test_id] = report_path

        # Generate summary report
        summary_path = self._generate_summary_report(tests, anomalies, timestamp)
        if summary_path:
            report_paths['summary'] = summary_path

        # Generate charts for analysis
        chart_paths = self._generate_charts(tests, anomalies, timestamp)
        if chart_paths:
            report_paths['charts'] = chart_paths

        # Generate rebucketing report
        rebucket_path = self._generate_rebucketing_report(tests, anomalies, timestamp)
        if rebucket_path:
            report_paths['rebucketing'] = rebucket_path

        self.logger.info(f"Generated {len(report_paths)} reports")
        return report_paths

    def _generate_test_report(self, test_id, test_data, anomaly_data, timestamp):
        """
        Generate a detailed report for a specific test

        Parameters:
        test_id (str): Test identifier
        test_data (dict): Test data with features
        anomaly_data (dict): Detected anomalies for this test
        timestamp (str): Report timestamp

        Returns:
        str: Path to the generated report file
        """
        self.logger.debug(f"Generating report for test {test_id}")

        try:
            # Load the test report template
            template = self.env.get_template('test_report.html')

            # Get anomalies
            anomalies = anomaly_data.get('anomalies', [])
            summary = anomaly_data.get('summary', {})

            # Skip if no anomalies
            if not anomalies:
                return None

            # Group anomalies by category
            anomalies_by_category = defaultdict(list)
            for anomaly in anomalies:
                category = anomaly.get('category', 'unknown')
                anomalies_by_category[category].append(anomaly)

            # Generate report context
            context = {
                'test_id': test_id,
                'test_status': test_data.get('status', 'unknown'),
                'timestamp': timestamp,
                'generation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'anomalies': anomalies,
                'anomalies_by_category': dict(anomalies_by_category),
                'summary': summary,
                'test_data': test_data
            }

            # Add hints based on anomalies
            hints = self._generate_hints(anomalies)
            context['hints'] = hints

            # Generate HTML report
            report_html = template.render(context)

            # Save report to file
            report_filename = f"{test_id}_{timestamp}.html"
            report_path = os.path.join(self.reports_dir, report_filename)

            with open(report_path, 'w') as f:
                f.write(report_html)

            # Save as JSON format for machine readability
            json_filename = f"{test_id}_{timestamp}.json"
            json_path = os.path.join(self.reports_dir, json_filename)

            with open(json_path, 'w') as f:
                # Convert non-serializable objects
                serializable_context = self._make_serializable(context)
                json.dump(serializable_context, f, indent=2)

            self.logger.debug(f"Report generated at {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"Error generating report for test {test_id}: {e}")
            return None

    def _generate_summary_report(self, tests, anomalies, timestamp):
        """
        Generate a summary report for all tests

        Parameters:
        tests (dict): Dictionary of tests with extracted features
        anomalies (dict): Dictionary of detected anomalies
        timestamp (str): Report timestamp

        Returns:
        str: Path to the generated summary report
        """
        self.logger.debug("Generating summary report")

        try:
            # Load the summary report template
            template = self.env.get_template('summary_report.html')

            # Extract test stats
            test_stats = {
                'total': len(tests),
                'passing': len([t for t in tests.values() if t.get('status', '').lower() == 'pass']),
                'failing': len([t for t in tests.values() if t.get('status', '').lower() == 'fail'])
            }

            # Extract anomaly stats
            anomaly_stats = {
                'tests_with_anomalies': len([a for a in anomalies.values() if a.get('anomalies', [])]),
                'total_anomalies': sum(len(a.get('anomalies', [])) for a in anomalies.values()),
                'category_counts': self._count_all_anomaly_categories(anomalies),
                'severity_counts': self._count_all_anomaly_severities(anomalies)
            }

            # Identify top anomalies by severity
            top_anomalies = self._get_top_anomalies(anomalies, limit=10)

            # Generate report context
            context = {
                'timestamp': timestamp,
                'generation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'test_stats': test_stats,
                'anomaly_stats': anomaly_stats,
                'top_anomalies': top_anomalies
            }

            # Generate HTML report
            report_html = template.render(context)

            # Save report to file
            report_filename = f"summary_{timestamp}.html"
            report_path = os.path.join(self.summary_dir, report_filename)

            with open(report_path, 'w') as f:
                f.write(report_html)

            # Save as JSON format for machine readability
            json_filename = f"summary_{timestamp}.json"
            json_path = os.path.join(self.summary_dir, json_filename)

            with open(json_path, 'w') as f:
                # Convert non-serializable objects
                serializable_context = self._make_serializable(context)
                json.dump(serializable_context, f, indent=2)

            # Generate CSV summary
            csv_filename = f"summary_{timestamp}.csv"
            csv_path = os.path.join(self.summary_dir, csv_filename)

            self._generate_csv_summary(tests, anomalies, csv_path)

            self.logger.debug(f"Summary report generated at {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return None

    def _generate_charts(self, tests, anomalies, timestamp):
        """
        Generate charts for analysis

        Parameters:
        tests (dict): Dictionary of tests with extracted features
        anomalies (dict): Dictionary of detected anomalies
        timestamp (str): Report timestamp

        Returns:
        dict: Paths to generated chart files
        """
        self.logger.debug("Generating analysis charts")

        try:
            chart_paths = {}

            # Set plot style
            plt.style.use('seaborn-v0_8-darkgrid')

            # Chart 1: Anomaly categories distribution
            category_counts = self._count_all_anomaly_categories(anomalies)
            if category_counts:
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(category_counts.keys(), category_counts.values())
                ax.set_title('Anomaly Categories Distribution')
                ax.set_xlabel('Category')
                ax.set_ylabel('Count')
                ax.set_xticklabels(category_counts.keys(), rotation=45, ha='right')

                # Add count labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                            f'{height:.0f}', ha='center', va='bottom')

                plt.tight_layout()

                # Save chart
                chart_filename = f"anomaly_categories_{timestamp}.png"
                chart_path = os.path.join(self.charts_dir, chart_filename)
                plt.savefig(chart_path, dpi=100)
                plt.close()

                chart_paths['anomaly_categories'] = chart_path

            # Chart 2: Anomaly severity distribution
            severity_counts = self._count_all_anomaly_severities(anomalies)
            if severity_counts:
                # Convert to bins
                severity_bins = {
                    'Low (0.0-0.3)': 0,
                    'Medium (0.3-0.7)': 0,
                    'High (0.7-1.0)': 0
                }

                for severity, count in severity_counts.items():
                    severity_val = float(severity)
                    if severity_val < 0.3:
                        severity_bins['Low (0.0-0.3)'] += count
                    elif severity_val < 0.7:
                        severity_bins['Medium (0.3-0.7)'] += count
                    else:
                        severity_bins['High (0.7-1.0)'] += count

                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#4CAF50', '#FFC107', '#F44336']  # Green, Yellow, Red
                bars = ax.bar(severity_bins.keys(), severity_bins.values(), color=colors)
                ax.set_title('Anomaly Severity Distribution')
                ax.set_xlabel('Severity')
                ax.set_ylabel('Count')

                # Add count labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                            f'{height:.0f}', ha='center', va='bottom')

                plt.tight_layout()

                # Save chart
                chart_filename = f"anomaly_severity_{timestamp}.png"
                chart_path = os.path.join(self.charts_dir, chart_filename)
                plt.savefig(chart_path, dpi=100)
                plt.close()

                chart_paths['anomaly_severity'] = chart_path

            # Chart 3: Test status distribution
            test_statuses = Counter(t.get('status', 'unknown').lower() for t in tests.values())
            if test_statuses:
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = ['#F44336', '#4CAF50', '#2196F3']  # Red, Green, Blue
                wedges, texts, autotexts = ax.pie(
                    [test_statuses.get('fail', 0), test_statuses.get('pass', 0), test_statuses.get('unknown', 0)],
                    labels=['Fail', 'Pass', 'Unknown'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors
                )
                ax.set_title('Test Status Distribution')
                plt.setp(autotexts, size=10, weight='bold')

                plt.tight_layout()

                # Save chart
                chart_filename = f"test_status_{timestamp}.png"
                chart_path = os.path.join(self.charts_dir, chart_filename)
                plt.savefig(chart_path, dpi=100)
                plt.close()

                chart_paths['test_status'] = chart_path

            self.logger.debug(f"Generated {len(chart_paths)} analysis charts")
            return chart_paths

        except Exception as e:
            self.logger.error(f"Error generating analysis charts: {e}")
            return {}

    def _generate_rebucketing_report(self, tests, anomalies, timestamp):
        """
        Generate a rebucketing report based on anomaly analysis

        Parameters:
        tests (dict): Dictionary of tests with extracted features
        anomalies (dict): Dictionary of detected anomalies
        timestamp (str): Report timestamp

        Returns:
        str: Path to the generated rebucketing report
        """
        self.logger.debug("Generating rebucketing report")

        try:
            # Load the rebucketing report template
            template = self.env.get_template('rebucketing_report.html')

            # Group tests by anomaly patterns
            test_groups = self._group_tests_by_anomaly_patterns(tests, anomalies)

            # Generate report context
            context = {
                'timestamp': timestamp,
                'generation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'test_groups': test_groups,
                'group_count': len(test_groups)
            }

            # Generate HTML report
            report_html = template.render(context)

            # Save report to file
            report_filename = f"rebucketing_{timestamp}.html"
            report_path = os.path.join(self.summary_dir, report_filename)

            with open(report_path, 'w') as f:
                f.write(report_html)

            # Save as JSON format for machine readability
            json_filename = f"rebucketing_{timestamp}.json"
            json_path = os.path.join(self.summary_dir, json_filename)

            with open(json_path, 'w') as f:
                # Convert non-serializable objects
                serializable_context = self._make_serializable(context)
                json.dump(serializable_context, f, indent=2)

            self.logger.debug(f"Rebucketing report generated at {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"Error generating rebucketing report: {e}")
            return None

    def _generate_csv_summary(self, tests, anomalies, output_path):
        """
        Generate a CSV summary of all tests and their anomalies

        Parameters:
        tests (dict): Dictionary of tests with extracted features
        anomalies (dict): Dictionary of detected anomalies
        output_path (str): Path to save the CSV file

        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Create CSV rows
            rows = [['Test ID', 'Status', 'Anomaly Count', 'Top Category', 'Max Severity', 'Top Hint']]

            for test_id, test_data in tests.items():
                status = test_data.get('status', 'unknown')

                if test_id in anomalies:
                    anomaly_data = anomalies[test_id]
                    anomaly_count = len(anomaly_data.get('anomalies', []))

                    # Get top category
                    categories = anomaly_data.get('summary', {}).get('categories', {})
                    top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else 'N/A'

                    # Get max severity
                    max_severity = max(
                        [a.get('severity', 0) for a in anomaly_data.get('anomalies', [])]) if anomaly_data.get(
                        'anomalies') else 0

                    # Get top hint
                    hints = self._generate_hints(anomaly_data.get('anomalies', []))
                    top_hint = hints[0] if hints else 'N/A'
                else:
                    anomaly_count = 0
                    top_category = 'N/A'
                    max_severity = 0
                    top_hint = 'N/A'

                rows.append([test_id, status, anomaly_count, top_category, f"{max_severity:.2f}", top_hint])

            # Write CSV file
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

            return True

        except Exception as e:
            self.logger.error(f"Error generating CSV summary: {e}")
            return False

    def _generate_hints(self, anomalies):
        """
        Generate debug hints based on detected anomalies

        Parameters:
        anomalies (list): List of detected anomalies

        Returns:
        list: List of debug hints
        """
        hints = []

        # Sort anomalies by severity
        sorted_anomalies = sorted(anomalies, key=lambda x: x.get('severity', 0), reverse=True)

        for anomaly in sorted_anomalies:
            category = anomaly.get('category', 'unknown')

            if category == 'register_value':
                register = anomaly.get('register', 'unknown')
                expected = anomaly.get('expected', {})
                actual = anomaly.get('actual', {})

                # Craft hint based on anomaly type
                for sub_anomaly in anomaly.get('anomalies', []):
                    anomaly_type = sub_anomaly.get('type', '')

                    if anomaly_type == 'mean_value':
                        hints.append(
                            f"Check register '{register}' - average value {actual.get('mean', 'N/A')} differs significantly from expected {expected.get('mean', 'N/A')}")

                    elif anomaly_type == 'min_value':
                        hints.append(
                            f"Check register '{register}' - minimum value {actual.get('min', 'N/A')} is below expected {expected.get('range', ['N/A', 'N/A'])[0]}")

                    elif anomaly_type == 'max_value':
                        hints.append(
                            f"Check register '{register}' - maximum value {actual.get('max', 'N/A')} exceeds expected {expected.get('range', ['N/A', 'N/A'])[1]}")

                    elif anomaly_type == 'distribution':
                        hints.append(f"Check register '{register}' - value distribution deviates from normal pattern")

            elif category == 'fsm_transition':
                fsm_name = anomaly.get('fsm_name', 'unknown')

                for sub_anomaly in anomaly.get('anomalies', []):
                    anomaly_type = sub_anomaly.get('type', '')

                    if anomaly_type == 'invalid_transition':
                        from_state = sub_anomaly.get('from_state', 'unknown')
                        to_state = sub_anomaly.get('to_state', 'unknown')
                        hints.append(f"Check FSM '{fsm_name}' - invalid transition from '{from_state}' to '{to_state}'")

                    elif anomaly_type == 'missing_state':
                        state = sub_anomaly.get('state', 'unknown')
                        hints.append(f"Check FSM '{fsm_name}' - expected state '{state}' not observed")

                    elif anomaly_type == 'cycle':
                        hints.append(f"Check FSM '{fsm_name}' - unexpected cyclic behavior detected")

            elif category == 'register_sequence':
                for sub_anomaly in anomaly.get('anomalies', []):
                    sequence = sub_anomaly.get('sequence', 'unknown')
                    hints.append(f"Check register sequence - expected sequence '{sequence}' not observed")

            elif category == 'temporal_pattern':
                for sub_anomaly in anomaly.get('anomalies', []):
                    metric = sub_anomaly.get('metric', 'unknown')
                    actual = sub_anomaly.get('actual', 'N/A')
                    expected = sub_anomaly.get('expected', 'N/A')
                    hints.append(f"Check temporal pattern - {metric} ({actual}) differs from expected ({expected})")

            elif category == 'register_access_pattern':
                register = anomaly.get('register', 'unknown')
                for sub_anomaly in anomaly.get('anomalies', []):
                    anomaly_type = sub_anomaly.get('type', '')

                    if anomaly_type == 'read_write_ratio':
                        actual = sub_anomaly.get('actual', 'N/A')
                        expected = sub_anomaly.get('expected', 'N/A')
                        hints.append(
                            f"Check register '{register}' - read/write ratio ({actual}) differs from expected ({expected})")

        return hints

    def _group_tests_by_anomaly_patterns(self, tests, anomalies):
        """
        Group tests by anomaly patterns for rebucketing

        Parameters:
        tests (dict): Dictionary of tests with extracted features
        anomalies (dict): Dictionary of detected anomalies

        Returns:
        list: List of test groups
        """
        # Group tests based on anomaly patterns
        pattern_groups = defaultdict(list)

        for test_id, anomaly_data in anomalies.items():
            if not anomaly_data.get('anomalies'):
                continue

            # Create a pattern signature based on top anomalies
            top_anomalies = sorted(anomaly_data['anomalies'], key=lambda x: x.get('severity', 0), reverse=True)[:3]

            # Create a pattern key
            pattern_key = []

            for anomaly in top_anomalies:
                category = anomaly.get('category', 'unknown')

                if category == 'register_value':
                    register = anomaly.get('register', 'unknown')
                    pattern_key.append(f"register_value:{register}")

                elif category == 'fsm_transition':
                    fsm_name = anomaly.get('fsm_name', 'unknown')
                    pattern_key.append(f"fsm_transition:{fsm_name}")

                elif category == 'register_sequence':
                    pattern_key.append("register_sequence")

                elif category == 'temporal_pattern':
                    pattern_key.append("temporal_pattern")

                elif category == 'register_access_pattern':
                    register = anomaly.get('register', 'unknown')
                    pattern_key.append(f"register_access_pattern:{register}")

            # Use top 2 patterns as the key
            pattern_signature = "|".join(pattern_key[:2]) if pattern_key else "no_pattern"

            # Add test to group
            pattern_groups[pattern_signature].append({
                'test_id': test_id,
                'status': tests[test_id].get('status', 'unknown'),
                'anomaly_count': len(anomaly_data['anomalies']),
                'top_anomalies': top_anomalies
            })

        # Convert to list of groups
        groups = []
        for signature, tests in pattern_groups.items():
            if len(tests) >= 2:  # Only consider groups with at least 2 tests
                # Identify common anomalies across all tests in group
                common_anomalies = self._identify_common_anomalies(tests)

                groups.append({
                    'signature': signature,
                    'tests': tests,
                    'count': len(tests),
                    'common_anomalies': common_anomalies,
                    'top_hints': self._generate_hints(common_anomalies)
                })

        # Sort groups by size (descending)
        groups.sort(key=lambda x: x['count'], reverse=True)

        return groups

    def _identify_common_anomalies(self, tests):
        """
        Identify common anomalies across all tests in a group

        Parameters:
        tests (list): List of tests with top anomalies

        Returns:
        list: List of common anomalies
        """
        # Extract all anomalies
        all_anomalies = []
        for test in tests:
            all_anomalies.extend(test.get('top_anomalies', []))

        # Group anomalies by category and key attributes
        grouped_anomalies = defaultdict(list)

        for anomaly in all_anomalies:
            category = anomaly.get('category', 'unknown')

            if category == 'register_value':
                register = anomaly.get('register', 'unknown')
                key = f"{category}:{register}"

            elif category == 'fsm_transition':
                fsm_name = anomaly.get('fsm_name', 'unknown')
                key = f"{category}:{fsm_name}"

            elif category == 'register_access_pattern':
                register = anomaly.get('register', 'unknown')
                key = f"{category}:{register}"

            else:
                key = category

            grouped_anomalies[key].append(anomaly)

        # Identify anomalies that appear multiple times
        common_anomalies = []

        for key, anomalies in grouped_anomalies.items():
            if len(anomalies) >= len(tests) // 2:  # Appear in at least half of the tests
                # Use the highest severity anomaly as representative
                representative = max(anomalies, key=lambda x: x.get('severity', 0))
                common_anomalies.append(representative)

        # Sort by severity
        common_anomalies.sort(key=lambda x: x.get('severity', 0), reverse=True)

        return common_anomalies

    def _count_anomaly_categories(self, anomalies):
        """
        Count anomalies by category for a single test

        Parameters:
        anomalies (list): List of anomalies

        Returns:
        dict: Category counts
        """
        category_counts = defaultdict(int)

        for anomaly in anomalies:
            category = anomaly.get('category', 'unknown')
            category_counts[category] += 1

        return dict(category_counts)

    def _count_anomaly_severities(self, anomalies):
        """
        Count anomalies by severity for a single test

        Parameters:
        anomalies (list): List of anomalies

        Returns:
        dict: Severity counts
        """
        severity_counts = defaultdict(int)

        for anomaly in anomalies:
            severity = anomaly.get('severity', 0)
            # Round to 1 decimal place
            severity_str = f"{severity:.1f}"
            severity_counts[severity_str] += 1

        return dict(severity_counts)

    def _count_all_anomaly_categories(self, anomalies_dict):
        """
        Count all anomalies by category across all tests

        Parameters:
        anomalies_dict (dict): Dictionary of anomalies by test ID

        Returns:
        dict: Category counts
        """
        category_counts = defaultdict(int)

        for test_id, anomaly_data in anomalies_dict.items():
            for anomaly in anomaly_data.get('anomalies', []):
                category = anomaly.get('category', 'unknown')
                category_counts[category] += 1

        return dict(category_counts)

    def _count_all_anomaly_severities(self, anomalies_dict):
        """
        Count all anomalies by severity across all tests

        Parameters:
        anomalies_dict (dict): Dictionary of anomalies by test ID

        Returns:
        dict: Severity counts
        """
        severity_counts = defaultdict(int)

        for test_id, anom