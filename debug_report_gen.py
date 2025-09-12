#!/usr/bin/env python3
"""
Enhanced debug report generator with divergence point detection.
Adds a line for each hint showing the first state where expected and actual data differ.
"""

import re
import json
import os
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path


class DebugHintsReporter:
    def __init__(self, logger=None):
        self.logger = logger
        self.register_failures = defaultdict(list)
        self.sequence_analyzer = None
        self.critical_registers = [
            'ring_cstate.ring_cstate_control.target_ring_cstate',
            'ring_cstate.ring_cstate_control.curr_ring_cstate', 
            'ring_cstate.ring_cstate_control.ring_cst_fsm',
            'hvm_compressed_data.state',
            'Registry::get_registry'
        ]
        
    def set_sequence_analyzer(self, analyzer):
        """Set the sequence analyzer for proper mismatch detection"""
        self.sequence_analyzer = analyzer
        if self.logger:
            self.logger.info("Sequence analyzer set for enhanced debug reporting")
    
    def find_first_divergence_point(self, expected_sequence, actual_sequence):
        """
        Find the first position where expected and actual sequences differ
        
        Parameters:
        expected_sequence (list): Expected values
        actual_sequence (list): Actual values
        
        Returns:
        dict: Divergence information or None if sequences are identical
        """
        if not expected_sequence or not actual_sequence:
            return {
                'position': 0,
                'reason': 'empty_sequence',
                'expected_value': None,
                'actual_value': None,
                'hex_expected': 'N/A',
                'hex_actual': 'N/A'
            }
        
        # Check for length differences first
        min_length = min(len(expected_sequence), len(actual_sequence))
        
        # Find first value difference
        for i in range(min_length):
            if expected_sequence[i] != actual_sequence[i]:
                return {
                    'position': i,
                    'reason': 'value_mismatch',
                    'expected_value': expected_sequence[i],
                    'actual_value': actual_sequence[i],
                    'hex_expected': f"0x{expected_sequence[i]:x}",
                    'hex_actual': f"0x{actual_sequence[i]:x}",
                    'position_description': f"position {i}"
                }
        
        # If we get here, all compared values are identical but lengths differ
        if len(expected_sequence) != len(actual_sequence):
            longer_seq = expected_sequence if len(expected_sequence) > len(actual_sequence) else actual_sequence
            shorter_seq = actual_sequence if len(expected_sequence) > len(actual_sequence) else expected_sequence
            is_expected_longer = len(expected_sequence) > len(actual_sequence)
            
            return {
                'position': min_length,
                'reason': 'length_mismatch',
                'expected_value': expected_sequence[min_length] if min_length < len(expected_sequence) else None,
                'actual_value': actual_sequence[min_length] if min_length < len(actual_sequence) else None,
                'hex_expected': f"0x{expected_sequence[min_length]:x}" if min_length < len(expected_sequence) else "END",
                'hex_actual': f"0x{actual_sequence[min_length]:x}" if min_length < len(actual_sequence) else "END",
                'position_description': f"position {min_length} (length difference: expected={len(expected_sequence)}, actual={len(actual_sequence)})"
            }
        
        # Sequences are identical
        return None
    
    def parse_sequence_values(self, sequence_str):
        """
        Parse sequence string into list of integer values
        
        Parameters:
        sequence_str (str): String containing hex values
        
        Returns:
        list: List of integer values
        """
        if not sequence_str:
            return []
        
        # Remove brackets, parentheses, and split by whitespace or commas
        cleaned = re.sub(r'[[\]()]', '', sequence_str)
        hex_values = re.findall(r'0x[0-9a-fA-F]+|\b[0-9]+\b', cleaned)
        
        values = []
        for hex_val in hex_values:
            try:
                if hex_val.startswith('0x'):
                    values.append(int(hex_val, 16))
                else:
                    values.append(int(hex_val))
            except ValueError:
                continue
        
        return values
    
    def parse_debug_output_file(self, debug_file):
        """Parse debug output from a file"""
        try:
            with open(debug_file, 'r') as f:
                content = f.read()
            self.parse_debug_output(content)
            if self.logger:
                self.logger.info(f"Successfully parsed debug file: {debug_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error parsing debug file {debug_file}: {e}")
    
    def parse_debug_output(self, debug_text):
        """Parse the debug output and extract register mismatch information"""
        
        if self.logger:
            self.logger.info("Parsing debug output for register mismatches")
        
        # Parse register status information
        register_pattern = r'(\S+.*?)\s+(\d+)\s+OBSERVED SEQ:\s*(.+?)\s+EXPECTED SEQ:\s+\((\d+) out of (\d+)\)\s+(.+)'
        
        matches = re.findall(register_pattern, debug_text, re.MULTILINE)
        
        if self.logger:
            self.logger.info(f"Found {len(matches)} register mismatch entries from debug output")
        
        for match in matches:
            register_name = match[0].strip()
            address = match[1] 
            observed_seq = match[2].strip()
            observed_count = int(match[3])
            total_expected = int(match[4])
            expected_seq = match[5].strip()
            
            # Parse sequences into value arrays
            expected_values = self.parse_sequence_values(expected_seq)
            actual_values = self.parse_sequence_values(observed_seq)
            
            # Find divergence point
            divergence = self.find_first_divergence_point(expected_values, actual_values)
            
            failure_info = {
                'register': register_name,
                'address': address,
                'observed_sequence': observed_seq,
                'expected_sequence': expected_seq,
                'observed_values': actual_values,
                'expected_values': expected_values,
                'observed_count': observed_count,
                'total_expected': total_expected,
                'success_rate': observed_count / total_expected if total_expected > 0 else 0,
                'failure_count': total_expected - observed_count,
                'divergence_point': divergence,
                'source': 'debug_output'
            }
            
            self.register_failures[register_name].append(failure_info)
    
    def generate_summary_report(self, passing_tests=None, failing_tests=None):
        """Generate comprehensive report"""
        
        if self.logger:
            self.logger.info("Generating debug summary report with divergence analysis")
        
        # Initialize basic report structure
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_registers_with_failures': len(self.register_failures),
                'debug_output_failures': sum(len(failures) for failures in self.register_failures.values()),
                'sequence_analysis_failures': 0,
                'most_problematic_registers': [],
                'critical_register_issues': 0,
                'divergence_statistics': {
                    'total_divergences_found': 0,
                    'value_mismatches': 0,
                    'length_mismatches': 0,
                    'early_divergences': 0  # Divergences in first 3 positions
                }
            },
            'debug_output_failures': dict(self.register_failures),
            'sequence_analysis': {},
            'register_details': {},
            'divergence_analysis': {}
        }
        
        # Collect divergence statistics
        divergence_stats = report['summary']['divergence_statistics']
        divergence_details = {}
        
        for register, failures in self.register_failures.items():
            register_divergences = []
            for failure in failures:
                if failure.get('divergence_point'):
                    div = failure['divergence_point']
                    divergence_stats['total_divergences_found'] += 1
                    
                    if div['reason'] == 'value_mismatch':
                        divergence_stats['value_mismatches'] += 1
                    elif div['reason'] == 'length_mismatch':
                        divergence_stats['length_mismatches'] += 1
                    
                    if div['position'] < 3:
                        divergence_stats['early_divergences'] += 1
                    
                    register_divergences.append({
                        'position': div['position'],
                        'reason': div['reason'],
                        'expected_hex': div['hex_expected'],
                        'actual_hex': div['hex_actual'],
                        'description': div.get('position_description', f"position {div['position']}")
                    })
            
            if register_divergences:
                divergence_details[register] = register_divergences
        
        report['divergence_analysis'] = divergence_details
        
        # If we have sequence analyzer and test data, do enhanced analysis
        if self.sequence_analyzer and passing_tests and failing_tests:
            try:
                if self.logger:
                    self.logger.info("Running enhanced sequence analysis")
                
                # Build reference model
                self.sequence_analyzer.build_reference_model(passing_tests)
                
                # Analyze failing tests
                analysis_results = self.sequence_analyzer.analyze_failing_tests(failing_tests)
                
                # Generate mismatch report
                sequence_report = self.sequence_analyzer.generate_enhanced_mismatch_report(analysis_results)
                
                # Update report with sequence analysis
                report['sequence_analysis'] = sequence_report
                if 'summary' in sequence_report:
                    report['summary']['sequence_analysis_failures'] = sequence_report['summary'].get('total_root_cause_candidates', 0)
                
                # Process sequence analysis divergences
                if 'ai_analysis' in sequence_report:
                    ai_errors = sequence_report['ai_analysis'].get('most_critical_errors', [])
                    for error in ai_errors:
                        register = error.get('register', 'unknown')
                        
                        # Add divergence analysis for sequence mismatches
                        if error.get('root_cause_type') == 'reference_model_mismatch':
                            expected = error.get('expected_sequence', [])
                            actual = error.get('actual_sequence', [])
                            
                            divergence = self.find_first_divergence_point(expected, actual)
                            if divergence:
                                if register not in divergence_details:
                                    divergence_details[register] = []
                                
                                divergence_details[register].append({
                                    'position': divergence['position'],
                                    'reason': divergence['reason'],
                                    'expected_hex': divergence['hex_expected'],
                                    'actual_hex': divergence['hex_actual'],
                                    'description': divergence.get('position_description', f"position {divergence['position']}"),
                                    'source': 'sequence_analysis'
                                })
                
                # Combine problematic registers
                problematic_registers = {}
                
                # From debug output
                for register, failures in self.register_failures.items():
                    total_failures = sum(f['failure_count'] for f in failures)
                    problematic_registers[register] = {
                        'count': total_failures,
                        'source': 'debug_output'
                    }
                
                # From sequence analysis
                if 'summary' in sequence_report and 'reference_model_errors' in sequence_report['summary']:
                    # This is a simplified way to get sequence analysis register counts
                    # You might need to adjust based on your actual sequence_report structure
                    ai_errors = sequence_report.get('ai_analysis', {}).get('most_critical_errors', [])
                    seq_register_counts = {}
                    for error in ai_errors:
                        register = error.get('register', 'unknown')
                        count = error.get('occurrence_count', 1)
                        seq_register_counts[register] = seq_register_counts.get(register, 0) + count
                    
                    for register, count in seq_register_counts.items():
                        if register in problematic_registers:
                            problematic_registers[register]['count'] += count
                            problematic_registers[register]['source'] = 'both'
                        else:
                            problematic_registers[register] = {
                                'count': count,
                                'source': 'sequence_analysis'
                            }
                
                # Sort by failure count
                most_problematic = sorted(problematic_registers.items(), 
                                        key=lambda x: x[1]['count'], reverse=True)[:15]
                report['summary']['most_problematic_registers'] = most_problematic
                
                # Count critical issues
                if 'ai_analysis' in sequence_report:
                    critical_errors = [e for e in sequence_report['ai_analysis'].get('most_critical_errors', []) 
                                     if e.get('significance_score', 0) >= 0.8]
                    report['summary']['critical_register_issues'] = len(critical_errors)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in enhanced sequence analysis: {e}")
                # Fall back to basic analysis
        
        # If no enhanced analysis, just use debug output data
        if not report['summary']['most_problematic_registers']:
            register_failure_counts = {}
            for register, failures in self.register_failures.items():
                total_failures = sum(f['failure_count'] for f in failures)
                register_failure_counts[register] = total_failures
            
            most_problematic = sorted(register_failure_counts.items(), 
                                    key=lambda x: x[1], reverse=True)[:15]
            report['summary']['most_problematic_registers'] = [
                (reg, {'count': count, 'source': 'debug_output'}) 
                for reg, count in most_problematic
            ]
        
        return report
    
    def format_report_text(self, report):
        """Format the report as readable text with divergence points"""
        
        text_report = []
        text_report.append("="*80)
        text_report.append("ENHANCED DEBUG HINTS REPORT WITH DIVERGENCE ANALYSIS")
        text_report.append("="*80)
        text_report.append(f"Generated: {report['timestamp']}")
        text_report.append("")
        
        # Summary section
        text_report.append("SUMMARY")
        text_report.append("-" * 40)
        summary = report['summary']
        text_report.append(f"Total registers with failures: {summary['total_registers_with_failures']}")
        text_report.append(f"Debug output failures: {summary['debug_output_failures']}")
        text_report.append(f"Sequence analysis failures: {summary['sequence_analysis_failures']}")
        text_report.append(f"Critical register issues: {summary['critical_register_issues']}")
        
        # Divergence statistics
        div_stats = summary.get('divergence_statistics', {})
        if div_stats['total_divergences_found'] > 0:
            text_report.append("")
            text_report.append("DIVERGENCE STATISTICS")
            text_report.append("-" * 20)
            text_report.append(f"Total divergences found: {div_stats['total_divergences_found']}")
            text_report.append(f"Value mismatches: {div_stats['value_mismatches']}")
            text_report.append(f"Length mismatches: {div_stats['length_mismatches']}")
            text_report.append(f"Early divergences (pos 0-2): {div_stats['early_divergences']}")
        
        text_report.append("")
        
        # Most problematic registers
        text_report.append("TOP 15 MOST PROBLEMATIC REGISTERS")
        text_report.append("-" * 40)
        if summary['most_problematic_registers']:
            for i, (register, details) in enumerate(summary['most_problematic_registers'], 1):
                if isinstance(details, dict):
                    count = details['count']
                    source = details.get('source', 'unknown')
                else:
                    count = details
                    source = 'legacy'
                text_report.append(f"{i:2d}. {register}")
                text_report.append(f"    Failures: {count} | Source: {source}")
                
                # Add divergence point information if available
                divergence_info = report.get('divergence_analysis', {}).get(register, [])
                if divergence_info:
                    # Show first divergence for this register
                    div = divergence_info[0]
                    text_report.append(f"    🔍 FIRST DIVERGENCE: {div['description']}")
                    text_report.append(f"       Expected: {div['expected_hex']} | Actual: {div['actual_hex']} | Reason: {div['reason']}")
        else:
            text_report.append("No problematic registers detected.")
        text_report.append("")
        
        # Enhanced divergence analysis section
        if 'divergence_analysis' in report and report['divergence_analysis']:
            text_report.append("DETAILED DIVERGENCE ANALYSIS")
            text_report.append("-" * 40)
            text_report.append("🔍 First divergence point for each register:")
            text_report.append("")
            
            # Sort registers by earliest divergence position
            sorted_divergences = []
            for register, divergences in report['divergence_analysis'].items():
                if divergences:
                    earliest_div = min(divergences, key=lambda x: x['position'])
                    sorted_divergences.append((register, earliest_div, len(divergences)))
            
            sorted_divergences.sort(key=lambda x: x[1]['position'])
            
            for register, div, total_divs in sorted_divergences[:10]:  # Show top 10
                text_report.append(f"📊 {register}")
                text_report.append(f"   Position: {div['position']} | Reason: {div['reason']}")
                text_report.append(f"   Expected: {div['expected_hex']} → Actual: {div['actual_hex']}")
                if total_divs > 1:
                    text_report.append(f"   (Total divergences for this register: {total_divs})")
                text_report.append("")
        
        # Sequence analysis results with enhanced transition pattern handling
        if 'sequence_analysis' in report and report['sequence_analysis']:
            seq_analysis = report['sequence_analysis']
            
            text_report.append("SEQUENCE ANALYSIS RESULTS")
            text_report.append("-" * 40)
            
            if 'summary' in seq_analysis:
                seq_summary = seq_analysis['summary']
                text_report.append(f"Tests analyzed: {seq_summary.get('total_failing_tests', 0)}")
                text_report.append(f"Tests with issues: {seq_summary.get('tests_with_issues', 0)}")
                text_report.append(f"Total root cause candidates: {seq_summary.get('total_root_cause_candidates', 0)}")
                text_report.append(f"Reference model errors: {seq_summary.get('reference_model_errors', 0)}")
                text_report.append(f"Fail-only divergence errors: {seq_summary.get('fail_only_divergence_errors', 0)}")
                text_report.append("")
            
            # Show critical errors with enhanced transition divergence points
            if 'ai_analysis' in seq_analysis and seq_analysis['ai_analysis'].get('most_critical_errors'):
                text_report.append("TRANSITION PATTERNS WITH DIVERGENCE POINTS")
                text_report.append("-" * 50)
                
                critical_errors = seq_analysis['ai_analysis']['most_critical_errors'][:10]  # Show more patterns
                pattern_counter = 1
                
                for error in critical_errors:
                    register = error.get('register', 'unknown')
                    count = error.get('occurrence_count', 0)
                    significance = error.get('significance_score', 0)
                    root_cause_type = error.get('root_cause_type', 'unknown')
                    
                    # Check if this looks like a transition pattern
                    is_transition_pattern = False
                    expected_transitions = ""
                    actual_transitions = ""
                    
                    if root_cause_type == 'reference_model_mismatch':
                        expected = error.get('expected_sequence', [])
                        actual = error.get('actual_sequence', [])
                        
                        # Check if this looks like state transitions (sequential values)
                        if expected and actual and len(expected) > 3 and len(actual) > 3:
                            # Convert to transition strings for display
                            expected_transitions = "→".join([f"0x{val:x}" for val in expected])
                            actual_transitions = "→".join([f"0x{val:x}" for val in actual])
                            is_transition_pattern = True
                    
                    if is_transition_pattern:
                        text_report.append(f"\nTRANSITION PATTERN #{pattern_counter}")
                        text_report.append(f"   Register: {register}")
                        text_report.append(f"   Occurs in: {count} tests")
                        text_report.append(f"   Severity: {significance:.3f}")
                        text_report.append(f"   Expected transitions: {expected_transitions}")
                        text_report.append(f"   Actual transitions:   {actual_transitions}")
                        
                        # Find and display transition divergence point
                        transition_divergence = self.find_transition_divergence_point(expected_transitions, actual_transitions)
                        if transition_divergence:
                            text_report.append(f"   🔍 First divergence point: {transition_divergence['divergence_description']}")
                        
                        # Determine issue category
                        if len(expected) != len(actual):
                            if len(actual) < len(expected):
                                issue_category = "sequence_truncated"
                                issue_desc = "Transition sequence ends prematurely"
                            else:
                                issue_category = "sequence_extended"
                                issue_desc = "Transition sequence continues unexpectedly"
                        else:
                            issue_category = "different_transition_path"
                            issue_desc = "Completely different state transition sequence"
                        
                        text_report.append(f"   Issue: {issue_desc}")
                        text_report.append(f"   Category: {issue_category}")
                        
                        pattern_counter += 1
                    
                    else:
                        # Handle non-transition patterns (original format)
                        text_report.append(f"\nSEQUENCE PATTERN #{pattern_counter}")
                        text_report.append(f"   Register: {register} ({root_cause_type})")
                        text_report.append(f"   Occurs in: {count} tests | Significance: {significance:.3f}")
                        
                        if root_cause_type == 'reference_model_mismatch':
                            expected = error.get('expected_sequence', [])
                            actual = error.get('actual_sequence', [])
                            
                            # Find and display divergence point
                            divergence = self.find_first_divergence_point(expected, actual)
                            if divergence:
                                text_report.append(f"   🔍 DIVERGENCE at {divergence['position_description']}")
                                text_report.append(f"   Expected: {divergence['hex_expected']} | Actual: {divergence['hex_actual']}")
                            
                            # Show sequence samples
                            expected_hex = ' '.join([f"0x{val:x}" for val in expected[:8]])
                            actual_hex = ' '.join([f"0x{val:x}" for val in actual[:8]])
                            text_report.append(f"   EXPECTED SEQ: {expected_hex}{'...' if len(expected) > 8 else ''}")
                            text_report.append(f"   OBSERVED SEQ: {actual_hex}{'...' if len(actual) > 8 else ''}")
                        
                        elif root_cause_type == 'fail_only_divergence':
                            div_point = error.get('divergence_point', 'unknown')
                            text_report.append(f"   🔍 DIVERGENCE at position: {div_point}")
                            if 'common_prefix' in error:
                                prefix_hex = ' '.join([f"0x{val:x}" for val in error['common_prefix'][:5]])
                                text_report.append(f"   Stable prefix: {prefix_hex}{'...' if len(error['common_prefix']) > 5 else ''}")
                        
                        pattern_counter += 1
        
        # Debug output analysis with divergence points
        if 'debug_output_failures' in report and report['debug_output_failures']:
            text_report.append("\nDEBUG OUTPUT ANALYSIS WITH DIVERGENCE POINTS")
            text_report.append("-" * 50)
            
            for register, failures in list(report['debug_output_failures'].items())[:5]:
                text_report.append(f"\n{register}:")
                for failure in failures[:2]:  # Show first 2 failures
                    text_report.append(f"  Address: {failure['address']}")
                    text_report.append(f"  Success rate: {failure['success_rate']:.2%}")
                    
                    # Show divergence point
                    if failure.get('divergence_point'):
                        div = failure['divergence_point']
                        text_report.append(f"  🔍 FIRST DIVERGENCE: {div.get('position_description', f'position {div['position']}')}")
                        text_report.append(f"     Expected: {div['hex_expected']} | Actual: {div['hex_actual']} | Type: {div['reason']}")
                    
                    text_report.append(f"  Expected: {failure['expected_sequence']}")
                    text_report.append(f"  Observed: {failure['observed_sequence']}")
        
        text_report.append("\n" + "="*80)
        text_report.append("💡 DEBUGGING HINTS:")
        text_report.append("• Focus on registers with early divergences (positions 0-2)")
        text_report.append("• Value mismatches often indicate logic errors")
        text_report.append("• Length mismatches may suggest timing or sequence issues")
        text_report.append("• Fail-only divergences point to bug trigger locations")
        text_report.append("• Transition divergences show where state machines go wrong")
        text_report.append("="*80)
        
        return "\n".join(text_report)']}")
                            text_report.append(f"   Expected: {divergence['hex_expected']} | Actual: {divergence['hex_actual']}")
                        
                        # Show sequence samples
                        expected_hex = ' '.join([f"0x{val:x}" for val in expected[:8]])
                        actual_hex = ' '.join([f"0x{val:x}" for val in actual[:8]])
                        text_report.append(f"   EXPECTED SEQ: {expected_hex}{'...' if len(expected) > 8 else ''}")
                        text_report.append(f"   OBSERVED SEQ: {actual_hex}{'...' if len(actual) > 8 else ''}")
                    
                    elif root_cause_type == 'fail_only_divergence':
                        div_point = error.get('divergence_point', 'unknown')
                        text_report.append(f"   🔍 DIVERGENCE at position: {div_point}")
                        if 'common_prefix' in error:
                            prefix_hex = ' '.join([f"0x{val:x}" for val in error['common_prefix'][:5]])
                            text_report.append(f"   Stable prefix: {prefix_hex}{'...' if len(error['common_prefix']) > 5 else ''}")
        
        # Debug output analysis with divergence points
        if 'debug_output_failures' in report and report['debug_output_failures']:
            text_report.append("\nDEBUG OUTPUT ANALYSIS WITH DIVERGENCE POINTS")
            text_report.append("-" * 50)
            
            for register, failures in list(report['debug_output_failures'].items())[:5]:
                text_report.append(f"\n{register}:")
                for failure in failures[:2]:  # Show first 2 failures
                    text_report.append(f"  Address: {failure['address']}")
                    text_report.append(f"  Success rate: {failure['success_rate']:.2%}")
                    
                    # Show divergence point
                    if failure.get('divergence_point'):
                        div = failure['divergence_point']
                        text_report.append(f"  🔍 FIRST DIVERGENCE: {div.get('position_description', f'position {div['position']}')}")
                        text_report.append(f"     Expected: {div['hex_expected']} | Actual: {div['hex_actual']} | Type: {div['reason']}")
                    
                    text_report.append(f"  Expected: {failure['expected_sequence']}")
                    text_report.append(f"  Observed: {failure['observed_sequence']}")
        
        text_report.append("\n" + "="*80)
        text_report.append("💡 DEBUGGING HINTS:")
        text_report.append("• Focus on registers with early divergences (positions 0-2)")
        text_report.append("• Value mismatches often indicate logic errors")
        text_report.append("• Length mismatches may suggest timing or sequence issues")
        text_report.append("• Fail-only divergences point to bug trigger locations")
        text_report.append("="*80)
        
        return "\n".join(text_report)
    
    def save_report(self, report, output_dir="debug_reports"):
        """Save the enhanced report to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = output_path / f"enhanced_debug_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save text report
        text_file = output_path / f"enhanced_debug_report_{timestamp}.txt"
        text_report = self.format_report_text(report)
        with open(text_file, 'w') as f:
            f.write(text_report)
        
        # Save divergence summary CSV for analysis
        csv_file = output_path / f"divergence_summary_{timestamp}.csv"
        self.save_divergence_csv(report, csv_file)
        
        if self.logger:
            self.logger.info(f"Enhanced debug reports saved to:")
            self.logger.info(f"  JSON: {json_file}")
            self.logger.info(f"  Text: {text_file}")
            self.logger.info(f"  CSV:  {csv_file}")
        else:
            print(f"Enhanced reports saved to:")
            print(f"  JSON: {json_file}")
            print(f"  Text: {text_file}")
            print(f"  CSV:  {csv_file}")
        
        return json_file, text_file, csv_file
    
    def save_divergence_csv(self, report, csv_file):
        """Save divergence points as CSV for further analysis"""
        try:
            import csv
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Register', 'Divergence_Position', 'Reason', 'Expected_Value', 'Actual_Value', 'Source'])
                
                divergence_data = report.get('divergence_analysis', {})
                for register, divergences in divergence_data.items():
                    for div in divergences:
                        writer.writerow([
                            register,
                            div['position'],
                            div['reason'],
                            div['expected_hex'],
                            div['actual_hex'],
                            div.get('source', 'debug_output')
                        ])
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not save CSV file: {e}")


def main():
    """Main function for standalone usage and testing"""
    print("Enhanced Debug Hints Reporter with Transition Divergence Point Detection")
    print("=" * 70)
    
    # Example usage
    reporter = DebugHintsReporter()
    
    # Example debug output for testing transition patterns
    sample_debug_output = """
ring_cstate.ring_cstate_control.target_ring_cstate  101  OBSERVED SEQ: 0x1 0x2 0x99 0x4 0x5  EXPECTED SEQ: (2 out of 5) 0x1 0x2 0x3 0x4 0x5
hvm_compressed_data.state  102  OBSERVED SEQ: 0xa 0xb  EXPECTED SEQ: (1 out of 3) 0xa 0xb 0xc
cbb_cstate.m_next_state  103  OBSERVED SEQ: 0x0 0x1 0x2 0x3 0x4 0x10 0xa 0xb 0xc 0xd 0xe 0x0  EXPECTED SEQ: (11 out of 17) 0x0 0x1 0x2 0x3 0x4 0x5 0x6 0x7 0x10 0x8 0x9 0xa 0xb 0xc 0xd 0xe 0x0
"""
    
    print("Testing with sample debug output including transition patterns...")
    reporter.parse_debug_output(sample_debug_output)
    
    # Generate sample sequence analysis data to test transition divergence
    sample_sequence_analysis = {
        'summary': {
            'total_failing_tests': 9,
            'tests_with_issues': 9,
            'total_root_cause_candidates': 3,
            'reference_model_errors': 2,
            'fail_only_divergence_errors': 1
        },
        'ai_analysis': {
            'most_critical_errors': [
                {
                    'register': 'cbb_cstate.m_next_state',
                    'occurrence_count': 9,
                    'significance_score': 1.000,
                    'root_cause_type': 'reference_model_mismatch',
                    'expected_sequence': [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x10, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0x0],
                    'actual_sequence': [0x0, 0x1, 0x2, 0x3, 0x4, 0x10, 0xa, 0xb, 0xc, 0xd, 0xe, 0x0],
                    'error_category': 'different_transition_path'
                },
                {
                    'register': 'power_control.voltage_state',
                    'occurrence_count': 5,
                    'significance_score': 0.800,
                    'root_cause_type': 'reference_model_mismatch',
                    'expected_sequence': [0x10, 0x20, 0x30, 0x40, 0x50],
                    'actual_sequence': [0x10, 0x20, 0x99, 0x40, 0x50],
                    'error_category': 'value_mismatch'
                }
            ]
        }
    }
    
    # Generate enhanced report with transition analysis
    report = reporter.generate_summary_report()
    
    # Add the sample sequence analysis to test transition formatting
    report['sequence_analysis'] = sample_sequence_analysis
    
    # Show formatted text report
    text_report = reporter.format_report_text(report)
    print("\nGenerated Report:")
    print(text_report)
    
    # Test transition divergence detection specifically
    print("\n" + "="*70)
    print("TESTING TRANSITION DIVERGENCE DETECTION:")
    print("="*70)
    
    expected_transitions = "0x0→0x1→0x2→0x3→0x4→0x5→0x6→0x7→0x10→0x8→0x9→0xa→0xb→0xc→0xd→0xe→0x0"
    actual_transitions = "0x0→0x1→0x2→0x3→0x4→0x10→0xa→0xb→0xc→0xd→0xe→0x0"
    
    print(f"Expected: {expected_transitions}")
    print(f"Actual:   {actual_transitions}")
    
    transition_divergence = reporter.find_transition_divergence_point(expected_transitions, actual_transitions)
    if transition_divergence:
        print(f"\n🔍 Transition Divergence Found:")
        print(f"   Position: {transition_divergence['transition_position']}")
        print(f"   Reason: {transition_divergence['reason']}")
        print(f"   From state: {transition_divergence['from_state']}")
        print(f"   Expected next: {transition_divergence['expected_next']}")
        print(f"   Actual next: {transition_divergence['actual_next']}")
        print(f"   📝 {transition_divergence['divergence_description']}")
    else:
        print("No transition divergence found")
    
    # Save reports
    json_file, text_file, csv_file = reporter.save_report(report, "test_output")
    
    print(f"\n✅ Test completed successfully!")
    print(f"Enhanced debug report now includes:")
    print(f"   • First divergence points for all sequence mismatches")
    print(f"   • Special transition pattern detection")
    print(f"   • Transition divergence analysis (0x4→0x10 instead of 0x4→0x5)")
    print(f"   • Enhanced formatting for state machine debugging")
    print(f"Files saved for review.")


if __name__ == "__main__":
    main()