#!/usr/bin/env python3
"""
TRANSITION-FOCUSED ptracker analysis - analyzes STATE TRANSITIONS not raw values.
Replace your standalone_main.py with this file for proper transition analysis.
ADDED: Reference model save/load capability
"""
import os
import logging
import argparse
import json
import glob
import gzip
import re
import pickle
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter


def setup_simple_logging(output_dir, log_level='INFO'):
    """Setup simple logging without complex logger classes"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - TRANSITION - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_path / f"transition_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    return logging.getLogger('transition_ptracker')


class TransitionLogProcessor:
    """TRANSITION-FOCUSED processor that analyzes state changes, not raw register values"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Ptracker log patterns
        self.patterns = [
            # Format 1: Multi-line format
            re.compile(
                r'\[\s*(\d+)\]\s+Timestamp:\s+(\d+)\s+'
                r'Action:\s+(\w+)\s+'
                r'Address:\s+(0x[0-9a-fA-F]+)\s+'
                r'Size:\s+(\w+)\s+'
                r'Register:\s+(.+?)\s+'
                r'Value:\s+(0x[0-9a-fA-F]+)',
                re.MULTILINE | re.DOTALL
            ),
            
            # Format 2: Single line format  
            re.compile(
                r'\[(\d+)\]\s+(MEM_[RW]|CR_[RW])\s+\[(0x[0-9a-fA-F]+)\]\s+(\w+)\s+([^\s].*?)\s+(0x[0-9a-fA-F]+)'
            ),
        ]
        
        self.reference_transitions = {}
    
    def save_reference_model(self, model_path):
        """Save the reference transition model to disk"""
        try:
            model_dir = Path(model_path).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'reference_transitions': self.reference_transitions,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'transition_focused'
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Reference model saved to {model_path}")
            self.logger.info(f"Model contains {len(self.reference_transitions)} transition patterns")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save reference model: {e}")
            return False
    
    def load_reference_model(self, model_path):
        """Load the reference transition model from disk"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file does not exist: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.reference_transitions = model_data.get('reference_transitions', {})
            timestamp = model_data.get('timestamp', 'unknown')
            model_type = model_data.get('model_type', 'unknown')
            
            self.logger.info(f"Reference model loaded from {model_path}")
            self.logger.info(f"Model created: {timestamp}")
            self.logger.info(f"Model type: {model_type}")
            self.logger.info(f"Loaded {len(self.reference_transitions)} transition patterns")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load reference model: {e}")
            return False
    
    def extract_transitions_from_sequence(self, values):
        """
        Extract state transitions from a sequence of register values
        
        Input:  [0x0, 0x0, 0x1, 0x1, 0x1, 0x0]
        Output: ['0x0→0x1', '0x1→0x0']  (only actual state changes)
        
        Parameters:
        values (list): List of register values
        
        Returns:
        list: List of transition strings
        """
        if len(values) < 2:
            return []
        
        transitions = []
        prev_value = values[0]
        
        for curr_value in values[1:]:
            if curr_value != prev_value:  # Only record actual state changes
                transition = f"0x{prev_value:x}→0x{curr_value:x}"
                transitions.append(transition)
                prev_value = curr_value
        
        return transitions
    
    def transitions_to_string(self, transitions):
        """
        Convert transition list to readable string
        
        Input:  ['0x0→0x1', '0x1→0x0']
        Output: "0x0→0x1→0x0"
        """
        if not transitions:
            return "No transitions"
        
        if len(transitions) == 1:
            return transitions[0]
        
        # Build transition chain
        result = transitions[0].split('→')[0]  # Start state
        for trans in transitions:
            end_state = trans.split('→')[1]
            result += f"→{end_state}"
        
        return result
    
    def find_log_pattern(self, content):
        """Test which pattern works for this log content"""
        for i, pattern in enumerate(self.patterns):
            matches = list(pattern.finditer(content))
            if matches:
                self.logger.info(f"Using pattern {i+1} - found {len(matches)} matches")
                return pattern, i
        
        self.logger.warning("No pattern matched")
        return None, -1
    
    def extract_register_transitions(self, log_files, file_type="unknown"):
        """Extract register sequences and convert to transitions"""
        self.logger.info(f"Extracting {file_type} transitions from {len(log_files)} files")
        
        all_transitions = {}
        processed_count = 0
        
        for log_file in log_files:
            try:
                transitions = self._process_single_log_for_transitions(log_file)
                if transitions:
                    test_id = os.path.basename(log_file)
                    all_transitions[test_id] = transitions
                    processed_count += 1
                    
                    if processed_count % 5 == 0:
                        self.logger.info(f"   Processed {processed_count}/{len(log_files)} files...")
                        
            except Exception as e:
                self.logger.warning(f"Failed to process {log_file}: {e}")
        
        self.logger.info(f"Successfully processed {processed_count} {file_type} tests")
        return all_transitions
    
    def _process_single_log_for_transitions(self, log_path):
        """Process a single log file and extract transitions"""
        
        # Read file
        try:
            if log_path.endswith('.gz'):
                with gzip.open(log_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            else:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
        except Exception as e:
            self.logger.warning(f"Cannot read {log_path}: {e}")
            return {}
        
        # Find working pattern
        pattern, pattern_type = self.find_log_pattern(content)
        if not pattern:
            return {}
        
        # Extract register accesses
        sequences = defaultdict(list)
        
        for match in pattern.finditer(content):
            try:
                if pattern_type == 0:  # Multi-line format
                    entry_id, timestamp, action, address, size, register, value = match.groups()
                elif pattern_type == 1:  # Single line format
                    entry_id, action, address, size, register, value = match.groups()
                    timestamp = entry_id
                else:
                    continue
                
                # Clean register name
                register = ' '.join(register.split()) if register else 'unknown'
                
                # Convert value
                value_int = int(value, 16) if value else 0
                
                # Store in sequence
                sequences[register].append({
                    'timestamp': int(timestamp) if timestamp else 0,
                    'value': value_int,
                    'action': action or 'UNKNOWN'
                })
                
            except (ValueError, AttributeError, TypeError):
                continue
        
        # Convert sequences to transitions
        register_transitions = {}
        for register, sequence in sequences.items():
            if len(sequence) >= 2:  # Need at least 2 values for a transition
                sequence.sort(key=lambda x: x['timestamp'])
                values = [item['value'] for item in sequence]
                
                # Extract transitions
                transitions = self.extract_transitions_from_sequence(values)
                
                if transitions:  # Only keep registers with actual state changes
                    register_transitions[register] = {
                        'raw_sequence': sequence,
                        'values': values,
                        'transitions': transitions,
                        'transition_string': self.transitions_to_string(transitions)
                    }
        
        self.logger.debug(f"Extracted {len(register_transitions)} transition patterns from {os.path.basename(log_path)}")
        return register_transitions
    
    def build_reference_transition_model(self, passing_transitions):
        """Build reference patterns from passing test transitions"""
        self.logger.info("Building reference transition model...")
        
        # Collect transitions by register
        register_transition_patterns = defaultdict(list)
        
        for test_data in passing_transitions.values():
            for register, transition_data in test_data.items():
                transitions = transition_data['transitions']
                if transitions:
                    register_transition_patterns[register].append(transitions)
        
        # Build patterns
        pattern_count = 0
        for register, transition_lists in register_transition_patterns.items():
            if len(transition_lists) >= 1:
                
                # Find most common transition pattern
                transition_counter = Counter()
                for trans_list in transition_lists:
                    transition_counter[tuple(trans_list)] += 1
                
                if transition_counter:
                    most_common_transitions, count = transition_counter.most_common(1)[0]
                    
                    self.reference_transitions[register] = {
                        'expected_transitions': list(most_common_transitions),
                        'expected_transition_string': self.transitions_to_string(list(most_common_transitions)),
                        'occurrences': count,
                        'total_tests': len(transition_lists)
                    }
                    pattern_count += 1
        
        self.logger.info(f"Built {pattern_count} reference transition patterns")
        return pattern_count > 0
    
    def analyze_failing_transitions(self, failing_transitions):
        """Analyze failing transitions against reference"""
        self.logger.info("Analyzing failing transitions against reference patterns...")
        
        analysis_results = {}
        error_patterns = []
        
        for test_id, test_transitions in failing_transitions.items():
            test_mismatches = {}
            
            for register, transition_data in test_transitions.items():
                if register in self.reference_transitions:
                    actual_transitions = transition_data['transitions']
                    expected_transitions = self.reference_transitions[register]['expected_transitions']
                    
                    # Compare transitions (not raw values!)
                    if actual_transitions != expected_transitions:
                        
                        # Analyze the transition difference
                        analysis = self._analyze_transition_difference(expected_transitions, actual_transitions)
                        
                        # Create transition-focused mismatch data
                        mismatch_data = {
                            'expected_transitions': expected_transitions,
                            'actual_transitions': actual_transitions,
                            'expected_transition_string': self.transitions_to_string(expected_transitions),
                            'actual_transition_string': self.transitions_to_string(actual_transitions),
                            'transition_analysis': analysis,
                            'raw_values': {
                                'expected': transition_data.get('values', []),
                                'actual': transition_data.get('values', [])
                            }
                        }
                        test_mismatches[register] = mismatch_data
                        
                        error_patterns.append({
                            'register': register,
                            'expected_transitions': expected_transitions,
                            'actual_transitions': actual_transitions,
                            'expected_transition_string': self.transitions_to_string(expected_transitions),
                            'actual_transition_string': self.transitions_to_string(actual_transitions),
                            'transition_analysis': analysis,
                            'test_id': test_id,
                            'error_category': analysis['type'],
                            'significance_score': analysis.get('severity_score', 0.5)
                        })
            
            if test_mismatches:
                analysis_results[test_id] = {'transition_mismatches': test_mismatches}
        
        self.logger.info(f"Found transition mismatches in {len(analysis_results)} tests")
        return analysis_results, error_patterns
    
    def _analyze_transition_difference(self, expected_transitions, actual_transitions):
        """
        Analyze the difference between expected and actual transitions
        """
        analysis = {
            'type': 'unknown',
            'description': '',
            'missing_transitions': [],
            'extra_transitions': [],
            'severity': 'low',
            'severity_score': 0.5
        }
        
        if not expected_transitions and not actual_transitions:
            analysis.update({
                'type': 'no_transitions',
                'description': 'No state transitions in either sequence',
                'severity': 'low',
                'severity_score': 0.1
            })
            return analysis
        
        if not actual_transitions:
            analysis.update({
                'type': 'missing_all_transitions',
                'description': f'Expected {len(expected_transitions)} transitions, but got none',
                'missing_transitions': expected_transitions,
                'severity': 'critical',
                'severity_score': 1.0
            })
            return analysis
        
        if not expected_transitions:
            analysis.update({
                'type': 'unexpected_transitions',
                'description': f'Got {len(actual_transitions)} unexpected transitions',
                'extra_transitions': actual_transitions,
                'severity': 'high',
                'severity_score': 0.8
            })
            return analysis
        
        # Compare transition sequences
        if expected_transitions == actual_transitions:
            analysis.update({
                'type': 'identical_transitions',
                'description': 'Transitions are identical',
                'severity': 'none',
                'severity_score': 0.0
            })
            return analysis
        
        # Check if actual is a subset of expected (incomplete transitions)
        if len(actual_transitions) < len(expected_transitions):
            if expected_transitions[:len(actual_transitions)] == actual_transitions:
                missing = expected_transitions[len(actual_transitions):]
                analysis.update({
                    'type': 'incomplete_transitions',
                    'description': f'Missing {len(missing)} final transitions',
                    'missing_transitions': missing,
                    'severity': 'high',
                    'severity_score': 0.9
                })
                return analysis
        
        # Check if expected is a subset of actual (extra transitions)
        if len(actual_transitions) > len(expected_transitions):
            if actual_transitions[:len(expected_transitions)] == expected_transitions:
                extra = actual_transitions[len(expected_transitions):]
                analysis.update({
                    'type': 'extra_transitions',
                    'description': f'Has {len(extra)} extra transitions',
                    'extra_transitions': extra,
                    'severity': 'medium',
                    'severity_score': 0.6
                })
                return analysis
        
        # Different transition paths
        analysis.update({
            'type': 'different_transition_path',
            'description': 'Completely different state transition sequence',
            'severity': 'critical',
            'severity_score': 1.0
        })
        return analysis
    
    def generate_transition_analysis(self, error_patterns):
        """Generate AI analysis focused on transitions"""
        self.logger.info("🤖 Generating transition-focused AI analysis...")
        
        # Aggregate patterns by register and transition sequence
        pattern_groups = defaultdict(lambda: {
            'register': '',
            'expected_transitions': [],
            'actual_transitions': [],
            'expected_transition_string': '',
            'actual_transition_string': '',
            'transition_analysis': {},
            'occurrence_count': 0,
            'affected_tests': [],
            'error_category': 'unknown',
            'significance_score': 0.0
        })
        
        for pattern in error_patterns:
            key = (
                pattern['register'],
                tuple(pattern['expected_transitions']),
                tuple(pattern['actual_transitions'])
            )
            
            group = pattern_groups[key]
            if group['occurrence_count'] == 0:  # First occurrence
                group.update({
                    'register': pattern['register'],
                    'expected_transitions': pattern['expected_transitions'],
                    'actual_transitions': pattern['actual_transitions'],
                    'expected_transition_string': pattern['expected_transition_string'],
                    'actual_transition_string': pattern['actual_transition_string'],
                    'transition_analysis': pattern['transition_analysis'],
                    'error_category': pattern['error_category'],
                    'significance_score': pattern['significance_score']
                })
            
            group['occurrence_count'] += 1
            group['affected_tests'].append(pattern['test_id'])
        
        # Sort by significance and occurrence
        sorted_errors = sorted(
            pattern_groups.values(),
            key=lambda x: (x['significance_score'], x['occurrence_count']),
            reverse=True
        )
        
        return {
            'transition_analysis': {
                'most_critical_errors': sorted_errors,
                'summary': {
                    'total_transition_errors': len(sorted_errors),
                    'high_severity_errors': len([e for e in sorted_errors if e['significance_score'] >= 0.8])
                }
            }
        }


def save_transition_results(output_dir, failing_transitions, mismatch_report, logger):
    """Save transition analysis results"""
    logger.info("💾 Saving transition analysis results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary file
    summary = {
        "timestamp": timestamp,
        "analysis_type": "transition_focused",
        "total_failing_tests": len(failing_transitions),
        "tests_with_transitions": len([t for t in failing_transitions.values() if t])
    }
    
    summary_file = Path(output_dir) / f"transition_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Transition hints file
    hints_file = Path(output_dir) / f"transition_hints_{timestamp}.txt"
    generate_transition_hints_file(hints_file, mismatch_report)
    
    logger.info(f"✅ Transition results saved:")
    logger.info(f"   📋 Summary: {summary_file.name}")
    logger.info(f"   📝 Hints: {hints_file.name}")
    
    return summary_file, hints_file


def generate_transition_hints_file(hints_file, mismatch_report):
    """Generate transition-focused hints"""
    lines = [
        "="*80,
        "🚀 TRANSITION-FOCUSED PTRACKER DEBUG HINTS",
        "="*80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Analyzes STATE TRANSITIONS instead of raw register values",
        "Shows state changes like 0x0→0x1→0x0 for easier debugging",
        ""
    ]
    
    if not mismatch_report or 'transition_analysis' not in mismatch_report:
        lines.extend([
            "❌ NO TRANSITION ANALYSIS RESULTS",
            "   • Check if log files contain state transitions",
            "   • Verify registers have actual state changes",
            ""
        ])
    else:
        transition_data = mismatch_report['transition_analysis']
        errors = transition_data.get('most_critical_errors', [])
        
        if errors:
            lines.extend([
                f"🎯 FOUND {len(errors)} TRANSITION ERROR PATTERNS",
                f"   🔥 High severity errors: {transition_data.get('summary', {}).get('high_severity_errors', 0)}",
                "",
                "📋 TRANSITION ERROR PATTERNS (by severity):",
                "="*80
            ])
            
            for i, error in enumerate(errors[:15], 1):
                register = error.get('register', 'unknown')
                count = error.get('occurrence_count', 0)
                significance = error.get('significance_score', 0)
                
                expected_trans = error.get('expected_transition_string', 'Unknown')
                actual_trans = error.get('actual_transition_string', 'Unknown')
                analysis = error.get('transition_analysis', {})
                
                lines.extend([
                    "",
                    f"🔍 TRANSITION PATTERN #{i}",
                    f"   Register: {register}",
                    f"   Occurs in: {count} tests",
                    f"   Severity: {significance:.3f}",
                    "",
                    f"   Expected transitions: {expected_trans}",
                    f"   Actual transitions:   {actual_trans}",
                    ""
                ])
                
                # Add transition analysis details
                if analysis:
                    issue_type = analysis.get('type', 'unknown')
                    description = analysis.get('description', 'Unknown issue')
                    
                    lines.extend([
                        f"   Issue: {description}",
                        f"   Category: {issue_type}",
                    ])
                    
                    missing = analysis.get('missing_transitions', [])
                    extra = analysis.get('extra_transitions', [])
                    
                    if missing:
                        lines.append(f"   Missing: {', '.join(missing)}")
                    if extra:
                        lines.append(f"   Extra: {', '.join(extra)}")
                    
                    lines.append("")
        else:
            lines.extend([
                "✅ NO TRANSITION ERROR PATTERNS DETECTED",
                "   All state transitions match expected patterns",
                ""
            ])
    
    lines.extend([
        "="*80,
        "💡 TRANSITION DEBUGGING GUIDE:",
        "   • incomplete_transitions: State machine started but stopped early",
        "   • missing_all_transitions: State machine never started",
        "   • different_transition_path: Wrong state sequence",
        "   • extra_transitions: State machine continued too long",
        "   • Focus on TRANSITIONS, not raw register values",
        "="*80
    ])
    
    with open(hints_file, 'w') as f:
        f.write('\n'.join(lines))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Transition-Focused Ptracker Analysis with Model Save/Load')
    parser.add_argument('--passing-dir', required=True, help='Directory with passing logs')
    parser.add_argument('--failing-dir', required=True, help='Directory with failing logs')
    parser.add_argument('--output-dir', default='transition_output', help='Output directory')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--max-files', type=int, help='Limit files for testing')
    
    # Reference model options
    parser.add_argument('--reference-model', help='Path to save/load reference model')
    parser.add_argument('--save-model', action='store_true', help='Save reference model after building')
    parser.add_argument('--load-model', action='store_true', help='Load existing reference model')
    
    return parser.parse_args()


def main():
    """Transition-focused main function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_simple_logging(args.output_dir, args.log_level)
    
    logger.info("🚀 Starting TRANSITION-FOCUSED Ptracker Analysis")
    logger.info("   ✨ Analyzes STATE TRANSITIONS not raw register values")
    logger.info("   🎯 Shows 0x0→0x1→0x0 patterns for easier debugging")
    if args.reference_model:
        logger.info(f"   💾 Reference model: {args.reference_model}")
    
    # Find files
    logger.info("Finding log files...")
    passing_files = glob.glob(os.path.join(args.passing_dir, "*.gz")) + glob.glob(os.path.join(args.passing_dir, "*.log"))
    failing_files = glob.glob(os.path.join(args.failing_dir, "*.gz")) + glob.glob(os.path.join(args.failing_dir, "*.log"))
    
    if args.max_files:
        passing_files = passing_files[:args.max_files]
        failing_files = failing_files[:args.max_files]
    
    logger.info(f"📊 Found {len(passing_files)} passing, {len(failing_files)} failing")
    
    if not failing_files:
        logger.error("❌ Need failing logs")
        return 1
    
    # Initialize processor
    processor = TransitionLogProcessor(logger)
    
    # Check if we should load an existing model
    if args.load_model and args.reference_model:
        logger.info(f"📥 Loading existing reference model from {args.reference_model}")
        if processor.load_reference_model(args.reference_model):
            logger.info("✅ Reference model loaded successfully")
            # Skip building model, go straight to analysis
            passing_transitions = {}
        else:
            logger.warning("⚠️  Failed to load model, will build new one")
            args.load_model = False
    
    # Build reference model if not loaded
    if not args.load_model:
        if not passing_files:
            logger.error("❌ Need passing logs to build reference model")
            return 1
        
        # Extract transitions from passing logs
        logger.info("🔨 STEP 1: Extracting transitions from passing tests")
        passing_transitions = processor.extract_register_transitions(passing_files, "passing")
        
        if not passing_transitions:
            logger.error("❌ No transitions in passing logs")
            return 1
        
        # Build reference transition model
        logger.info("📚 STEP 2: Building reference transition model")
        success = processor.build_reference_transition_model(passing_transitions)
        if not success:
            logger.error("❌ Failed to build reference transition model")
            return 1
        
        # Save model if requested
        if args.save_model and args.reference_model:
            logger.info(f"💾 Saving reference model to {args.reference_model}")
            if processor.save_reference_model(args.reference_model):
                logger.info("✅ Reference model saved successfully")
            else:
                logger.warning("⚠️  Failed to save reference model")
    
    # Extract transitions from failing logs
    logger.info("🔍 STEP 3: Extracting transitions from failing tests")
    failing_transitions = processor.extract_register_transitions(failing_files, "failing")
    
    if not failing_transitions:
        logger.error("❌ No transitions in failing logs")
        return 1
    
    # Analyze transition differences
    logger.info("🤖 STEP 4: Transition analysis")
    analysis_results, error_patterns = processor.analyze_failing_transitions(failing_transitions)
    
    # Generate transition report
    mismatch_report = processor.generate_transition_analysis(error_patterns)
    
    # Save results
    logger.info("💾 STEP 5: Saving transition results")
    summary_file, hints_file = save_transition_results(
        args.output_dir, failing_transitions, mismatch_report, logger
    )
    
    # Print final summary
    if mismatch_report and 'transition_analysis' in mismatch_report:
        transition_data = mismatch_report['transition_analysis']
        errors = transition_data.get('most_critical_errors', [])
        
        print("\n" + "="*80)
        print("🎉 TRANSITION-FOCUSED ANALYSIS COMPLETE!")
        print("="*80)
        print(f"📊 Results:")
        print(f"   • Failing tests processed: {len(failing_transitions)}")
        print(f"   • Tests with transition differences: {len(analysis_results)}")
        print(f"   • Transition error patterns: {len(errors)}")
        print(f"   • High severity errors: {transition_data.get('summary', {}).get('high_severity_errors', 0)}")
        
        if errors:
            print(f"\n🔍 Top 3 Critical Transition Errors:")
            for i, error in enumerate(errors[:3], 1):
                register = error.get('register', 'unknown')
                expected = error.get('expected_transition_string', 'Unknown')
                actual = error.get('actual_transition_string', 'Unknown')
                analysis = error.get('transition_analysis', {})
                
                print(f"   {i}. {register}")
                print(f"      Expected: {expected}")
                print(f"      Actual:   {actual}")
                print(f"      Issue:    {analysis.get('description', 'Unknown')}")
        
        # Reference model info
        if args.reference_model:
            if args.save_model:
                print(f"\n💾 Reference Model:")
                print(f"   • Saved to: {args.reference_model}")
                print(f"   • Can be reused with --load-model flag")
            elif args.load_model:
                print(f"\n📥 Reference Model:")
                print(f"   • Loaded from: {args.reference_model}")
                print(f"   • Used existing transition patterns for analysis")
        
        print(f"\n📁 Files:")
        print(f"   • Summary: {summary_file}")
        print(f"   • Transition Hints: {hints_file}")
        
        print(f"\n💡 Key Features:")
        print(f"   • Analyzes TRANSITIONS (0x0→0x1) not raw values")
        print(f"   • Shows missing/extra state changes clearly")
        print(f"   • Focuses on state machine behavior")
        print(f"   • Reference model save/load capability")
        print("="*80)
    else:
        logger.error("❌ Transition analysis failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)