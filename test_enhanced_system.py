#!/usr/bin/env python3
"""
Test script to verify the enhanced system is working
"""
import sys
import logging

# Test imports
try:
    from tokenizer import OptimizedLogTokenizer
    print("✓ tokenizer.py imported successfully")
except ImportError as e:
    print(f"✗ Failed to import tokenizer: {e}")
    sys.exit(1)

try:
    from sequence_analyzer import RegisterSequenceAnalyzer
    print("✓ sequence_analyzer.py imported successfully") 
except ImportError as e:
    print(f"✗ Failed to import sequence_analyzer: {e}")
    sys.exit(1)

try:
    from debug_report_gen import DebugHintsReporter
    print("✓ debug_report_gen.py imported successfully")
except ImportError as e:
    print(f"✗ Failed to import debug_report_gen: {e}")
    sys.exit(1)

# Test that debug reporter has the enhanced methods
debug_reporter = DebugHintsReporter()
if hasattr(debug_reporter, 'set_sequence_analyzer'):
    print("✓ DebugHintsReporter has set_sequence_analyzer method")
else:
    print("✗ DebugHintsReporter missing set_sequence_analyzer method")
    sys.exit(1)

# Test tokenizer on a single file
def test_single_file():
    print("\nTesting tokenizer on a single file...")
    
    log_file = "/nfs/site/disks/oramsale_wa02/cbb_ptracker_logs/passing_tests/soc_pm_AI_assist_noam_or_test_FGP.57_ptracker.log.gz"
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test tokenizer
    tokenizer = OptimizedLogTokenizer(logger=logger)
    
    # Create test data structure
    test_data = {
        'test_1': {
            'log_path': log_file,
            'test_id': 'test_1'
        }
    }
    
    # Tokenize
    result = tokenizer.tokenize_logs(test_data)
    
    # Check results
    test_result = result['test_1']
    if 'tokens' in test_result and 'error' not in test_result['tokens']:
        tokens = test_result['tokens']
        sequences = test_result.get('register_sequences', {})
        
        print(f"✓ Tokenization successful!")
        print(f"  Register accesses: {len(tokens.get('registers', []))}")
        print(f"  Register sequences: {len(sequences)}")
        
        # Show some example sequences
        if sequences:
            print(f"  Example sequences:")
            for i, (register, sequence) in enumerate(list(sequences.items())[:3]):
                values = [item['value'] for item in sequence[:5]]  # First 5 values
                print(f"    {register}: {values}...")
        
        return True
    else:
        print(f"✗ Tokenization failed: {test_result}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced Ptracker Analysis System")
    print("=" * 50)
    
    # Test imports
    print("Testing imports...")
    
    # Test tokenizer
    if test_single_file():
        print("\n✓ Enhanced system is working correctly!")
        print("\nYou should now run:")
        print("python tmp_main.py --passing-dir /path/to/passing --failing-dir /path/to/failing --generate-debug-report")
    else:
        print("\n✗ Enhanced system has issues - check your file updates")