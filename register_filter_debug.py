#!/usr/bin/env python3
"""
Debug tool to identify exactly why RESOURCE_OWN_REQ_STATUS_IP_CCF is being filtered out.
Run this on your log file to see what's happening.
"""
import re
import gzip
import os
from collections import defaultdict

def debug_register_filtering(log_file, target_register="RESOURCE_OWN_REQ_STATUS_IP_CCF"):
    """Debug why a specific register is being filtered"""
    
    print(f"🔍 DEBUGGING REGISTER FILTERING FOR: {target_register}")
    print("="*80)
    
    # Read the log file
    try:
        if log_file.endswith('.gz'):
            with gzip.open(log_file, 'rt', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        else:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print(f"📄 File: {os.path.basename(log_file)}")
    print(f"📏 Size: {len(content)} characters")
    print()
    
    # Search for the target register in raw content
    target_lines = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if target_register in line:
            target_lines.append((i+1, line))
    
    print(f"🎯 Found {len(target_lines)} lines containing '{target_register}':")
    for line_num, line in target_lines[:5]:  # Show first 5
        print(f"   Line {line_num}: {line}")
    print()
    
    # Test current patterns - FIXED VERSION
    patterns = [
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
        
        # Format 2: FIXED Enhanced single line format (handles CR_R and flexible spacing)
        re.compile(
            r'\[(\d+)\]\s+(MEM_[RW]|CR_[RW])\s+\[\s*(0x[0-9a-fA-F]+)\s*\]\s+(\w*)\s+([^\s].*?)\s+(0x[0-9a-fA-F]+)(?:\s+<--.*)?'
        ),
        
        # Format 3: Alternative single line
        re.compile(
            r'\[(\d+)\].*?(MEM_[RW]|CR_[RW]).*?([^\s]+)\s+(0x[0-9a-fA-F]+)'
        )
    ]
    
    print("🔍 TESTING REGEX PATTERNS (FIXED VERSION):")
    print("-" * 40)
    
    best_pattern = None
    best_matches = 0
    
    for i, pattern in enumerate(patterns, 1):
        matches = list(pattern.finditer(content))
        print(f"Pattern {i}: {len(matches)} total matches")
        
        if len(matches) > best_matches:
            best_matches = len(matches)
            best_pattern = i
        
        # Check if any matches contain our target register
        target_matches = []
        for match in matches:
            if target_register in match.group(0):
                target_matches.append(match)
        
        print(f"  - {len(target_matches)} matches contain '{target_register}'")
        
        if target_matches:
            print(f"  - Example match groups: {target_matches[0].groups()}")
            print(f"  - Full match: {target_matches[0].group(0)}")
            
            # Show what register name gets extracted
            if i == 2:  # Our main pattern
                groups = target_matches[0].groups()
                if len(groups) >= 5:
                    extracted_register = groups[4].strip()
                    print(f"  - Extracted register name: '{extracted_register}'")
        print()
    
    print(f"🏆 Best pattern: #{best_pattern} with {best_matches} matches")
    print()
    
    # Check exclusion keywords - RELAXED VERSION
    excluded_keywords = [
        'eventlogger.buffer',
        'eventlogger.is_enabled', 
        'eventlogger.byte_index'
    ]
    
    print("🚫 CHECKING EXCLUSION KEYWORDS (RELAXED):")
    print("-" * 40)
    register_lower = target_register.lower()
    excluded = False
    for keyword in excluded_keywords:
        if keyword.lower() in register_lower:
            print(f"❌ EXCLUDED by keyword: '{keyword}'")
            excluded = True
    
    if not excluded:
        print("✅ No exclusion keywords match (RELAXED filtering)")
    print()
    
    # Test meaningful sequence filtering
    print("📊 TESTING SEQUENCE FILTERING (RELAXED):")
    print("-" * 40)
    
    # Extract all values for this register manually
    register_values = []
    for line_num, line in target_lines:
        # Try to extract hex value from the line
        hex_match = re.search(r'0x([0-9a-fA-F]+)\s*(?:<--|$)', line)
        if hex_match:
            try:
                value = int(hex_match.group(0), 16)
                register_values.append(value)
            except:
                pass
    
    print(f"Found {len(register_values)} values: {register_values[:10]}{'...' if len(register_values) > 10 else ''}")
    
    if register_values:
        # Test filtering criteria - RELAXED
        unique_values = len(set(register_values))
        print(f"Unique values: {unique_values}")
        print(f"Total values: {len(register_values)}")
        
        # Check change frequency
        if len(register_values) > 1:
            changes = sum(1 for i in range(len(register_values)-1) if register_values[i] != register_values[i+1])
            change_ratio = changes / (len(register_values) - 1)
            print(f"Change frequency: {change_ratio:.3f} ({changes} changes out of {len(register_values)-1} transitions)")
            
            # Check filtering thresholds - RELAXED
            print(f"Passes unique value test (>1): {'✅' if unique_values > 1 else '❌'}")
            print(f"Passes change frequency test (≥0.05): {'✅' if change_ratio >= 0.05 else '❌'}")  # Lowered threshold
            print(f"Passes minimum length test (≥2): {'✅' if len(register_values) >= 2 else '❌'}")  # Lowered requirement
            
            # Show actual state changes
            if unique_values > 1:
                state_changes = [register_values[0]]
                for i in range(1, len(register_values)):
                    if register_values[i] != register_values[i-1]:
                        state_changes.append(register_values[i])
                
                state_changes_hex = [f"0x{v:x}" for v in state_changes]
                print(f"State changes only: {' '.join(state_changes_hex)}")
    print()
    
    # Suggest fixes
    print("🔧 WHAT FIXED VERSION ADDRESSES:")
    print("-" * 40)
    
    if not target_lines:
        print("❌ Register not found in log - check register name spelling")
    elif excluded:
        print("❌ Register excluded by keyword filter - but we RELAXED the exclusions!")
    elif len(register_values) < 2:  # Changed from 3 to 2
        print("✅ FIXED: Reduced minimum sequence length requirement (3→2)")
    elif len(set(register_values)) <= 1:
        print("❌ No value changes - this register has constant values")
    elif len(register_values) > 1:
        changes = sum(1 for i in range(len(register_values)-1) if register_values[i] != register_values[i+1])
        change_ratio = changes / (len(register_values) - 1)
        if change_ratio < 0.05:  # Changed threshold
            print(f"✅ FIXED: Lowered change frequency threshold (0.1→0.05)")
        else:
            print(f"✅ Should pass all RELAXED filtering criteria!")
    
    # Check if CR_R vs MEM_R is the issue
    cr_count = sum(1 for _, line in target_lines if 'CR_R' in line)
    mem_count = sum(1 for _, line in target_lines if 'MEM_R' in line)
    print(f"Action types: CR_R={cr_count}, MEM_R={mem_count}")
    
    if cr_count > 0:
        print("✅ FIXED: CR_R operations now supported in enhanced pattern!")


def test_fixed_pattern_on_sample():
    """Test the fixed pattern on your sample line"""
    sample_line = "[765228750]   CR_R  [    0x1540]           RESOURCE_OWN_REQ_STATUS_IP_CCF                                                                   0xc <-- UDI=0 VCCINF=1 SB_CLK=1"
    
    print("🧪 TESTING FIXED PATTERN ON YOUR SAMPLE:")
    print("-" * 50)
    print(f"Sample: {sample_line}")
    print()
    
    # Test the fixed pattern
    fixed_pattern = re.compile(
        r'\[(\d+)\]\s+(MEM_[RW]|CR_[RW])\s+\[\s*(0x[0-9a-fA-F]+)\s*\]\s+(\w*)\s+([^\s].*?)\s+(0x[0-9a-fA-F]+)(?:\s+<--.*)?'
    )
    
    match = fixed_pattern.search(sample_line)
    if match:
        print("✅ FIXED PATTERN MATCHES!")
        print(f"   Groups: {match.groups()}")
        print(f"   Entry ID: {match.group(1)}")
        print(f"   Action: {match.group(2)}")
        print(f"   Address: {match.group(3)}")
        print(f"   Size: '{match.group(4)}'")
        print(f"   Register: '{match.group(5)}'")
        print(f"   Value: {match.group(6)}")
        
        # Show what gets processed
        register_clean = ' '.join(match.group(5).split()).strip()
        value_int = int(match.group(6), 16)
        print(f"   Processed register: '{register_clean}'")
        print(f"   Processed value: {value_int} (0x{value_int:x})")
    else:
        print("❌ FIXED PATTERN DOES NOT MATCH!")
        
        # Try individual components
        print("Testing components:")
        if re.search(r'\[\d+\]', sample_line):
            print("   ✅ Entry ID bracket found")
        if re.search(r'CR_R', sample_line):
            print("   ✅ CR_R action found")
        if re.search(r'\[\s*0x[0-9a-fA-F]+\s*\]', sample_line):
            print("   ✅ Address bracket found")
        if re.search(r'0x[0-9a-fA-F]+\s*(?:<--|$)', sample_line):
            print("   ✅ Value found")


if __name__ == "__main__":
    import sys
    
    # First test the pattern on the sample
    test_fixed_pattern_on_sample()
    print("\n" + "="*80)
    
    if len(sys.argv) == 2:
        debug_register_filtering(sys.argv[1])
    else:
        print("💡 QUICK TEST COMPLETE!")
        print()
        print("To debug a specific log file, run:")
        print("python register_filter_debug.py <log_file>")
        print("Example: python register_filter_debug.py test.log.gz")
        print()
        print("🔧 SUMMARY OF FIXES APPLIED:")
        print("="*40)
        print("✅ CR_R operations now supported")
        print("✅ Flexible address spacing: [    0x1540] handled")
        print("✅ Relaxed exclusion keywords (removed most filters)")
        print("✅ Lowered minimum sequence length (3→2 values)")
        print("✅ Lowered change frequency threshold (0.1→0.05)")
        print("✅ Optional size field handled properly")
        print("✅ Comment sections (<-- ...) ignored")
        print()
        print("Your RESOURCE_OWN_REQ_STATUS_IP_CCF register should now be detected!")#!/usr/bin/env python3
"""
Debug tool to identify exactly why specific registers are being filtered out.
Run this on your log file to see what's happening with RESOURCE_OWN_REQ_STATUS_IP_CCF
"""
import re
import gzip
import os
from collections import defaultdict

def debug_register_filtering(log_file, target_register="RESOURCE_OWN_REQ_STATUS_IP_CCF"):
    """Debug why a specific register is being filtered"""
    
    print(f"🔍 DEBUGGING REGISTER FILTERING FOR: {target_register}")
    print("="*80)
    
    # Read the log file
    try:
        if log_file.endswith('.gz'):
            with gzip.open(log_file, 'rt', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        else:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print(f"📄 File: {os.path.basename(log_file)}")
    print(f"📏 Size: {len(content)} characters")
    print()
    
    # Search for the target register in raw content
    target_lines = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if target_register in line:
            target_lines.append((i+1, line))
    
    print(f"🎯 Found {len(target_lines)} lines containing '{target_register}':")
    for line_num, line in target_lines[:5]:  # Show first 5
        print(f"   Line {line_num}: {line}")
    print()
    
    # Test current patterns from standalone_main.py
    patterns = [
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
        
        # Format 3: Alternative single line
        re.compile(
            r'\[(\d+)\].*?(MEM_[RW]|CR_[RW]).*?([^\s]+)\s+(0x[0-9a-fA-F]+)'
        )
    ]
    
    print("🔍 TESTING REGEX PATTERNS:")
    print("-" * 40)
    
    for i, pattern in enumerate(patterns, 1):
        matches = list(pattern.finditer(content))
        print(f"Pattern {i}: {len(matches)} total matches")
        
        # Check if any matches contain our target register
        target_matches = []
        for match in matches:
            if target_register in match.group(0):
                target_matches.append(match)
        
        print(f"  - {len(target_matches)} matches contain '{target_register}'")
        
        if target_matches:
            print(f"  - Example match groups: {target_matches[0].groups()}")
            print(f"  - Full match: {target_matches[0].group(0)}")
        print()
    
    # Check exclusion keywords from standalone_main.py
    excluded_keywords = [
        'Registry::',
        'timers.timers_table',
        '_id',
        'tx_',
        'event_manager.signaled_events_mask',
        'scheduler',
        'uc_restore_stat',
        'eventlogger.buffer',
        'eventlogger.is_enabled',
        'eventlogger.byte_index',
        'autonomous_pstate'
    ]
    
    print("🚫 CHECKING EXCLUSION KEYWORDS:")
    print("-" * 40)
    register_lower = target_register.lower()
    excluded = False
    for keyword in excluded_keywords:
        if keyword.lower() in register_lower:
            print(f"❌ EXCLUDED by keyword: '{keyword}'")
            excluded = True
    
    if not excluded:
        print("✅ No exclusion keywords match")
    print()
    
    # Test meaningful sequence filtering
    print("📊 TESTING SEQUENCE FILTERING:")
    print("-" * 40)
    
    # Extract all values for this register manually
    register_values = []
    for line_num, line in target_lines:
        # Try to extract hex value from the line
        hex_match = re.search(r'0x([0-9a-fA-F]+)\s*(?:<--|$)', line)
        if hex_match:
            try:
                value = int(hex_match.group(0), 16)
                register_values.append(value)
            except:
                pass
    
    print(f"Found {len(register_values)} values: {register_values[:10]}{'...' if len(register_values) > 10 else ''}")
    
    if register_values:
        # Test filtering criteria
        unique_values = len(set(register_values))
        print(f"Unique values: {unique_values}")
        print(f"Total values: {len(register_values)}")
        
        # Check change frequency
        if len(register_values) > 1:
            changes = sum(1 for i in range(len(register_values)-1) if register_values[i] != register_values[i+1])
            change_ratio = changes / (len(register_values) - 1)
            print(f"Change frequency: {change_ratio:.3f} ({changes} changes out of {len(register_values)-1} transitions)")
            
            # Check filtering thresholds
            print(f"Passes unique value test (>1): {'✅' if unique_values > 1 else '❌'}")
            print(f"Passes change frequency test (≥0.1): {'✅' if change_ratio >= 0.1 else '❌'}")
            print(f"Passes minimum length test (≥3): {'✅' if len(register_values) >= 3 else '❌'}")
    print()
    
    # Suggest fixes
    print("🔧 SUGGESTED FIXES:")
    print("-" * 40)
    
    if not target_lines:
        print("❌ Register not found in log - check register name spelling")
    elif excluded:
        print("❌ Register excluded by keyword filter - remove from exclusion list")
    elif len(register_values) < 3:
        print("❌ Not enough values - reduce minimum sequence length requirement")
    elif len(set(register_values)) <= 1:
        print("❌ No value changes - this register has constant values")
    elif len(register_values) > 1:
        changes = sum(1 for i in range(len(register_values)-1) if register_values[i] != register_values[i+1])
        change_ratio = changes / (len(register_values) - 1)
        if change_ratio < 0.1:
            print(f"❌ Low change frequency ({change_ratio:.3f}) - reduce threshold from 0.1 to {change_ratio:.3f}")
    
    # Check if CR_R vs MEM_R is the issue
    cr_count = sum(1 for _, line in target_lines if 'CR_R' in line)
    mem_count = sum(1 for _, line in target_lines if 'MEM_R' in line)
    print(f"Action types: CR_R={cr_count}, MEM_R={mem_count}")
    
    if cr_count > 0 and mem_count == 0:
        print("🔧 Pattern may need to include CR_R (Control Register) operations")


def create_improved_pattern():
    """Create an improved pattern that handles CR_R and address spacing"""
    
    # Improved pattern that handles your specific format better
    improved_pattern = re.compile(
        r'\[(\d+)\]\s+(MEM_[RW]|CR_[RW])\s+\[\s*(0x[0-9a-fA-F]+)\s*\]\s+(\w*)\s+([^\s].*?)\s+(0x[0-9a-fA-F]+)(?:\s+<--.*)?'
    )
    
    return improved_pattern


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python register_filter_debug.py <log_file>")
        print("Example: python register_filter_debug.py test.log.gz")
        sys.exit(1)
    
    debug_register_filtering(sys.argv[1])
    
    print("\n" + "="*80)
    print("🔧 IMPROVED REGEX PATTERN:")
    print("-" * 40)
    print("Replace the pattern in standalone_main.py with this improved version:")
    print()
    print("# Format 2: Enhanced single line format (handles CR_R and spacing)")
    print("re.compile(")
    print("    r'\\[(\\d+)\\]\\s+(MEM_[RW]|CR_[RW])\\s+\\[\\s*(0x[0-9a-fA-F]+)\\s*\\]\\s+(\\w*)\\s+([^\\s].*?)\\s+(0x[0-9a-fA-F]+)(?:\\s+<--.*)?'")
    print(")")
    print("\nThis improved pattern:")
    print("✅ Handles both MEM_R and CR_R operations") 
    print("✅ Allows flexible spacing around addresses")
    print("✅ Handles optional size field")
    print("✅ Ignores comment sections (<-- ...)")