#!/usr/bin/env python3
"""
Test script to debug log format and pattern matching
"""
import gzip
import re
import sys

def test_log_format(log_file):
    """Test what format the log file actually has"""
    
    print(f"Testing log file: {log_file}")
    
    # Read the file
    try:
        if log_file.endswith('.gz'):
            with gzip.open(log_file, 'rt', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        else:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"File size: {len(content)} characters")
    
    # Show first 2000 characters
    print("\nFirst 2000 characters:")
    print("=" * 50)
    print(content[:2000])
    print("=" * 50)
    
    # Test multiline pattern
    multiline_pattern = re.compile(
        r'\[\s*(\d+)\]\s+Timestamp:\s+(\d+)\s+'
        r'Action:\s+(\w+)\s+'
        r'Address:\s+(0x[0-9a-fA-F]+)\s+'
        r'Size:\s+(\w+)\s+'
        r'Register:\s+(.+?)\s+'
        r'Value:\s+(0x[0-9a-fA-F]+)',
        re.MULTILINE | re.DOTALL
    )
    
    # Test single line pattern (updated)
    singleline_pattern = re.compile(
        r'\[(\d+)\]\s+(MEM_[RW]|CR_[RW])\s+\[(0x[0-9a-fA-F]+)\]\s+(\w+)\s+([^\s].*?)\s+(0x[0-9a-fA-F]+)(?:\s+<--\s+(.+))?'
    )
    
    multiline_matches = list(multiline_pattern.finditer(content))
    singleline_matches = list(singleline_pattern.finditer(content))
    
    print(f"\nPattern matching results:")
    print(f"Multiline pattern matches: {len(multiline_matches)}")
    print(f"Single line pattern matches: {len(singleline_matches)}")
    
    if multiline_matches:
        print(f"\nFirst multiline match:")
        match = multiline_matches[0]
        print(f"  Full match: {match.group(0)}")
        print(f"  Groups: {match.groups()}")
    
    if singleline_matches:
        print(f"\nFirst single line match:")
        match = singleline_matches[0]
        print(f"  Full match: {match.group(0)}")
        print(f"  Groups: {match.groups()}")
    
    # Look for any lines with brackets and numbers
    lines = content.split('\n')
    bracket_lines = [line for line in lines if '[' in line and ']' in line and any(c.isdigit() for c in line)]
    
    print(f"\nFound {len(bracket_lines)} lines with brackets and numbers")
    if bracket_lines:
        print("First 10 bracket lines:")
        for i, line in enumerate(bracket_lines[:10]):
            print(f"  {i+1}: {line}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_log_format.py <log_file_path>")
        sys.exit(1)
    
    test_log_format(sys.argv[1])