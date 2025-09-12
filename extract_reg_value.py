#!/usr/bin/env python3
"""
Enhanced register value feature extraction with RELAXED filtering for better diagnostic coverage.
Now keeps more registers that might have diagnostic value even with minimal changes.
"""

import logging
import numpy as np
from collections import defaultdict, Counter


def should_skip_register(register_name):
    """
    Check if register should be skipped - MUCH MORE SELECTIVE now
    Only skips truly useless registers, keeps all diagnostic ones
    
    Parameters:
    register_name (str): Register name/path
    
    Returns:
    bool: True if register should be skipped
    """
    # CRITICAL REGISTERS - NEVER SKIP THESE
    NEVER_SKIP = [
        'ring_cstate.ring_cstate_control.target_ring_cstate',
        'ring_cstate.ring_cstate_control.curr_ring_cstate', 
        'ring_cstate.ring_cstate_control.ring_cst_fsm',
        'hvm_compressed_data.state',
        'Registry::get_registry'
    ]
    
    # Check if this is a critical register
    for critical in NEVER_SKIP:
        if critical in register_name:
            return False  # NEVER skip critical registers
    
    # IMPORTANT PATTERNS - Also never skip
    register_lower = register_name.lower()
    IMPORTANT_PATTERNS = [
        'state',       # State registers are important
        'control',     # Control registers
        'status',      # Status registers  
        'error',       # Error registers
        'interrupt',   # Interrupt registers
        'power',       # Power management
        'voltage',     # Voltage registers
        'frequency',   # Frequency registers
        'clock',       # Clock registers
        'fault',       # Fault registers
        'exception',   # Exception registers
        'fsm',         # Finite state machines
        'cstate',      # C-state registers
    ]
    
    # Don't skip registers with important patterns
    for pattern in IMPORTANT_PATTERNS:
        if pattern in register_lower:
            return False
    
    # VERY RESTRICTIVE SKIP PATTERNS - Only skip truly useless registers
    SKIP_PATTERNS = [
        'zero_value',
        'placeholder', 
        'unused_register',
        'reserved_bits',
        # Only skip very specific patterns that we know provide no diagnostic value
        'test_pattern',
        'dummy_register',
    ]
    
    # Check for exact pattern matches - much more restrictive now
    for pattern in SKIP_PATTERNS:
        if pattern in register_lower:
            return True
    
    return False  # Default: keep the register


def has_meaningful_changes(values, min_change_ratio=0.01):
    """
    Check if a value sequence has meaningful changes
    VERY RELAXED thresholds to keep more diagnostic registers
    
    Parameters:
    values (list): List of register values
    min_change_ratio (float): Minimum ratio of values that must change (VERY LOW: 1%)
    
    Returns:
    bool: True if sequence has meaningful changes
    """
    if len(values) <= 1:
        return False
    
    # Only filter out completely constant sequences
    unique_values = set(values)
    if len(unique_values) <= 1:
        return False  # Still filter out [1,1,1,1,1...] constant arrays
    
    # For sequences with minimal variation, be very lenient
    if len(unique_values) <= 2 and len(values) > 100:  # Only for very long sequences
        unique_ratio = len(unique_values) / len(values)
        if unique_ratio < 0.01:  # Less than 1% unique values
            return False
    
    # Calculate actual change frequency - VERY RELAXED
    changes = sum(1 for i in range(len(values)-1) 
                 if values[i] != values[i+1])
    
    change_ratio = changes / (len(values) - 1)
    return change_ratio >= min_change_ratio  # Only need 1% changes


def has_sufficient_variation(values, min_unique_ratio=0.01):
    """
    Check if sequence has sufficient value variation
    VERY RELAXED to keep diagnostic registers
    
    Parameters:
    values (list): List of register values
    min_unique_ratio (float): Minimum ratio of unique values (VERY LOW: 1%)
    
    Returns:
    bool: True if sufficient variation exists
    """
    if len(values) <= 1:
        return False
    
    unique_count = len(set(values))
    
    # Still filter out completely constant sequences
    if unique_count == 1:
        return False
    
    # Very lenient for sequences with few unique values
    if unique_count >= 2:  # Any sequence with 2 or more unique values is kept
        unique_ratio = unique_count / len(values)
        return unique_ratio >= min_unique_ratio  # Only need 1%
    
    return True  # Default: keep it


def extract_register_value_features(self, test_records):
    """
    Extract features related to register values and their distributions
    WITH RELAXED FILTERING to capture more diagnostic registers
    
    Parameters:
    test_records (list): List of test records with tokens

    Returns:
    dict: Dictionary mapping test IDs to register value features
    """
    self.logger.debug("Extracting register value features with RELAXED filtering for better coverage")

    features = {}
    
    # Track filtering statistics for debugging
    total_registers = 0
    kept_registers = 0

    # Identify important registers (using AI tokenizer results if available)
    important_registers = self.identify_important_registers(test_records)

    # Process each test
    for record in test_records:
        test_id = record['test_id']
        tokens = record['tokens']

        # Skip if no register tokens
        if 'registers' not in tokens or not tokens['registers']:
            features[test_id] = {}
            continue

        # Group register values by register path WITH RELAXED FILTERING
        register_values = self.group_register_values_by_path_filtered(tokens['registers'])
        
        # Update filtering statistics
        unique_registers_before = len(set(token['description'] for token in tokens['registers']))
        total_registers += unique_registers_before
        kept_registers += len(register_values)
        
        # Calculate statistics for each meaningful register
        reg_features = self.calculate_register_statistics(register_values, important_registers)
        features[test_id] = reg_features

    # Log filtering results
    filtered_count = total_registers - kept_registers
    if total_registers > 0:
        filter_percentage = (filtered_count / total_registers) * 100
        self.logger.info(f"RELAXED filtering: removed {filtered_count}/{total_registers} "
                       f"({filter_percentage:.1f}%) truly constant registers")
        self.logger.info(f"KEPT {kept_registers} registers with potential diagnostic value")
    else:
        self.logger.warning("No registers found in any test")

    return features


def group_register_values_by_path_filtered(self, register_tokens):
    """
    Group register values by register path WITH RELAXED FILTERING
    Now includes many more registers that might have diagnostic value
    
    Parameters:
    register_tokens (list): List of register tokens
    
    Returns:
    dict: Dictionary mapping register paths to values (relaxed filtering)
    """
    # First pass: collect all values per register
    temp_register_values = defaultdict(list)
    
    for token in register_tokens:
        register = token['description']
        if 'value' in token and token['value']:
            try:
                # Convert hex value to integer
                value = int(token['value'], 16)
                temp_register_values[register].append({
                    'timestamp': int(token['timestamp']),
                    'value': value,
                    'raw': token['value']
                })
            except (ValueError, TypeError):
                # Skip if value can't be converted
                pass
    
    # Second pass: apply VERY RELAXED filtering
    filtered_register_values = {}
    
    for register, values in temp_register_values.items():
        # Extract just the values for analysis
        value_list = [v['value'] for v in values]
        
        # Apply MUCH MORE RELAXED filtering criteria
        if (not should_skip_register(register) and
            has_meaningful_changes(value_list) and
            has_sufficient_variation(value_list) and
            len(values) >= 2):  # Still need minimum sequence length
            
            filtered_register_values[register] = values
        else:
            # ENHANCED logging to show WHY registers are filtered
            filter_reason = "unknown"
            if should_skip_register(register):
                filter_reason = "explicit_skip_pattern"
            elif not has_meaningful_changes(value_list):
                filter_reason = "completely_constant"
            elif not has_sufficient_variation(value_list):
                filter_reason = f"insufficient_variation_({len(set(value_list))}_unique_of_{len(value_list)})"
            elif len(values) < 2:
                filter_reason = "insufficient_data"
            
            # Log differently based on register importance
            register_lower = register.lower()
            important_keywords = ['state', 'control', 'status', 'error', 'interrupt', 
                                'power', 'voltage', 'frequency', 'clock', 'fault']
            
            if any(keyword in register_lower for keyword in important_keywords):
                # Important register was filtered - this might be a problem
                self.logger.warning(f"IMPORTANT register filtered: {register}: "
                                  f"values={len(values)}, "
                                  f"unique={len(set(value_list))}, "
                                  f"reason={filter_reason}")
            else:
                # Regular register filtered - debug level
                self.logger.debug(f"Filtered register {register}: "
                                f"values={len(values)}, "
                                f"unique={len(set(value_list))}, "
                                f"reason={filter_reason}")
    
    return filtered_register_values


def calculate_register_statistics(self, register_values, important_registers):
    """
    Calculate statistics for register values (already filtered with relaxed criteria)
    
    Parameters:
    register_values (dict): Dictionary mapping register paths to values
    important_registers (dict): Dictionary mapping register paths to importance scores
    
    Returns:
    dict: Dictionary of register statistics
    """
    reg_features = {}
    
    for register, values in register_values.items():
        # Extract values only
        value_list = [v['value'] for v in values]

        # Calculate basic statistics
        stats = self.calculate_basic_statistics(value_list)

        # Calculate additional statistics for bit patterns
        bit_stats = self.calculate_bit_statistics(value_list)
        stats.update(bit_stats)

        # Calculate sequence-based statistics
        if len(value_list) > 1:
            seq_stats = self.calculate_sequence_statistics(value_list)
            stats.update(seq_stats)

        # Add importance score if available from AI tokenizer
        if register in important_registers:
            stats['importance'] = important_registers[register]
        else:
            stats['importance'] = 0.0

        # Add diagnostic flags
        stats['has_meaningful_changes'] = True  # Already filtered for this
        stats['unique_value_count'] = len(set(value_list))
        stats['change_frequency'] = self.calculate_change_frequency(value_list)
        stats['is_diagnostic_register'] = self._is_diagnostic_register(register)

        # Add to register features
        reg_features[register] = stats
    
    return reg_features


def _is_diagnostic_register(self, register_name):
    """
    Determine if a register is likely to have diagnostic value
    """
    register_lower = register_name.lower()
    diagnostic_patterns = [
        'state', 'control', 'status', 'error', 'interrupt', 'power', 
        'voltage', 'frequency', 'clock', 'fault', 'exception', 'fsm', 
        'cstate', 'debug', 'trace', 'performance'
    ]
    
    return any(pattern in register_lower for pattern in diagnostic_patterns)


def calculate_change_frequency(self, value_list):
    """
    Calculate how frequently the register value changes
    
    Parameters:
    value_list (list): List of register values
    
    Returns:
    float: Frequency of changes (0.0 to 1.0)
    """
    if len(value_list) <= 1:
        return 0.0
    
    changes = sum(1 for i in range(len(value_list)-1) 
                 if value_list[i] != value_list[i+1])
    
    return changes / (len(value_list) - 1)


def calculate_basic_statistics(self, value_list):
    """
    Calculate basic statistics for a list of values
    
    Parameters:
    value_list (list): List of numeric values
    
    Returns:
    dict: Basic statistics
    """
    if not value_list:
        return {}
    
    return {
        'count': len(value_list),
        'min': min(value_list),
        'max': max(value_list),
        'mean': np.mean(value_list),
        'median': np.median(value_list),
        'std': np.std(value_list),
        'unique_values': len(set(value_list)),
        'range': max(value_list) - min(value_list)
    }


def calculate_sequence_statistics(self, value_list):
    """
    Calculate sequence-based statistics for consecutive values
    
    Parameters:
    value_list (list): List of numeric values
    
    Returns:
    dict: Sequence-based statistics
    """
    if len(value_list) <= 1:
        return {}
        
    # Calculate differences between consecutive values
    diffs = np.diff(value_list)
    stats = {
        'mean_diff': np.mean(diffs),
        'max_diff': np.max(np.abs(diffs)),
        'diff_std': np.std(diffs)
    }

    # Count value transitions
    transitions = Counter(zip(value_list[:-1], value_list[1:]))
    stats['unique_transitions'] = len(transitions)

    # Calculate how often the register changes value
    stats['change_frequency'] = np.sum(np.abs(diffs) > 0) / (len(value_list) - 1)
    
    return stats


def calculate_bit_statistics(self, value_list):
    """
    Calculate bit-level statistics for register values
    
    Parameters:
    value_list (list): List of numeric values
    
    Returns:
    dict: Bit-level statistics
    """
    if not value_list:
        return {}
    
    bit_stats = {}
    
    # Find the maximum number of bits needed
    max_value = max(value_list)
    if max_value > 0:
        bit_width = max_value.bit_length()
        
        # Analyze each bit position
        bit_changes = []
        for bit_pos in range(bit_width):
            bit_values = [(val >> bit_pos) & 1 for val in value_list]
            bit_changes.append(len(set(bit_values)) > 1)  # True if bit changes
        
        bit_stats['active_bits'] = sum(bit_changes)
        bit_stats['bit_width'] = bit_width
        bit_stats['bit_change_ratio'] = sum(bit_changes) / bit_width if bit_width > 0 else 0
    
    return bit_stats


# Backward compatibility - keep original function names
def group_register_values_by_path(self, register_tokens):
    """
    Original function name - now calls the relaxed filtered version
    """
    return self.group_register_values_by_path_filtered(register_tokens)


if __name__ == "__main__":
    print("Enhanced register value extraction with RELAXED filtering")
    print("Key improvements:")
    print("- Keeps more diagnostic registers (state, control, status, etc.)")
    print("- Only filters truly constant sequences")
    print("- Warns when important registers are filtered") 
    print("- Much more selective about what gets removed")
    print("- Should capture registers critical for debugging")