def _is_meaningful_sequence(self, register, sequence):
        """
        Determine if a register sequence is meaningful for analysis
        This filtering eliminates constant-value sequences that waste processing time
        NOW ALSO FILTERS OUT FALSE POSITIVE "IDENTICAL SEQUENCE" PATTERNS
        
        Parameters:
        register (str): Register name
        sequence (list): List of register access entries
        
        Returns:
        bool: True if sequence is meaningful, False if should be filtered out
        """
        
        # Must have minimum length
        if len(sequence) < 3:
            return False
        
        # Extract values for analysis
        values = [entry['value'] for entry in sequence]
        
        # Filter 1: Must have more than one unique value (not constant)
        unique_values = set(values)
        if len(unique_values) <= 1:
            return False
        
        # NEW: Filter 2: Check for "identical transition" patterns that are false positives
        # These are patterns where expected and actual are the same but flagged as errors
        if self._is_likely_false_positive_pattern(register, values):
            return False
        
        # Filter 3: Must have sufficient variation (not just 2 values in long sequence)
        if len(unique_values) == 2 and len(values) > 20:
            unique_ratio = len(unique_values) / len(values)
            if unique_ratio < 0.1:  # Less than 10% unique values
                return False
        
        # Filter 4: Must have meaningful change frequency
        changes = sum(1 for i in range(len(values)-1) if values[i] != values[i+1])
        change_ratio = changes / (len(values) - 1)
        if change_ratio < 0.05:  # Less than 5% of values change
            return False
        
        # Filter 5: Skip registers with known constant patterns
        register_lower = register.lower()
        constant_patterns = [
            'zero_value', 'constant_state', 'static_config', 
            'reserved_bits', 'unused_register', 'placeholder'
        ]
        
        if any(pattern in register_lower for pattern in constant_patterns):
            return False
        
        return True
    
    def _is_likely_false_positive_pattern(self, register, values):
        """
        NEW: Check if this is likely a false positive pattern
        
        Common false positive patterns:
        - Simple transitions like [0x0, 0x1] that complete correctly
        - State machine patterns that work but have extra accesses
        - Registers that complete their intended transitions
        
        Parameters:
        register (str): Register name  
        values (list): Sequence values
        
        Returns:
        bool: True if this is likely a false positive (should be filtered)
        """
        # Check for known false positive register patterns
        register_lower = register.lower()
        for pattern in self.false_positive_patterns:
            if pattern.lower() in register_lower:
                self.logger.debug(f"Filtered known false positive pattern: {register}")
                return True
        
        # Simple transition patterns that are often false positives
        if len(values) <= 4:  # Short sequences
            # Check for simple 0->1 or 1->0 transitions (common false positives)
            if len(set(values)) == 2:
                unique_vals = list(set(values))
                
                # Common state machine transitions that work correctly
                if (0 in unique_vals and 1 in unique_vals) or \
                   (0x0 in unique_vals and 0x1 in unique_vals):
                    # These are often correct state transitions flagged as errors
                    self.logger.debug(f"Filtered simple state transition: {register} ({unique_vals})")
                    return True
        
        # Check register names that often have false positive patterns
        false_positive_keywords = [
            'enabled',      # sst_manager.hwp_enabled - often false positive
            'configs_done', # sst_manager.configs_done - often false positive  
            'initialized',  # initialization flags - often false positive
            'target',       # target values - often false positive
            'last_request', # request tracking - often false positive
        ]
        
        for pattern in false_positive_keywords:
            if pattern in register_lower:
                # For these register types, be extra strict about variation
                if len(set(values)) <= 2:  # Very simple transitions
                    self.logger.debug(f"Filtered false positive keyword pattern: {register} ({pattern})")
                    return True
        
        return False