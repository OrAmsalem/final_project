def analyze_failing_tests(self, failing_tests):
        """
        ENHANCED failing test analysis - includes fail-only register analysis and divergence detection
        NOW FILTERS OUT IDENTICAL SEQUENCES (FALSE POSITIVES)
        
        Parameters:
        failing_tests (dict): Dictionary of failing test data
        
        Returns:
        dict: Enhanced analysis results
        """
        self.logger.info("⚡ Analyzing failing tests with divergence detection")
        
        results = {}
        total_comparisons = 0
        total_mismatches = 0
        filtered_identical = 0  # NEW: Track filtered identical sequences
        
        # NEW: Step 1 - Identify fail-only registers
        fail_only_sequences = defaultdict(list)  # register -> list of sequences from failing tests
        all_failing_registers = set()
        
        for test_id, test_data in failing_tests.items():
            sequences = test_data.get('register_sequences', {})
            if not sequences and 'features' in test_data:
                continue
            
            for register in sequences.keys():
                all_failing_registers.add(register)
                
                # Check if this register is fail-only (not in reference model)
                if register not in self.reference_registers:
                    # Extract values for divergence analysis
                    values = [entry['value'] for entry in sequences[register] if isinstance(entry, dict)]
                    if values and len(values) >= self.min_common_prefix_length:
                        fail_only_sequences[register].append({
                            'test_id': test_id,
                            'values': values,
                            'timestamps': [entry['timestamp'] for entry in sequences[register] if isinstance(entry, dict)]
                        })
        
        self.fail_only_registers = all_failing_registers - self.reference_registers
        
        self.logger.info(f"🔍 Found {len(self.fail_only_registers)} fail-only registers")
        
        # NEW: Step 2 - Analyze divergence points in fail-only registers
        divergence_analysis = self._analyze_divergence_points(fail_only_sequences)
        
        # Step 3 - Existing reference model comparison WITH IDENTICAL SEQUENCE FILTERING
        for test_id, test_data in failing_tests.items():
            sequences = test_data.get('register_sequences', {})
            if not sequences and 'features' in test_data:
                continue
            
            test_mismatches = {}
            
            for register, sequence in sequences.items():
                total_comparisons += 1
                
                if register in self.reference_sequences:
                    # Extract actual values
                    actual_values = [entry['value'] for entry in sequence if isinstance(entry, dict)]
                    expected_values = self.reference_sequences[register]['expected_sequence']
                    
                    # NEW: Filter out identical sequences (false positives)
                    if actual_values == expected_values:
                        filtered_identical += 1
                        self.logger.debug(f"Filtered identical sequence for {register} in {test_id}")
                        continue  # Skip identical sequences
                    
                    # Simple exact comparison (fast) - only for truly different sequences
                    if actual_values != expected_values:
                        test_mismatches[register] = {
                            'expected_sequence': expected_values,
                            'actual_sequence': actual_values,
                            'mismatch_type': self._categorize_mismatch(expected_values, actual_values),
                            'analysis_type': 'reference_model_mismatch'
                        }
                        total_mismatches += 1
            
            # NEW: Step 4 - Add divergence point findings to test results
            test_divergences = {}
            for register, divergence_info in divergence_analysis.items():
                # Check if this test has this register
                if register in sequences:
                    actual_values = [entry['value'] for entry in sequences[register] if isinstance(entry, dict)]
                    
                    # Check if this test matches the divergence pattern
                    if self._test_matches_divergence_pattern(actual_values, divergence_info):
                        test_divergences[register] = {
                            'divergence_point': divergence_info['divergence_point'],
                            'common_prefix': divergence_info['common_prefix'],
                            'pre_divergence_values': actual_values[:divergence_info['divergence_point']],
                            'post_divergence_values': actual_values[divergence_info['divergence_point']:],
                            'analysis_type': 'fail_only_divergence',
                            'confidence_score': divergence_info['confidence_score']
                        }
            
            # Combine results
            combined_results = {}
            if test_mismatches:
                combined_results['reference_model_mismatches'] = test_mismatches
            if test_divergences:
                combined_results['fail_only_divergences'] = test_divergences
                
            if combined_results:
                results[test_id] = combined_results
        
        # Store divergence patterns for reporting
        self.divergence_patterns = divergence_analysis
        
        self.logger.info(f"⚡ Analysis complete: {total_mismatches}/{total_comparisons} reference mismatches, "
                        f"{len(divergence_analysis)} divergence patterns")
        self.logger.info(f"🚫 Filtered {filtered_identical} identical sequences (false positives)")
        return results