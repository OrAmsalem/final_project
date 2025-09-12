"""
Reference model for ptracker hardware validation logs.
Builds and maintains a reference model based on passing test logs.
Used for anomaly detection and comparative analysis.
"""
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import logging
import re
import os
import glob
import gzip
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class PtrackerReferenceModel:
    def __init__(self, logger=None):
        """
        Initialize the Reference Model for ptracker logs

        Parameters:
        logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Storage for reference data
        self.reference_features = {}
        self.reference_models = {}
        self.register_models = {}
        self.scaler = StandardScaler()
        self.temporal_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        
        # Status flags
        self.is_trained = False
    
    def load_reference_logs(self, reference_log_dir):
        """
        Load and process reference logs from a directory of passing gz files
        
        Parameters:
        reference_log_dir (str): Directory containing passing log files
        
        Returns:
        dict: Processed reference data
        """
        self.logger.info(f"Loading reference logs from {reference_log_dir}")
        
        # Find all ptracker log files
        log_files = glob.glob(os.path.join(reference_log_dir, "*.log.gz"))
        
        if not log_files:
            self.logger.warning(f"No reference log files found in {reference_log_dir}")
            return {}
            
        self.logger.info(f"Found {len(log_files)} reference log files")
        
        # Process each reference log file 
        reference_data = []
        for log_file in log_files:
            try:
                # Process the log file to extract tokens
                tokens = self._tokenize_log_file(log_file)
                
                if tokens:
                    test_id = os.path.basename(log_file)
                    reference_data.append({
                        'test_id': test_id,
                        'tokens': tokens,
                        'status': 'pass'  # These are passing logs
                    })
            except Exception as e:
                self.logger.warning(f"Error processing reference log {log_file}: {e}")
        
        if not reference_data:
            self.logger.warning("No reference data could be extracted from log files")
            return {}
        
        self.logger.info(f"Processed {len(reference_data)} reference logs")
        return reference_data
    
    def _tokenize_log_file(self, log_file):
        """
        Parse a raw log file and extract tokens
        
        Parameters:
        log_file (str): Path to gzipped log file
        
        Returns:
        dict: Tokenized log data
        """
        # This is a placeholder - you would need to implement actual log parsing
        # based on your log tokenizer's capabilities
        self.logger.info(f"Tokenizing reference log: {log_file}")
        
        try:
            # Open gzipped file
            with gzip.open(log_file, 'rt', encoding='utf-8', errors='ignore') as f:
                # Read first 1000 lines for processing
                lines = []
                for i, line in enumerate(f):
                    if i >= 1000:  # Limit to first 1000 lines for performance
                        break
                    lines.append(line.strip())
                
            # Parse log lines into tokens
            # This is simplified - you would need actual parsing logic here
            registers = []
            fsm_states = []
            errors = []
            
            # Simple regex patterns for demonstration
            reg_pattern = r'\[([\d\.]+)\]\s+(MEM_[RW])\s+(\w+)\s+(\d+)\s+([^\s]+)\s+([0-9a-fA-F]+)'
            fsm_pattern = r'FSM\s+(\w+):\s+(\w+)\s*->\s*(\w+)'
            error_pattern = r'ERROR:\s+(.+)'
            
            for line in lines:
                # Check for register accesses
                reg_match = re.search(reg_pattern, line)
                if reg_match:
                    timestamp, action, address, size, description, value = reg_match.groups()
                    registers.append({
                        'timestamp': timestamp,
                        'action': action,
                        'address': address,
                        'size': size,
                        'description': description,
                        'value': value,
                        'raw': line
                    })
                    continue
                
                # Check for FSM transitions
                fsm_match = re.search(fsm_pattern, line)
                if fsm_match:
                    fsm_name, from_state, to_state = fsm_match.groups()
                    fsm_states.append({
                        'timestamp': '0',  # Timestamp might not be available
                        'match': [fsm_name, from_state, to_state],
                        'raw': line
                    })
                    continue
                
                # Check for errors
                error_match = re.search(error_pattern, line)
                if error_match:
                    error_msg = error_match.group(1)
                    errors.append({
                        'timestamp': '0',  # Timestamp might not be available
                        'match': [error_msg],
                        'raw': line
                    })
            
            # Create token dictionary
            tokens = {
                'registers': registers,
                'fsm_states': fsm_states,
                'errors': errors
            }
            
            return tokens
            
        except Exception as e:
            self.logger.error(f"Error parsing log file {log_file}: {e}")
            return None
    
    def build_reference_model(self, extracted_features):
        """
        Build reference models based on extracted features
        
        Parameters:
        extracted_features (dict): Dictionary of tests with extracted features
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Building reference model from extracted features")
            
            # Consolidate features
            self.reference_features = self._consolidate_reference_features(extracted_features)
            
            # Train anomaly detection models
            self._train_models(extracted_features)
            
            self.is_trained = True
            self.logger.info("Reference model built successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to build reference model: {e}")
            return False
    
    def _consolidate_reference_features(self, reference_tests):
        """
        Consolidate features from multiple reference tests into reference patterns
        
        Parameters:
        reference_tests (dict): Dictionary of reference tests with extracted features
        
        Returns:
        dict: Consolidated reference features
        """
        consolidated = {}
        
        # Process each feature type
        feature_types = ['register_values', 'register_sequences', 'fsm_transitions', 
                         'register_access_patterns', 'temporal_patterns', 'error_contexts']
        
        for feature_type in feature_types:
            feature_data = {}
            
            # Collect all feature values for this type
            for test_id, test_data in reference_tests.items():
                if 'features' in test_data and feature_type in test_data['features']:
                    feature_values = test_data['features'][feature_type]
                    
                    # Merge with existing data based on feature type
                    if feature_type == 'register_values':
                        # For register values, collect statistics per register
                        for register, stats in feature_values.items():
                            if register not in feature_data:
                                feature_data[register] = []
                            feature_data[register].append(stats)
                    
                    elif feature_type == 'fsm_transitions':
                        # For FSM transitions, collect by FSM name
                        for fsm_name, fsm_stats in feature_values.items():
                            if fsm_name not in feature_data:
                                feature_data[fsm_name] = []
                            feature_data[fsm_name].append(fsm_stats)
                    
                    elif feature_type in ['register_sequences', 'register_access_patterns', 
                                          'temporal_patterns', 'error_contexts']:
                        # For other features, collect all values
                        if 'values' not in feature_data:
                            feature_data['values'] = []
                        feature_data['values'].append(feature_values)
            
            # Calculate statistics for collected features
            consolidated_feature = {}
            
            if feature_type == 'register_values':
                # Calculate statistics for each register
                for register, stats_list in feature_data.items():
                    if not stats_list:
                        continue
                        
                    # Extract numeric values for each statistic
                    consolidated_stats = {}
                    for stat_name in stats_list[0].keys():
                        if stat_name not in ['importance']:  # Skip non-numeric
                            try:
                                values = [s[stat_name] for s in stats_list if stat_name in s]
                                if values:
                                    consolidated_stats[f"{stat_name}_mean"] = np.mean(values)
                                    consolidated_stats[f"{stat_name}_std"] = np.std(values)
                                    consolidated_stats[f"{stat_name}_min"] = np.min(values)
                                    consolidated_stats[f"{stat_name}_max"] = np.max(values)
                            except:
                                pass
                    
                    consolidated_feature[register] = consolidated_stats
            
            elif feature_type == 'fsm_transitions':
                # Consolidate FSM statistics
                for fsm_name, fsm_stats_list in feature_data.items():
                    if not fsm_stats_list:
                        continue
                        
                    # Collect all transitions seen in reference data
                    all_transitions = {}
                    for fsm_stats in fsm_stats_list:
                        if 'transition_counts' in fsm_stats:
                            for transition, count in fsm_stats['transition_counts'].items():
                                if transition not in all_transitions:
                                    all_transitions[transition] = []
                                all_transitions[transition].append(count)
                    
                    # Calculate statistics for transitions
                    transition_stats = {}
                    for transition, counts in all_transitions.items():
                        transition_stats[transition] = {
                            'mean': np.mean(counts),
                            'std': np.std(counts),
                            'min': np.min(counts),
                            'max': np.max(counts)
                        }
                    
                    consolidated_feature[fsm_name] = {
                        'transitions': transition_stats
                    }
            
            elif feature_type in ['register_sequences', 'register_access_patterns', 
                                  'temporal_patterns', 'error_contexts']:
                # For other features, just store all values
                consolidated_feature = feature_data
            
            consolidated[feature_type] = consolidated_feature
            
        return consolidated
    
    def _train_models(self, reference_tests):
        """
        Train machine learning models on reference data
        
        Parameters:
        reference_tests (dict): Dictionary of reference tests with extracted features
        
        Returns:
        bool: True if successful
        """
        # Extract temporal features for anomaly detection
        temporal_features = []
        
        for test_id, test_data in reference_tests.items():
            if 'features' in test_data and 'temporal_patterns' in test_data['features']:
                temp_data = test_data['features']['temporal_patterns']
                
                # Extract key features for anomaly detection
                if 'activity_distribution' in temp_data:
                    reads = temp_data['activity_distribution'].get('reads', [])
                    writes = temp_data['activity_distribution'].get('writes', [])
                    
                    # Create feature vector (use only first 20 values if available)
                    feature_vector = []
                    feature_vector.extend(reads[:20] if len(reads) >= 20 else reads + [0] * (20 - len(reads)))
                    feature_vector.extend(writes[:20] if len(writes) >= 20 else writes + [0] * (20 - len(writes)))
                    
                    # Add summary stats
                    feature_vector.append(temp_data.get('read_count', 0))
                    feature_vector.append(temp_data.get('write_count', 0))
                    feature_vector.append(temp_data.get('access_rate', 0))
                    
                    temporal_features.append(feature_vector)
        
        # Train temporal anomaly model if we have enough data
        if len(temporal_features) >= 10:
            # Scale features
            temporal_features = np.array(temporal_features)
            scaled_features = self.scaler.fit_transform(temporal_features)
            
            # Train model
            self.temporal_model.fit(scaled_features)
            self.logger.info(f"Trained temporal anomaly model on {len(temporal_features)} samples")
        
        # Extract register value features for anomaly detection
        register_stats = {}
        
        for test_id, test_data in reference_tests.items():
            if 'features' in test_data and 'register_values' in test_data['features']:
                reg_values = test_data['features']['register_values']
                
                # Process each register
                for register, stats in reg_values.items():
                    # Skip if not enough data
                    if 'count' not in stats or stats['count'] < 3:
                        continue
                        
                    # Save register statistics for later anomaly detection
                    if register not in register_stats:
                        register_stats[register] = []
                    
                    # Create feature vector from stats
                    feature_vector = [
                        stats.get('mean', 0),
                        stats.get('std', 0),
                        stats.get('min', 0),
                        stats.get('max', 0),
                        stats.get('range', 0),
                        stats.get('unique_values', 0),
                        stats.get('change_frequency', 0) if 'change_frequency' in stats else 0
                    ]
                    
                    register_stats[register].append(feature_vector)
        
        # Train register value anomaly models for registers with enough data
        for register, feature_vectors in register_stats.items():
            if len(feature_vectors) >= 5:  # Need at least 5 samples
                try:
                    # Convert to numpy array
                    feature_vectors = np.array(feature_vectors)
                    
                    # Skip if all values are the same
                    if np.all(feature_vectors == feature_vectors[0]):
                        continue
                        
                    # Scale features
                    scaler = StandardScaler().fit(feature_vectors)
                    scaled_features = scaler.transform(feature_vectors)
                    
                    # Train model
                    model = IsolationForest(
                        n_estimators=100,
                        contamination=0.05,
                        random_state=42
                    )
                    model.fit(scaled_features)
                    
                    # Store model
                    self.register_models[register] = {
                        'model': model,
                        'scaler': scaler
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to train model for register {register}: {e}")
        
        self.logger.info(f"Trained anomaly detection models for {len(self.register_models)} registers")
        return True
    
    def detect_anomalies(self, test_data):
        """
        Detect anomalies in test data based on reference models
        
        Parameters:
        test_data (dict): Test data with features
        
        Returns:
        dict: Detected anomalies
        """
        if not self.is_trained:
            self.logger.warning("Reference model is not trained yet")
            return {}
            
        anomalies = {}
        
        # Skip if no features
        if 'features' not in test_data:
            return anomalies
            
        # Detect temporal anomalies
        if 'temporal_patterns' in test_data['features']:
            temp_data = test_data['features']['temporal_patterns']
            
            if 'activity_distribution' in temp_data:
                try:
                    # Prepare feature vector
                    reads = temp_data['activity_distribution'].get('reads', [])
                    writes = temp_data['activity_distribution'].get('writes', [])
                    
                    feature_vector = []
                    feature_vector.extend(reads[:20] if len(reads) >= 20 else reads + [0] * (20 - len(reads)))
                    feature_vector.extend(writes[:20] if len(writes) >= 20 else writes + [0] * (20 - len(writes)))
                    
                    feature_vector.append(temp_data.get('read_count', 0))
                    feature_vector.append(temp_data.get('write_count', 0))
                    feature_vector.append(temp_data.get('access_rate', 0))
                    
                    # Scale features
                    scaled_vector = self.scaler.transform([feature_vector])
                    
                    # Predict anomaly
                    anomaly_score = self.temporal_model.score_samples(scaled_vector)[0]
                    is_anomaly = self.temporal_model.predict(scaled_vector)[0] == -1
                    
                    if is_anomaly:
                        anomalies['temporal'] = {
                            'score': float(anomaly_score),
                            'is_anomaly': True
                        }
                except Exception as e:
                    self.logger.warning(f"Error detecting temporal anomalies: {e}")
        
        # Detect register value anomalies
        if 'register_values' in test_data['features'] and self.register_models:
            reg_values = test_data['features']['register_values']
            register_anomalies = {}
            
            for register, stats in reg_values.items():
                # Skip if no model for this register
                if register not in self.register_models:
                    continue
                    
                try:
                    # Create feature vector
                    feature_vector = [
                        stats.get('mean', 0),
                        stats.get('std', 0),
                        stats.get('min', 0),
                        stats.get('max', 0),
                        stats.get('range', 0),
                        stats.get('unique_values', 0),
                        stats.get('change_frequency', 0) if 'change_frequency' in stats else 0
                    ]
                    
                    # Get model and scaler
                    model = self.register_models[register]['model']
                    scaler = self.register_models[register]['scaler']
                    
                    # Scale features
                    scaled_vector = scaler.transform([feature_vector])
                    
                    # Predict anomaly
                    anomaly_score = model.score_samples(scaled_vector)[0]
                    is_anomaly = model.predict(scaled_vector)[0] == -1
                    
                    if is_anomaly:
                        register_anomalies[register] = {
                            'score': float(anomaly_score),
                            'is_anomaly': True
                        }
                except Exception as e:
                    self.logger.warning(f"Error detecting anomaly for register {register}: {e}")
            
            if register_anomalies:
                anomalies['register_values'] = register_anomalies
                
        # Compare FSM transitions with reference data
        if ('fsm_transitions' in test_data['features'] and 
            'fsm_transitions' in self.reference_features):
            
            fsm_transitions = test_data['features']['fsm_transitions']
            ref_fsm = self.reference_features['fsm_transitions']
            fsm_anomalies = {}
            
            for fsm_name, fsm_stats in fsm_transitions.items():
                # Skip if no reference data for this FSM
                if fsm_name not in ref_fsm:
                    continue
                    
                ref_transitions = ref_fsm[fsm_name].get('transitions', {})
                
                # Check for unusual transitions
                unusual_transitions = {}
                if 'transition_counts' in fsm_stats:
                    for transition, count in fsm_stats['transition_counts'].items():
                        if transition not in ref_transitions:
                            # This transition was never seen in reference data
                            unusual_transitions[transition] = {
                                'count': count,
                                'reason': 'transition_not_in_reference'
                            }
                        else:
                            # Check if count is outside normal range
                            ref_stats = ref_transitions[transition]
                            mean = ref_stats.get('mean', 0)
                            std = ref_stats.get('std', 1)
                            
                            # Consider unusual if more than 3 standard deviations from mean
                            if abs(count - mean) > 3 * std:
                                unusual_transitions[transition] = {
                                    'count': count,
                                    'reference_mean': mean,
                                    'reference_std': std,
                                    'reason': 'count_outside_normal_range'
                                }
                
                if unusual_transitions:
                    fsm_anomalies[fsm_name] = {
                        'unusual_transitions': unusual_transitions
                    }
            
            if fsm_anomalies:
                anomalies['fsm_transitions'] = fsm_anomalies
        
        return anomalies
    
    def save_model(self, output_dir):
        """
        Save reference model to disk
        
        Parameters:
        output_dir (str): Directory to save model
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save reference features
            joblib.dump(self.reference_features, os.path.join(output_dir, 'reference_features.pkl'))
            
            # Save temporal model
            joblib.dump(self.temporal_model, os.path.join(output_dir, 'temporal_model.pkl'))
            
            # Save scaler
            joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
            
            # Save register models
            os.makedirs(os.path.join(output_dir, 'register_models'), exist_ok=True)
            for register, model_data in self.register_models.items():
                # Create a safe filename
                safe_name = re.sub(r'[^\w\-]', '_', register)
                joblib.dump(model_data, os.path.join(output_dir, 'register_models', f'{safe_name}.pkl'))
            
            self.logger.info(f"Saved reference model to {output_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save reference model: {e}")
            return False
    
    def load_model(self, model_dir):
        """
        Load reference model from disk
        
        Parameters:
        model_dir (str): Directory containing saved model
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Check model directory exists
            if not os.path.exists(model_dir):
                self.logger.error(f"Model directory does not exist: {model_dir}")
                return False
            
            # Load reference features
            features_path = os.path.join(model_dir, 'reference_features.pkl')
            if os.path.exists(features_path):
                self.reference_features = joblib.load(features_path)
            
            # Load temporal model
            model_path = os.path.join(model_dir, 'temporal_model.pkl')
            if os.path.exists(model_path):
                self.temporal_model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load register models
            register_model_dir = os.path.join(model_dir, 'register_models')
            if os.path.exists(register_model_dir):
                for model_file in glob.glob(os.path.join(register_model_dir, '*.pkl')):
                    try:
                        model_data = joblib.load(model_file)
                        # Extract register name from filename
                        register = os.path.splitext(os.path.basename(model_file))[0]
                        # Convert underscores back to original format (approximate)
                        register = register.replace('_', '.')
                        self.register_models[register] = model_data
                    except Exception as e:
                        self.logger.warning(f"Failed to load register model {model_file}: {e}")
            
            self.is_trained = True
            self.logger.info(f"Loaded reference model from {model_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load reference model: {e}")
            return False