"""
Anomaly detection functionality for AI Debug Assist.
Detects deviations from expected behaviors in failing tests.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import joblib
import time
from sklearn.preprocessing import StandardScaler
import warnings


class AnomalyDetector:
    def __init__(self, behavior_models=None, logger=None):
        """
        Initialize the Anomaly Detector

        Parameters:
        behavior_models (dict): Dictionary of behavior models
        logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.behavior_models = behavior_models or {}

        # Initialize thresholds
        self.value_threshold = 3.0  # Standard deviations from mean
        self.anomaly_score_threshold = -0.5  # Isolation Forest anomaly score threshold

        # Initialize results storage
        self.anomalies = {}

    def detect_anomalies(self, tests, behavior_models=None):
        """
        Detect anomalies in tests based on learned behaviors

        Parameters:
        tests (dict): Dictionary of tests with extracted features
        behavior_models (dict): Dictionary of behavior models (optional)

        Returns:
        dict: Dictionary of detected anomalies
        """
        start_time = time.time()
        self.logger.info("Detecting anomalies in tests")

        # Use provided behavior models if specified
        if behavior_models:
            self.behavior_models = behavior_models

        # Check if behavior models exist
        if not self.behavior_models:
            self.logger.error("No behavior models available for anomaly detection")
            return {}

        # Initialize anomalies dictionary
        self.anomalies = {test_id: {'anomalies': []} for test_id in tests.keys()}

        # Detect anomalies by feature type
        if 'register_values' in self.behavior_models:
            self._detect_register_value_anomalies(tests)

        if 'register_sequences' in self.behavior_models:
            self._detect_sequence_anomalies(tests)

        if 'fsm_transitions' in self.behavior_models:
            self._detect_fsm_anomalies(tests)

        if 'temporal_patterns' in self.behavior_models:
            self._detect_temporal_anomalies(tests)

        if 'register_access_patterns' in self.behavior_models:
            self._detect_access_pattern_anomalies(tests)

        # Calculate anomaly statistics
        for test_id, anomaly_data in self.anomalies.items():
            anomaly_count = len(anomaly_data['anomalies'])

            # Add summary
            anomaly_data['summary'] = {
                'total_anomalies': anomaly_count,
                'categories': self._count_anomaly_categories(anomaly_data['anomalies']),
                'severity_counts': self._count_anomaly_severities(anomaly_data['anomalies'])
            }

            # Sort anomalies by severity (descending)
            anomaly_data['anomalies'].sort(key=lambda x: x['severity'], reverse=True)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Anomaly detection complete in {elapsed_time:.2f} seconds")

        return self.anomalies

    def _detect_register_value_anomalies(self, tests):
        """
        Detect anomalies in register values

        Parameters:
        tests (dict): Dictionary of tests with features
        """
        self.logger.debug("Detecting register value anomalies")

        # Get register value models
        register_models = self.behavior_models.get('register_values', {})

        if not register_models:
            return

        # Process each test
        for test_id, test_data in tests.items():
            if 'features' not in test_data or 'register_values' not in test_data['features']:
                continue

            reg_features = test_data['features']['register_values']

            # Check each register
            for register, stats in reg_features.items():
                # Skip if no model for this register
                if register not in register_models:
                    continue

                # Get model data
                model_data = register_models[register]
                expected = model_data['expected_values']
                model = model_data.get('model')
                scaler = model_data.get('scaler')

                # Skip if missing data
                if not expected or not model or not scaler:
                    continue

                # Check for value anomalies using statistical methods
                anomalies = []

                # Mean value check
                if 'mean' in stats and 'mean' in expected:
                    mean_diff = abs(stats['mean'] - expected['mean'])
                    mean_std = expected['std'] if expected['std'] > 0 else 1.0
                    mean_z_score = mean_diff / mean_std

                    if mean_z_score > self.value_threshold:
                        anomalies.append({
                            'type': 'mean_value',
                            'actual': stats['mean'],
                            'expected': expected['mean'],
                            'z_score': mean_z_score,
                            'severity': min(1.0, mean_z_score / (2 * self.value_threshold))
                        })

                # Range check
                if 'min' in stats and 'max' in stats and 'range' in expected:
                    exp_min, exp_max = expected['range']

                    if stats['min'] < exp_min:
                        anomalies.append({
                            'type': 'min_value',
                            'actual': stats['min'],
                            'expected': exp_min,
                            'severity': min(1.0, abs(stats['min'] - exp_min) / (
                                        exp_max - exp_min) if exp_max > exp_min else 1.0)
                        })

                    if stats['max'] > exp_max:
                        anomalies.append({
                            'type': 'max_value',
                            'actual': stats['max'],
                            'expected': exp_max,
                            'severity': min(1.0, abs(stats['max'] - exp_max) / (
                                        exp_max - exp_min) if exp_max > exp_min else 1.0)
                        })

                # Model-based anomaly detection
                if 'mean' in stats and 'std' in stats and 'min' in stats and 'max' in stats:
                    # Create feature vector
                    feature_vector = np.array([[
                        stats['mean'],
                        stats['std'],
                        stats['min'],
                        stats['max']
                    ]])

                    # Scale features
                    scaled_vector = scaler.transform(feature_vector)

                    # Predict anomaly score
                    try:
                        score = model.decision_function(scaled_vector)[0]
                        prediction = model.predict(scaled_vector)[0]

                        if prediction == -1 or score < self.anomaly_score_threshold:
                            anomalies.append({
                                'type': 'distribution',
                                'score': score,
                                'severity': min(1.0, abs(score - self.anomaly_score_threshold) / abs(
                                    self.anomaly_score_threshold) if self.anomaly_score_threshold != 0 else 1.0)
                            })
                    except Exception as e:
                        self.logger.warning(f"Error predicting anomaly for register {register}: {e}")

                # Add detected anomalies
                if anomalies:
                    # Get maximum severity
                    max_severity = max(a['severity'] for a in anomalies)

                    # Create anomaly record
                    anomaly = {
                        'category': 'register_value',
                        'register': register,
                        'description': f"Anomalous values for register {register}",
                        'anomalies': anomalies,
                        'severity': max_severity,
                        'expected': expected,
                        'actual': stats
                    }

                    # Add to test anomalies
                    self.anomalies[test_id]['anomalies'].append(anomaly)

    def _detect_sequence_anomalies(self, tests):
        """
        Detect anomalies in register sequences

        Parameters:
        tests (dict): Dictionary of tests with features
        """
        self.logger.debug("Detecting register sequence anomalies")

        # Get sequence models
        sequence_models = self.behavior_models.get('register_sequences', {})

        if not sequence_models:
            return

        # Get common sequences
        common_sequences = sequence_models.get('common_sequences', {})

        if not common_sequences:
            return

        # Process each test
        for test_id, test_data in tests.items():
            if 'features' not in test_data or 'register_sequences' not in test_data['features']:
                continue

            seq_features = test_data['features']['register_sequences']

            # Skip if no top sequences
            if 'top_sequences' not in seq_features:
                continue

            # Check for missing common sequences
            missing_common_seqs = []

            for common_seq, common_data in common_sequences.items():
                # Skip sequences with low frequency
                if common_data['frequency'] < 0.05:
                    continue

                # Check if sequence is present in test
                test_seqs = dict(seq_features['top_sequences'])

                if common_seq not in test_seqs:
                    missing_common_seqs.append({
                        'sequence': common_seq,
                        'expected_frequency': common_data['frequency'],
                        'expected_count': common_data['count'],
                        'severity': min(1.0, common_data['frequency'] * 2)  # Scale by frequency
                    })

            # Add anomaly if missing important sequences
            if missing_common_seqs:
                # Sort by severity
                missing_common_seqs.sort(key=lambda x: x['severity'], reverse=True)

                # Create anomaly record
                anomaly = {
                    'category': 'register_sequence',
                    'description': f"Missing {len(missing_common_seqs)} common register sequences",
                    'anomalies': missing_common_seqs,
                    'severity': missing_common_seqs[0]['severity'] if missing_common_seqs else 0.5
                }

                # Add to test anomalies
                self.anomalies[test_id]['anomalies'].append(anomaly)

            # Check for unusual sequences
            if 'top_sequences' in seq_features:
                unusual_seqs = []

                for seq, count in seq_features['top_sequences']:
                    # Skip if it's a common sequence
                    if seq in common_sequences:
                        continue

                    # Add as unusual sequence if it appears frequently in this test
                    if count > 5:
                        unusual_seqs.