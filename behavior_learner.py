"""
Behavior learning functionality for AI Debug Assist.
Learns expected behaviors from passing tests using extracted features.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings


class BehaviorLearner:
    def __init__(self, model_dir=None, logger=None):
        """
        Initialize the Behavior Learner

        Parameters:
        model_dir (str): Directory to save trained models
        logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_dir = model_dir

        # Create model directory if it doesn't exist
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)

        # Initialize models
        self.models = {}
        self.scalers = {}
        self.expected_values = {}
        self.register_groups = {}
        self.fsm_models = {}
        self.sequence_models = {}
        self.temporal_models = {}

        # Initialize thresholds and parameters
        self.contamination = 0.05  # Default anomaly ratio

    def learn_behaviors(self, tests):
        """
        Learn expected behaviors from passing tests

        Parameters:
        tests (dict): Dictionary of tests with extracted features

        Returns:
        dict: Dictionary of learned models and behaviors
        """
        self.logger.info("Learning expected behaviors from passing tests")

        # Filter passing tests
        passing_tests = {test_id: test_data for test_id, test_data in tests.items()
                         if test_data.get('status', '').lower() == 'pass'}

        if not passing_tests:
            self.logger.warning("No passing tests found for learning behaviors")
            return {}

        self.logger.info(f"Learning behaviors from {len(passing_tests)} passing tests")

        # Learn behaviors from different feature types
        if self._check_features(passing_tests, 'register_values'):
            self._learn_register_value_behaviors(passing_tests)

        if self._check_features(passing_tests, 'register_sequences'):
            self._learn_sequence_behaviors(passing_tests)

        if self._check_features(passing_tests, 'fsm_transitions'):
            self._learn_fsm_behaviors(passing_tests)

        if self._check_features(passing_tests, 'temporal_patterns'):
            self._learn_temporal_behaviors(passing_tests)

        if self._check_features(passing_tests, 'register_access_patterns'):
            self._learn_access_pattern_behaviors(passing_tests)

        # Save models if directory is specified
        if self.model_dir:
            self._save_models()

        self.logger.info("Behavior learning complete")
        return self.models

    def _check_features(self, tests, feature_type):
        """
        Check if a feature type exists in the tests

        Parameters:
        tests (dict): Dictionary of tests
        feature_type (str): Feature type to check

        Returns:
        bool: True if feature type exists, False otherwise
        """
        for test_id, test_data in tests.items():
            if 'features' in test_data and feature_type in test_data['features']:
                return True
        return False

    def _learn_register_value_behaviors(self, tests):
        """
        Learn expected behaviors for register values

        Parameters:
        tests (dict): Dictionary of passing tests with features
        """
        self.logger.info("Learning register value behaviors")

        # Collect register values across all tests
        all_registers = set()
        register_data = defaultdict(list)

        for test_id, test_data in tests.items():
            if 'features' in test_data and 'register_values' in test_data['features']:
                reg_features = test_data['features']['register_values']

                for register, stats in reg_features.items():
                    all_registers.add(register)
                    # Extract key statistics
                    if 'mean' in stats and 'std' in stats and 'min' in stats and 'max' in stats:
                        register_data[register].append({
                            'mean': stats['mean'],
                            'std': stats['std'],
                            'min': stats['min'],
                            'max': stats['max'],
                            'test_id': test_id
                        })

        # Learn behaviors for each register
        register_models = {}

        for register, data_points in register_data.items():
            # Skip if not enough data points
            if len(data_points) < 5:
                continue

            # Convert to DataFrame
            df = pd.DataFrame(data_points)

            # Calculate expected values
            expected_values = {
                'mean': df['mean'].mean(),
                'std': df['std'].mean(),
                'min': df['min'].min(),
                'max': df['max'].max(),
                'range': [df['min'].min(), df['max'].max()],
                'percentiles': [
                    df['mean'].quantile(0.05),
                    df['mean'].quantile(0.95)
                ]
            }

            # Create feature matrix for anomaly detection
            features = df[['mean', 'std', 'min', 'max']].values

            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Train isolation forest model for anomaly detection
            model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=42
            )

            try:
                model.fit(scaled_features)

                # Store model, scaler, and expected values
                register_models[register] = {
                    'model': model,
                    'scaler': scaler,
                    'expected_values': expected_values,
                    'num_samples': len(data_points)
                }
            except Exception as e:
                self.logger.warning(f"Failed to train model for register {register}: {e}")

        # Group registers with similar behaviors
        self._group_similar_registers(register_models)

        # Store models
        self.models['register_values'] = register_models
        self.logger.info(f"Learned behaviors for {len(register_models)} registers")

    def _learn_sequence_behaviors(self, tests):
        """
        Learn expected behaviors for register sequences

        Parameters:
        tests (dict): Dictionary of passing tests with features
        """
        self.logger.info("Learning register sequence behaviors")

        # Collect sequence data across all tests
        sequence_counts = defaultdict(int)
        total_sequences = 0

        for test_id, test_data in tests.items():
            if 'features' in test_data and 'register_sequences' in test_data['features']:
                seq_features = test_data['features']['register_sequences']

                if 'top_sequences' in seq_features:
                    for seq, count in seq_features['top_sequences']:
                        sequence_counts[seq] += count
                        total_sequences += count

        # Identify common sequences
        if total_sequences > 0:
            common_sequences = {}
            for seq, count in sequence_counts.items():
                if count > 5:  # Minimum occurrence threshold
                    common_sequences[seq] = {
                        'count': count,
                        'frequency': count / total_sequences
                    }

            # Store common sequences
            self.sequence_models['common_sequences'] = common_sequences
            self.models['register_sequences'] = self.sequence_models

            self.logger.info(f"Learned {len(common_sequences)} common register sequences")

    def _learn_fsm_behaviors(self, tests):
        """
        Learn expected behaviors for FSM transitions

        Parameters:
        tests (dict): Dictionary of passing tests with features
        """
        self.logger.info("Learning FSM transition behaviors")

        # Collect FSM data across all tests
        fsm_data = defaultdict(lambda: defaultdict(list))

        for test_id, test_data in tests.items():
            if 'features' in test_data and 'fsm_transitions' in test_data['features']:
                fsm_features = test_data['features']['fsm_transitions']

                for fsm_name, stats in fsm_features.items():
                    if 'transition_counts' in stats:
                        # Add transition counts
                        fsm_data[fsm_name]['transitions'].append(stats['transition_counts'])

                    if 'state_counts' in stats:
                        # Add state counts
                        fsm_data[fsm_name]['states'].append(stats['state_counts'])

        # Learn behaviors for each FSM
        fsm_models = {}

        for fsm_name, data in fsm_data.items():
            fsm_model = {
                'expected_transitions': self._merge_transition_counts(data['transitions']),
                'expected_states': self._merge_state_counts(data['states']),
                'valid_states': self._get_valid_states(data['states'])
            }

            fsm_models[fsm_name] = fsm_model

        # Store FSM models
        self.fsm_models = fsm_models
        self.models['fsm_transitions'] = fsm_models

        self.logger.info(f"Learned behaviors for {len(fsm_models)} FSMs")

    def _learn_temporal_behaviors(self, tests):
        """
        Learn expected behaviors for temporal patterns

        Parameters:
        tests (dict): Dictionary of passing tests with features
        """
        self.logger.info("Learning temporal pattern behaviors")

        # Collect temporal data across all tests
        temporal_data = []

        for test_id, test_data in tests.items():
            if 'features' in test_data and 'temporal_patterns' in test_data['features']:
                temp_features = test_data['features']['temporal_patterns']

                # Extract key metrics
                metrics = {}
                for key in ['test_duration', 'access_rate', 'read_count', 'write_count', 'read_write_ratio']:
                    if key in temp_features:
                        metrics[key] = temp_features[key]

                if metrics:
                    metrics['test_id'] = test_id
                    temporal_data.append(metrics)

        # Skip if not enough data
        if len(temporal_data) < 5:
            self.logger.warning("Not enough temporal data for learning behaviors")
            return

        # Convert to DataFrame
        df = pd.DataFrame(temporal_data)

        # Calculate expected values
        expected_values = {}
        for col in df.columns:
            if col != 'test_id':
                expected_values[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'percentiles': [
                        df[col].quantile(0.05),
                        df[col].quantile(0.95)
                    ]
                }

        # Create feature matrix for anomaly detection
        numeric_cols = [col for col in df.columns if col != 'test_id' and pd.api.types.is_numeric_dtype(df[col])]

        if not numeric_cols:
            self.logger.warning("No numeric columns found for temporal pattern learning")
            return

        features = df[numeric_cols].values

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Train isolation forest model for anomaly detection
        model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42
        )

        try:
            model.fit(scaled_features)

            # Store model, scaler, and expected values
            self.temporal_models['global'] = {
                'model': model,
                'scaler': scaler,
                'expected_values': expected_values,
                'feature_columns': numeric_cols
            }

            self.models['temporal_patterns'] = self.temporal_models

            self.logger.info("Learned global temporal behavior patterns")
        except Exception as e:
            self.logger.warning(f"Failed to train temporal model: {e}")

    def _learn_access_pattern_behaviors(self, tests):
        """
        Learn expected behaviors for register access patterns

        Parameters:
        tests (dict): Dictionary of passing tests with features
        """
        self.logger.info("Learning register access pattern behaviors")

        # Collect read-write ratios across all tests
        rw_ratios = defaultdict(list)

        for test_id, test_data in tests.items():
            if 'features' in test_data and 'register_access_patterns' in test_data['features']:
                access_features = test_data['features']['register_access_patterns']

                if 'read_write_ratios' in access_features:
                    for register, ratio in access_features['read_write_ratios'].items():
                        # Handle infinite ratios (read-only registers)
                        if ratio == float('inf'):
                            rw_ratios[register].append(1000.0)  # Use a large number instead
                        else:
                            rw_ratios[register].append(ratio)

        # Calculate expected read-write ratios
        expected_rw_ratios = {}
        for register, ratios in rw_ratios.items():
            if len(ratios) >= 5:  # Minimum sample threshold
                expected_rw_ratios[register] = {
                    'mean': np.mean(ratios),
                    'std': np.std(ratios),
                    'min': np.min(ratios),
                    'max': np.max(ratios),
                    'read_only': np.mean(ratios) > 100.0  # Consider read-only if average ratio is very high
                }

        # Store expected read-write ratios
        self.models['register_access_patterns'] = {
            'read_write_ratios': expected_rw_ratios
        }

        self.logger.info(f"Learned access patterns for {len(expected_rw_ratios)} registers")

    def _merge_transition_counts(self, transition_lists):
        """
        Merge transition counts from multiple tests

        Parameters:
        transition_lists (list): List of transition count dictionaries

        Returns:
        dict: Merged transition counts
        """
        merged = defaultdict(int)
        for transitions in transition_lists:
            for transition, count in transitions.items():
                merged[transition] += count

        return dict(merged)

    def _merge_state_counts(self, state_lists):
        """
        Merge state counts from multiple tests

        Parameters:
        state_lists (list): List of state count dictionaries

        Returns:
        dict: Merged state counts
        """
        merged = defaultdict(int)
        for states in state_lists:
            for state, count in states.items():
                merged[state] += count

        return dict(merged)

    def _get_valid_states(self, state_lists):
        """
        Get valid states from all tests

        Parameters:
        state_lists (list): List of state count dictionaries

        Returns:
        set: Set of valid states
        """
        valid_states = set()
        for states in state_lists:
            for state in states.keys():
                valid_states.add(state)

        return valid_states

    def _group_similar_registers(self, register_models):
        """
        Group registers with similar behaviors

        Parameters:
        register_models (dict): Dictionary of register models
        """
        # Extract features for clustering
        registers = []
        features = []

        for register, model_data in register_models.items():
            expected = model_data['expected_values']

            # Extract key metrics as features
            feature_vector = [
                expected['mean'],
                expected['std'],
                expected['min'],
                expected['max']
            ]

            registers.append(register)
            features.append(feature_vector)

        # Skip if not enough registers
        if len(registers) < 5:
            return

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Cluster registers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                # Use DBSCAN for clustering
                clustering = DBSCAN(
                    eps=0.5,
                    min_samples=3,
                    metric='euclidean'
                )

                cluster_labels = clustering.fit_predict(scaled_features)

                # Group registers by cluster
                register_groups = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    if label != -1:  # Skip noise points
                        register_groups[int(label)].append(registers[i])

                # Store register groups
                self.register_groups = {f"group_{i}": regs for i, regs in register_groups.items()}

                self.logger.info(f"Grouped registers into {len(register_groups)} clusters")
            except Exception as e:
                self.logger.warning(f"Failed to cluster registers: {e}")

    def _save_models(self):
        """
        Save trained models to disk
        """
        try:
            # Save models dictionary
            models_path = os.path.join(self.model_dir, 'behavior_models.pkl')
            joblib.dump(self.models, models_path)

            # Save register groups
            groups_path = os.path.join(self.model_dir, 'register_groups.pkl')
            joblib.dump(self.register_groups, groups_path)

            self.logger.info(f"Saved behavior models to {self.model_dir}")
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")

    def load_models(self, model_dir=None):
        """
        Load trained models from disk

        Parameters:
        model_dir (str): Directory containing trained models

        Returns:
        bool: True if successful, False otherwise
        """
        # Use instance model directory if not specified
        if model_dir is None:
            model_dir = self.model_dir

        if not model_dir or not os.path.exists(model_dir):
            self.logger.error(f"Model directory {model_dir} does not exist")
            return False

        try:
            # Load models dictionary
            models_path = os.path.join(model_dir, 'behavior_models.pkl')
            if os.path.exists(models_path):
                self.models = joblib.load(models_path)

            # Load register groups
            groups_path = os.path.join(model_dir, 'register_groups.pkl')
            if os.path.exists(groups_path):
                self.register_groups = joblib.load(groups_path)

            self.logger.info(f"Loaded behavior models from {model_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False