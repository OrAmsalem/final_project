"""
AI-enhanced feature extraction for ptracker hardware validation logs.
Extracts meaningful features from tokenized logs using machine learning techniques.
Specifically designed for Intel's pre-silicon validation log format with focus on:
- Register value patterns in ptracker.log.gz
- FSM state transitions
- Temporal access patterns
- Anomaly detection in hardware behavior
"""
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime


class AIFeatureExtractor:
    def __init__(self, logger=None):
        """
        Initialize the AI Feature Extractor

        Parameters:
        logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Initialize vectorizers for textual features
        # Modified for ptracker.log format which has register descriptions
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),  # Capture longer register path components
            stop_words='english',
            analyzer='word',
            token_pattern=r'[^\[\]\.]+|\[\d+\]'  # Better handle register paths with array indices
        )

        # Initialize dimensionality reduction
        self.pca = PCA(n_components=10)

        # Initialize anomaly detection models
        self.temporal_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

        self.value_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

        # Initialize scaler
        self.scaler = StandardScaler()

        # Storage for extracted features
        self.feature_cache = {}
        self.model_cache = {}

    def extract_features(self, tests, feature_types=None):
        """
        Extract features from tokenized test logs

        Parameters:
        tests (dict): Dictionary of tests with tokenized logs
        feature_types (list): List of feature types to extract (None for all)

        Returns:
        dict: Dictionary of tests with extracted features
        """
        self.logger.info("Extracting features from tokenized logs")

        # Default feature types to extract all
        if feature_types is None:
            feature_types = [
                'register_values',
                'register_sequences',
                'fsm_transitions',
                'register_access_patterns',
                'temporal_patterns',
                'error_contexts',
                'token_embeddings'
            ]

        # Convert tests to a list of records for easier processing
        test_records = []
        for test_id, test_data in tests.items():
            if 'tokens' in test_data:
                test_records.append({
                    'test_id': test_id,
                    'tokens': test_data['tokens'],
                    'status': test_data.get('status', 'unknown')
                })

        self.logger.info(f"Processing {len(test_records)} tests for feature extraction")

        # Extract each feature type
        features = {}

        if 'register_values' in feature_types:
            register_features = self._extract_register_value_features(test_records)
            features['register_values'] = register_features

        if 'register_sequences' in feature_types:
            sequence_features = self._extract_register_sequence_features(test_records)
            features['register_sequences'] = sequence_features

        if 'fsm_transitions' in feature_types:
            fsm_features = self._extract_fsm_transition_features(test_records)
            features['fsm_transitions'] = fsm_features

        if 'register_access_patterns' in feature_types:
            access_features = self._extract_register_access_patterns(test_records)
            features['register_access_patterns'] = access_features

        if 'temporal_patterns' in feature_types:
            temporal_features = self._extract_temporal_patterns(test_records)
            features['temporal_patterns'] = temporal_features

        if 'error_contexts' in feature_types:
            error_features = self._extract_error_context_features(test_records)
            features['error_contexts'] = error_features

        if 'token_embeddings' in feature_types:
            embedding_features = self._extract_token_embeddings(test_records)
            features['token_embeddings'] = embedding_features

        # Update each test with the extracted features
        for test_id, test_data in tests.items():
            test_data['features'] = {}
            for feature_type, feature_data in features.items():
                if test_id in feature_data:
                    test_data['features'][feature_type] = feature_data[test_id]

        self.logger.info(f"Feature extraction complete")
        return tests

    def _extract_register_value_features(self, test_records):
        """
        Extract features related to register values and their distributions
        Specifically tailored for ptracker.log format with:
        [timestamp] action [address] size description value [comment]

        Parameters:
        test_records (list): List of test records with tokens

        Returns:
        dict: Dictionary mapping test IDs to register value features
        """
        self.logger.debug("Extracting register value features from ptracker logs")

        # Features keyed by test_id
        features = {}

        # Identify important registers (using AI tokenizer results if available)
        important_registers = self._identify_important_registers(test_records)

        # Process each test
        for record in test_records:
            test_id = record['test_id']
            tokens = record['tokens']

            # Skip if no register tokens
            if 'registers' not in tokens or not tokens['registers']:
                features[test_id] = {}
                continue

            # Group register values by register path
            register_values = defaultdict(list)
            for token in tokens['registers']:
                register = token['description']
                if 'value' in token and token['value']:
                    try:
                        # Convert hex value to integer
                        value = int(token['value'], 16)
                        register_values[register].append({
                            'timestamp': int(token['timestamp']),
                            'value': value,
                            'raw': token['value']
                        })
                    except (ValueError, TypeError):
                        # Skip if value can't be converted
                        pass

            # Calculate statistics for each register
            reg_features = {}
            for register, values in register_values.items():
                # Skip if not enough values for meaningful statistics
                if len(values) < 3:
                    continue

                # Extract values only
                value_list = [v['value'] for v in values]

                # Calculate basic statistics
                stats = {
                    'count': len(value_list),
                    'min': min(value_list),
                    'max': max(value_list),
                    'mean': np.mean(value_list),
                    'median': np.median(value_list),
                    'std': np.std(value_list),
                    'unique_values': len(set(value_list)),
                    'range': max(value_list) - min(value_list)
                }

                # Calculate additional statistics for bit patterns
                bit_stats = self._calculate_bit_statistics(value_list)
                stats.update(bit_stats)

                # Calculate sequence-based statistics
                if len(value_list) > 1:
                    # Calculate differences between consecutive values
                    diffs = np.diff(value_list)
                    stats['mean_diff'] = np.mean(diffs)
                    stats['max_diff'] = np.max(np.abs(diffs))
                    stats['diff_std'] = np.std(diffs)

                    # Count value transitions
                    transitions = Counter(zip(value_list[:-1], value_list[1:]))
                    stats['unique_transitions'] = len(transitions)

                    # Calculate how often the register changes value
                    stats['change_frequency'] = np.sum(np.abs(diffs) > 0) / (len(value_list) - 1)

                # Add importance score if available from AI tokenizer
                if register in important_registers:
                    stats['importance'] = important_registers[register]
                else:
                    stats['importance'] = 0.0

                reg_features[register] = stats

            features[test_id] = reg_features

        return features

    def _extract_register_sequence_features(self, test_records):
        """
        Extract features related to register access sequences

        Parameters:
        test_records (list): List of test records with tokens

        Returns:
        dict: Dictionary mapping test IDs to register sequence features
        """
        self.logger.debug("Extracting register sequence features")

        # Features keyed by test_id
        features = {}

        # Process each test
        for record in test_records:
            test_id = record['test_id']
            tokens = record['tokens']

            # Skip if no register tokens
            if 'registers' not in tokens or not tokens['registers']:
                features[test_id] = {}
                continue

            # Sort register tokens by timestamp
            sorted_tokens = sorted(tokens['registers'], key=lambda x: int(x['timestamp']))

            # Extract register access sequences
            sequences = []
            sequence_length = 3  # Length of sequences to analyze

            # Create sequences of consecutive accesses
            for i in range(len(sorted_tokens) - sequence_length + 1):
                seq = sorted_tokens[i:i + sequence_length]
                seq_data = {
                    'registers': [token['description'] for token in seq],
                    'actions': [token['action'] for token in seq],
                    'values': [token['value'] for token in seq],
                    'start_time': int(seq[0]['timestamp']),
                    'end_time': int(seq[-1]['timestamp']),
                    'duration': int(seq[-1]['timestamp']) - int(seq[0]['timestamp'])
                }
                sequences.append(seq_data)

            # Count occurrences of each register sequence
            register_sequences = ['->'.join(s['registers']) for s in sequences]
            sequence_counts = Counter(register_sequences)

            # Calculate sequence statistics
            seq_features = {
                'total_sequences': len(sequences),
                'unique_sequences': len(sequence_counts),
                'top_sequences': sequence_counts.most_common(10),
                'sequence_entropy': self._calculate_entropy(sequence_counts.values())
            }

            # If AI tokenizer identified important sequences, add them
            if 'ai_sequences' in tokens:
                seq_features['ai_identified_sequences'] = tokens['ai_sequences']

            features[test_id] = seq_features

        return features

    def _extract_fsm_transition_features(self, test_records):
        """
        Extract features related to FSM state transitions

        Parameters:
        test_records (list): List of test records with tokens

        Returns:
        dict: Dictionary mapping test IDs to FSM transition features
        """
        self.logger.debug("Extracting FSM transition features")

        # Features keyed by test_id
        features = {}

        # Process each test
        for record in test_records:
            test_id = record['test_id']
            tokens = record['tokens']

            # Skip if no FSM state tokens
            if 'fsm_states' not in tokens or not tokens['fsm_states']:
                features[test_id] = {}
                continue

            # Group FSM transitions by FSM name
            # Specifically handling the FSM patterns from the tokenizer:
            # - Sleep State FSM transition: from_state -> to_state
            # - Generic FSM state transitions: FSM fsm_name: from_state -> to_state
            fsm_transitions = defaultdict(list)
            for token in tokens['fsm_states']:
                # Process based on token format
                if 'match' in token:
                    if len(token['match']) == 3:  # Generic FSM format: FSM name, from_state, to_state
                        fsm_name, from_state, to_state = token['match']
                    elif len(token['match']) == 2:  # Sleep state FSM format: from_state, to_state
                        fsm_name = "SleepStateFSM"
                        from_state, to_state = token['match']
                    else:
                        continue

                    fsm_transitions[fsm_name].append({
                        'from_state': from_state,
                        'to_state': to_state,
                        'timestamp': int(token.get('timestamp', 0))
                    })

            # Calculate FSM transition statistics
            fsm_features = {}
            for fsm_name, transitions in fsm_transitions.items():
                # Skip if not enough transitions
                if len(transitions) < 2:
                    continue

                # Sort transitions by timestamp
                sorted_transitions = sorted(transitions, key=lambda x: x['timestamp'])

                # Count occurrences of each state
                states = [t['from_state'] for t in sorted_transitions] + [sorted_transitions[-1]['to_state']]
                state_counts = Counter(states)

                # Count occurrences of each transition
                transition_pairs = [(t['from_state'], t['to_state']) for t in sorted_transitions]
                transition_counts = Counter(transition_pairs)

                # Create state transition matrix
                unique_states = list(state_counts.keys())
                transition_matrix = np.zeros((len(unique_states), len(unique_states)))
                state_to_index = {state: i for i, state in enumerate(unique_states)}

                for (from_state, to_state), count in transition_counts.items():
                    from_idx = state_to_index[from_state]
                    to_idx = state_to_index[to_state]
                    transition_matrix[from_idx, to_idx] = count

                # Calculate statistics
                fsm_stats = {
                    'states': unique_states,
                    'state_counts': dict(state_counts),
                    'transition_counts': {f"{from_}->{to_}": count for (from_, to_), count in
                                          transition_counts.items()},
                    'total_transitions': len(transitions),
                    'unique_transitions': len(transition_counts),
                    'unique_states': len(unique_states),
                    'start_state': sorted_transitions[0]['from_state'],
                    'end_state': sorted_transitions[-1]['to_state'],
                    'transition_entropy': self._calculate_entropy(transition_counts.values())
                }

                # Identify cyclic behaviors
                cycle_stats = self._identify_cycles(transition_matrix, unique_states)
                fsm_stats.update(cycle_stats)

                fsm_features[fsm_name] = fsm_stats

            features[test_id] = fsm_features

        return features

    def _extract_register_access_patterns(self, test_records):
        """
        Extract features related to register access patterns

        Parameters:
        test_records (list): List of test records with tokens

        Returns:
        dict: Dictionary mapping test IDs to register access pattern features
        """
        self.logger.debug("Extracting register access pattern features")

        # Features keyed by test_id
        features = {}

        # Process each test
        for record in test_records:
            test_id = record['test_id']
            tokens = record['tokens']

            # Skip if no register tokens
            if 'registers' not in tokens or not tokens['registers']:
                features[test_id] = {}
                continue

            # Convert to dataframe for easier manipulation
            register_df = pd.DataFrame(tokens['registers'])

            # Ensure timestamp is numeric
            register_df['timestamp'] = register_df['timestamp'].astype(int)

            # Create access pattern features
            access_features = {}

            # 1. Read-to-write ratio for each register
            reg_access_counts = defaultdict(lambda: {'reads': 0, 'writes': 0})
            for _, row in register_df.iterrows():
                register = row['description']
                action = row['action']

                if action == 'MEM_R':
                    reg_access_counts[register]['reads'] += 1
                elif action == 'MEM_W':
                    reg_access_counts[register]['writes'] += 1

            # Calculate read/write ratios
            rw_ratios = {}
            for register, counts in reg_access_counts.items():
                if counts['writes'] > 0:
                    rw_ratios[register] = counts['reads'] / counts['writes']
                else:
                    rw_ratios[register] = float('inf')  # Infinite ratio for read-only registers

            access_features['read_write_ratios'] = rw_ratios

            # 2. Temporal access patterns
            if len(register_df) > 0:
                # Convert timestamps to relative time (seconds from start)
                min_ts = register_df['timestamp'].min()
                register_df['relative_time'] = (register_df['timestamp'] - min_ts) / 1e9  # Assuming ns

                # Group by register
                reg_access_times = defaultdict(lambda: {'read_times': [], 'write_times': []})
                for _, row in register_df.iterrows():
                    register = row['description']
                    action = row['action']
                    rel_time = row['relative_time']

                    if action == 'MEM_R':
                        reg_access_times[register]['read_times'].append(rel_time)
                    elif action == 'MEM_W':
                        reg_access_times[register]['write_times'].append(rel_time)

                # Calculate temporal statistics
                temporal_stats = {}
                for register, times in reg_access_times.items():
                    read_times = times['read_times']
                    write_times = times['write_times']

                    stats = {}

                    # Skip if not enough data points
                    if len(read_times) + len(write_times) < 3:
                        continue

                    # Read timing statistics
                    if len(read_times) > 1:
                        read_diffs = np.diff(read_times)
                        stats['read_mean_interval'] = np.mean(read_diffs)
                        stats['read_std_interval'] = np.std(read_diffs)

                    # Write timing statistics
                    if len(write_times) > 1:
                        write_diffs = np.diff(write_times)
                        stats['write_mean_interval'] = np.mean(write_diffs)
                        stats['write_std_interval'] = np.std(write_diffs)

                    # Combined timing statistics
                    all_times = sorted(read_times + write_times)
                    if len(all_times) > 1:
                        all_diffs = np.diff(all_times)
                        stats['mean_interval'] = np.mean(all_diffs)
                        stats['std_interval'] = np.std(all_diffs)
                        stats['max_interval'] = np.max(all_diffs)
                        stats['access_frequency'] = len(all_times) / (all_times[-1] - all_times[0]) if all_times[-1] > \
                                                                                                       all_times[
                                                                                                           0] else 0

                    temporal_stats[register] = stats

                access_features['temporal_statistics'] = temporal_stats

            # 3. Register access bursts
            # Define a burst as multiple accesses to the same register within a short time window
            burst_window = 1e-6  # 1 microsecond
            register_bursts = defaultdict(list)

            for register, times in reg_access_times.items():
                all_times = sorted(times['read_times'] + times['write_times'])

                if len(all_times) < 2:
                    continue

                # Identify bursts
                current_burst = [all_times[0]]
                for t in all_times[1:]:
                    if t - current_burst[-1] <= burst_window:
                        current_burst.append(t)
                    else:
                        if len(current_burst) > 1:
                            register_bursts[register].append(current_burst)
                        current_burst = [t]

                # Add the last burst if it contains multiple accesses
                if len(current_burst) > 1:
                    register_bursts[register].append(current_burst)

            # Calculate burst statistics
            burst_stats = {}
            for register, bursts in register_bursts.items():
                if not bursts:
                    continue

                burst_lengths = [len(burst) for burst in bursts]
                burst_stats[register] = {
                    'number_of_bursts': len(bursts),
                    'mean_burst_length': np.mean(burst_lengths),
                    'max_burst_length': np.max(burst_lengths),
                    'total_burst_accesses': sum(burst_lengths)
                }

            access_features['burst_statistics'] = burst_stats

            features[test_id] = access_features

        return features

    def _extract_temporal_patterns(self, test_records):
        """
        Extract features related to temporal patterns across all register accesses

        Parameters:
        test_records (list): List of test records with tokens

        Returns:
        dict: Dictionary mapping test IDs to temporal pattern features
        """
        self.logger.debug("Extracting temporal pattern features")

        # Features keyed by test_id
        features = {}

        # Process each test
        for record in test_records:
            test_id = record['test_id']
            tokens = record['tokens']

            # Skip if no register tokens
            if 'registers' not in tokens or not tokens['registers']:
                features[test_id] = {}
                continue

            # Convert to dataframe
            register_df = pd.DataFrame(tokens['registers'])

            # Ensure timestamp is numeric
            register_df['timestamp'] = register_df['timestamp'].astype(int)

            # Skip if no data
            if len(register_df) == 0:
                features[test_id] = {}
                continue

            # Convert timestamps to relative time (seconds from start)
            min_ts = register_df['timestamp'].min()
            max_ts = register_df['timestamp'].max()
            register_df['relative_time'] = (register_df['timestamp'] - min_ts) / 1e9  # Assuming ns

            # Calculate test duration
            test_duration = (max_ts - min_ts) / 1e9  # seconds

            # Create time bins (100 bins over the test duration)
            n_bins = 100
            bins = np.linspace(0, register_df['relative_time'].max(), n_bins + 1)
            register_df['time_bin'] = pd.cut(register_df['relative_time'], bins, labels=False)

            # Count accesses by type in each time bin
            read_counts = register_df[register_df['action'] == 'MEM_R'].groupby('time_bin').size()
            write_counts = register_df[register_df['action'] == 'MEM_W'].groupby('time_bin').size()

            # Fill in missing bins with zeros
            full_bins = pd.Series(np.zeros(n_bins), index=range(n_bins))
            read_counts = read_counts.add(full_bins, fill_value=0)
            write_counts = write_counts.add(full_bins, fill_value=0)

            # Calculate temporal activity features
            temporal_features = {
                'test_duration': test_duration,
                'total_accesses': len(register_df),
                'access_rate': len(register_df) / test_duration if test_duration > 0 else 0,
                'read_count': (register_df['action'] == 'MEM_R').sum(),
                'write_count': (register_df['action'] == 'MEM_W').sum(),
                'read_write_ratio': (register_df['action'] == 'MEM_R').sum() /
                                    (register_df['action'] == 'MEM_W').sum()
                if (register_df['action'] == 'MEM_W').sum() > 0 else float('inf'),
                'peak_activity_bin': read_counts.add(write_counts).idxmax(),
                'activity_distribution': {
                    'reads': read_counts.tolist(),
                    'writes': write_counts.tolist()
                }
            }

            # Identify activity phases
            # Phase is a period with similar activity patterns
            phase_features = self._identify_activity_phases(read_counts, write_counts)
            temporal_features['phase_analysis'] = phase_features

            # Calculate autocorrelation to identify periodic patterns
            if len(read_counts) > 10:
                read_autocorr = self._calculate_autocorrelation(read_counts.values, 10)
                write_autocorr = self._calculate_autocorrelation(write_counts.values, 10)

                temporal_features['periodicity'] = {
                    'read_autocorrelation': read_autocorr.tolist(),
                    'write_autocorrelation': write_autocorr.tolist(),
                    'has_periodicity': np.max(np.abs(read_autocorr[1:])) > 0.5 or np.max(
                        np.abs(write_autocorr[1:])) > 0.5
                }

            features[test_id] = temporal_features

        return features

    def _extract_error_context_features(self, test_records):
        """
        Extract features related to error messages and their context

        Parameters:
        test_records (list): List of test records with tokens

        Returns:
        dict: Dictionary mapping test IDs to error context features
        """
        self.logger.debug("Extracting error context features")

        # Features keyed by test_id
        features = {}

        # Process each test
        for record in test_records:
            test_id = record['test_id']
            tokens = record['tokens']

            # Skip if no error tokens
            if 'errors' not in tokens or not tokens['errors']:
                features[test_id] = {'error_count': 0}
                continue

            # Extract error messages
            error_messages = []
            for token in tokens['errors']:
                if 'match' in token and token['match']:
                    error_messages.append({
                        'message': token['match'][0],
                        'timestamp': int(token.get('timestamp', 0)),
                        'raw': token.get('raw', '')
                    })

            # Skip if no error messages
            if not error_messages:
                features[test_id] = {'error_count': 0}
                continue

            # Sort errors by timestamp
            sorted_errors = sorted(error_messages, key=lambda x: x['timestamp'])

            # Find register accesses around each error
            error_contexts = []
            for error in sorted_errors:
                error_time = error['timestamp']

                # Find register accesses within 1 microsecond before the error
                time_window = 1000  # 1 microsecond in nanoseconds

                context_tokens = []
                if 'registers' in tokens:
                    for token in tokens['registers']:
                        token_time = int(token['timestamp'])
                        if error_time - time_window <= token_time <= error_time:
                            context_tokens.append(token)

                error_contexts.append({
                    'error': error,
                    'context_tokens': context_tokens,
                    'context_size': len(context_tokens)
                })

            # Calculate error context features
            error_features = {
                'error_count': len(sorted_errors),
                'first_error_time': sorted_errors[0]['timestamp'],
                'last_error_time': sorted_errors[-1]['timestamp'],
                'error_messages': [e['message'] for e in sorted_errors],
                'error_contexts': error_contexts,
                'avg_context_size': np.mean([c['context_size'] for c in error_contexts]) if error_contexts else 0
            }

            # Extract common words and phrases from error messages
            error_text = ' '.join([e['message'] for e in sorted_errors])
            common_terms = self._extract_common_terms(error_text)
            error_features['common_terms'] = common_terms

            features[test_id] = error_features

        return features

    def _extract_token_embeddings(self, test_records):
        """
        Extract embeddings from tokenized logs using TF-IDF

        Parameters:
        test_records (list): List of test records with tokens

        Returns:
        dict: Dictionary mapping test IDs to token embeddings
        """
        self.logger.debug("Extracting token embeddings")

        # Features keyed by test_id
        features = {}

        # Collect register descriptions from all tests
        all_descriptions = []
        test_ids = []

        for record in test_records:
            test_id = record['test_id']
            tokens = record['tokens']

            if 'registers' in tokens and tokens['registers']:
                # Create a document for this test by joining all register descriptions
                descriptions = [token['description'] for token in tokens['registers']]
                document = ' '.join(descriptions)

                all_descriptions.append(document)
                test_ids.append(test_id)

        # Skip if no data
        if not all_descriptions:
            return features

        # Create TF-IDF vectors
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_descriptions)

        # Optionally, apply dimensionality reduction
        if tfidf_matrix.shape[1] > 10:
            tfidf_reduced = self.pca.fit_transform(tfidf_matrix.toarray())
        else:
            tfidf_reduced = tfidf_matrix.toarray()

        # Store embeddings for each test
        for i, test_id in enumerate(test_ids):
            if tfidf_reduced.shape[0] > i:
                features[test_id] = {
                    'embedding': tfidf_reduced[i].tolist(),
                    'embedding_size': tfidf_reduced.shape[1]
                }

        return features

    def save_models(self, output_dir):
        """
        Save trained models and feature extractors for later use

        Parameters:
        output_dir (str): Directory to save models

        Returns:
        bool: True if successful, False otherwise
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Save TF-IDF vectorizer
            joblib.dump(self.tfidf_vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.pkl'))

            # Save PCA model
            joblib.dump(self.pca, os.path.join(output_dir, 'pca_model.pkl'))

            # Save anomaly detection models
            joblib.dump(self.temporal_model, os.path.join(output_dir, 'temporal_model.pkl'))
            joblib.dump(self.value_model, os.path.join(output_dir,