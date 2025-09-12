"""
AI-based token extraction from ptracker logs
Using statistical AI techniques to identify relevant patterns
"""
import numpy as np
import pandas as pd
import re
from collections import defaultdict, Counter
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy


class AITokenExtractor:
    def __init__(self, logger=None):
        """
        Initialize the AI Token Extractor

        Parameters:
        logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # For finding important registers based on descriptions
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )

        # For clustering similar access patterns
        self.cluster_model = DBSCAN(
            eps=0.5,
            min_samples=5,
            metric='euclidean'
        )

        # For identifying anomalous patterns
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

        # Data scaler for numerical features
        self.scaler = StandardScaler()

        # Storage for extracted patterns
        self.important_registers = []
        self.access_patterns = []
        self.sequence_patterns = []

    def extract_tokens(self, parsed_logs):
        """
        Extract relevant tokens using statistical AI techniques

        Parameters:
        parsed_logs (list): List of dictionaries containing parsed log entries

        Returns:
        dict: Extracted tokens and patterns
        """
        self.logger.info("Extracting tokens using statistical AI techniques")

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(parsed_logs)

        # Skip if empty
        if df.empty:
            return {}

        # Process access patterns
        try:
            # Find important registers based on TF-IDF scores of descriptions
            self.important_registers = self._find_important_registers(df)

            # Identify access patterns using clustering
            self.access_patterns = self._identify_access_patterns(df)

            # Extract sequence patterns
            self.sequence_patterns = self._extract_sequence_patterns(df)

            # Detect anomalous patterns
            anomalous_patterns = self._detect_anomalous_patterns(df)

            # Combine results
            tokens = self._combine_results(df, anomalous_patterns)

            self.logger.info(f"Extracted {len(tokens['registers'])} relevant register tokens")
            return tokens

        except Exception as e:
            self.logger.error(f"Error extracting tokens: {e}")
            return {}

    def _find_important_registers(self, df):
        """
        Find important registers based on TF-IDF analysis of descriptions

        Parameters:
        df (DataFrame): DataFrame containing parsed log entries

        Returns:
        list: Important register paths with scores
        """
        self.logger.debug("Finding important registers using TF-IDF")

        # Extract register descriptions
        descriptions = df['description'].fillna('').tolist()

        # Create TF-IDF representation
        tfidf_matrix = self.tfidf.fit_transform(descriptions)

        # Get feature names
        feature_names = self.tfidf.get_feature_names_out()

        # Calculate importance scores (sum of TF-IDF values)
        importance_scores = tfidf_matrix.sum(axis=0).A1

        # Map scores to feature names
        feature_scores = list(zip(feature_names, importance_scores))

        # Sort by importance score
        sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)

        # Get top N features
        top_features = sorted_features[:100]

        # Find register paths containing these important terms
        important_registers = []
        for term, score in top_features:
            # Find register paths containing this term
            matching_registers = df[df['description'].str.contains(term, na=False)]

            if not matching_registers.empty:
                for _, row in matching_registers.iterrows():
                    important_registers.append({
                        'register': row['description'],
                        'term': term,
                        'score': score,
                        'address': row.get('address', 'unknown')
                    })

        # Remove duplicates and sort by score
        unique_registers = {}
        for reg in important_registers:
            if reg['register'] not in unique_registers:
                unique_registers[reg['register']] = reg
            elif reg['score'] > unique_registers[reg['register']]['score']:
                unique_registers[reg['register']] = reg

        important_registers = list(unique_registers.values())
        important_registers.sort(key=lambda x: x['score'], reverse=True)

        self.logger.debug(f"Found {len(important_registers)} important registers")
        return important_registers

    def _identify_access_patterns(self, df):
        """
        Identify common access patterns using clustering

        Parameters:
        df (DataFrame): DataFrame containing parsed log entries

        Returns:
        list: Access pattern clusters
        """
        self.logger.debug("Identifying access patterns using clustering")

        # Extract features for clustering
        features = []

        # We'll use a representation of register access sequences
        # Group by timestamp and create features

        # First, convert timestamps to relative time (seconds from start)
        if 'timestamp' in df.columns:
            try:
                # Convert numeric timestamps to relative time (seconds from start)
                timestamps = df['timestamp'].astype(float)
                relative_time = (timestamps - timestamps.min()) / 1e9  # Assuming ns
                df['relative_time'] = relative_time
            except:
                # If conversion fails, just use a sequence number
                df['relative_time'] = range(len(df))
        else:
            df['relative_time'] = range(len(df))

        # Create features for each register
        register_features = {}
        for _, row in df.iterrows():
            register = row['description']
            time = row['relative_time']
            action = row.get('action', 'UNKNOWN')

            if register not in register_features:
                register_features[register] = {
                    'read_times': [],
                    'write_times': [],
                    'values': [],
                    'read_count': 0,
                    'write_count': 0
                }

            if 'MEM_R' in action:
                register_features[register]['read_times'].append(time)
                register_features[register]['read_count'] += 1
            elif 'MEM_W' in action:
                register_features[register]['write_times'].append(time)
                register_features[register]['write_count'] += 1

            # Add value as feature (if numeric)
            try:
                if 'value' in row and row['value']:
                    value = int(row['value'], 16)
                    register_features[register]['values'].append(value)
            except:
                pass

        # Convert to feature vectors
        for register, data in register_features.items():
            # Skip registers with few accesses
            if data['read_count'] + data['write_count'] < 3:
                continue

            # Create feature vector
            feature_vector = [
                data['read_count'],
                data['write_count'],
                len(data['values']),  # Number of unique values
                np.mean(data['values']) if data['values'] else 0,  # Mean value
                np.std(data['values']) if len(data['values']) > 1 else 0,  # Value std dev
                # Timing features
                np.mean(np.diff(data['read_times'])) if len(data['read_times']) > 1 else 0,
                np.mean(np.diff(data['write_times'])) if len(data['write_times']) > 1 else 0
            ]

            features.append({
                'register': register,
                'features': feature_vector
            })

        if not features:
            return []

        # Convert to array for clustering
        feature_array = np.array([f['features'] for f in features])

        # Normalize features
        normalized_features = self.scaler.fit_transform(feature_array)

        # Apply clustering
        labels = self.cluster_model.fit_predict(normalized_features)

        # Group registers by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Skip noise points
                clusters[int(label)].append(features[i]['register'])

        # Convert to list of clusters
        access_patterns = [{'cluster_id': cluster_id, 'registers': registers}
                           for cluster_id, registers in clusters.items()]

        self.logger.debug(f"Identified {len(access_patterns)} access pattern clusters")
        return access_patterns

    def _extract_sequence_patterns(self, df):
        """
        Extract common access sequence patterns

        Parameters:
        df (DataFrame): DataFrame containing parsed log entries

        Returns:
        list: Common sequence patterns
        """
        self.logger.debug("Extracting common sequence patterns")

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        # Extract sequences of consecutive accesses
        sequences = []
        sequence_length = 3  # Length of sequences to extract

        # Create sequences from consecutive entries
        for i in range(len(df) - sequence_length + 1):
            sequence = df.iloc[i:i + sequence_length]
            sequence_str = '->'.join(sequence['description'].tolist())
            sequences.append(sequence_str)

        # Count occurrences of each sequence
        sequence_counter = Counter(sequences)

        # Get the most common sequences
        common_sequences = sequence_counter.most_common(20)

        # Format results
        sequence_patterns = [
            {'sequence': seq, 'count': count}
            for seq, count in common_sequences
        ]

        self.logger.debug(f"Extracted {len(sequence_patterns)} common sequence patterns")
        return sequence_patterns

    def _detect_anomalous_patterns(self, df):
        """
        Detect anomalous patterns using isolation forest

        Parameters:
        df (DataFrame): DataFrame containing parsed log entries

        Returns:
        list: Anomalous patterns
        """
        self.logger.debug("Detecting anomalous patterns")

        # Create features for anomaly detection
        features = []
        register_values = defaultdict(list)

        # Group values by register
        for _, row in df.iterrows():
            register = row['description']
            if 'value' in row and row['value']:
                try:
                    value = int(row['value'], 16)
                    register_values[register].append(value)
                except:
                    pass

        # Calculate statistics for each register
        for register, values in register_values.items():
            if len(values) < 3:
                continue

            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val

            # Calculate entropy of value distribution
            hist, _ = np.histogram(values, bins=10)
            if np.sum(hist) > 0:
                probs = hist / np.sum(hist)
                entropy_val = entropy(probs)
            else:
                entropy_val = 0

            # Create feature vector
            feature_vector = [
                mean_val,
                std_val,
                min_val,
                max_val,
                range_val,
                entropy_val,
                len(values)  # Number of occurrences
            ]

            features.append({
                'register': register,
                'features': feature_vector
            })

        if not features:
            return []

        # Convert to array for anomaly detection
        feature_array = np.array([f['features'] for f in features])

        # Normalize features
        normalized_features = self.scaler.fit_transform(feature_array)

        # Apply anomaly detection
        anomaly_scores = self.anomaly_detector.fit_predict(normalized_features)

        # Extract anomalous registers (where score is -1)
        anomalous_registers = []
        for i, score in enumerate(anomaly_scores):
            if score == -1:
                anomalous_registers.append({
                    'register': features[i]['register'],
                    'anomaly_type': 'value_distribution',
                    'confidence': self.anomaly_detector.decision_function([normalized_features[i]])[0] * -1
                })

        # Sort by confidence
        anomalous_registers.sort(key=lambda x: x['confidence'], reverse=True)

        self.logger.debug(f"Detected {len(anomalous_registers)} anomalous patterns")
        return anomalous_registers

    def _combine_results(self, df, anomalous_patterns):
        """
        Combine results from different analyses to identify relevant tokens

        Parameters:
        df (DataFrame): DataFrame containing parsed log entries
        anomalous_patterns (list): Anomalous patterns detected

        Returns:
        dict: Combined results with relevant tokens
        """
        self.logger.debug("Combining results to identify relevant tokens")

        relevant_tokens = {
            'registers': [],
            'register_arrays': [],
            'sequences': [],
            'anomalies': anomalous_patterns
        }

        # Add important registers
        for reg in self.important_registers[:50]:  # Top 50 important registers
            relevant_tokens['registers'].append({
                'register': reg['register'],
                'importance': reg['score'],
                'reason': f"High TF-IDF score ({reg['score']:.4f}) - contains term '{reg['term']}'"
            })

        # Add registers involved in common access patterns
        for pattern in self.access_patterns:
            for reg in pattern['registers'][:20]:  # Limit to 20 registers per cluster
                if reg not in [r['register'] for r in relevant_tokens['registers']]:
                    relevant_tokens['registers'].append({
                        'register': reg,
                        'importance': 0.7,  # Default importance score
                        'reason': f"Part of access pattern cluster {pattern['cluster_id']}"
                    })

        # Add registers from common sequences
        for pattern in self.sequence_patterns:
            registers = pattern['sequence'].split('->')
            for reg in registers:
                if reg not in [r['register'] for r in relevant_tokens['registers']]:
                    relevant_tokens['registers'].append({
                        'register': reg,
                        'importance': pattern['count'] / 10,  # Scale by occurrence count
                        'reason': f"Part of common access sequence (occurs {pattern['count']} times)"
                    })

        # Add sequence patterns
        relevant_tokens['sequences'] = self.sequence_patterns

        # Find register arrays
        array_pattern = r'([^\[\]]+)(?:\[(\d+)\])+'
        for _, row in df.iterrows():
            if 'description' in row and row['description']:
                match = re.search(array_pattern, row['description'])
                if match:
                    array_name = match.group(1)
                    if array_name not in [a['array_name'] for a in relevant_tokens['register_arrays']]:
                        relevant_tokens['register_arrays'].append({
                            'array_name': array_name,
                            'example': row['description'],
                            'importance': 0.8,
                            'reason': "Register array detected"
                        })

        # Deduplicate and sort by importance
        for category in ['registers', 'register_arrays']:
            unique_items = {}
            for item in relevant_tokens[category]:
                key = item.get('register', item.get('array_name', ''))
                if key not in unique_items or item['importance'] > unique_items[key]['importance']:
                    unique_items[key] = item

            relevant_tokens[category] = list(unique_items.values())
            relevant_tokens[category].sort(key=lambda x: x['importance'], reverse=True)

        return relevant_tokens