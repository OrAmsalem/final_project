# Complete extract_temporal_patterns.py

import numpy as np
import pandas as pd
import logging

def extract_temporal_patterns(self, test_records):
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

        # Prepare register dataframe with time information
        register_df = self.prepare_register_dataframe(tokens['registers'])
        
        # Skip if no data
        if len(register_df) == 0:
            features[test_id] = {}
            continue
        
        # Create time bins and count accesses
        binned_data = self.create_time_bins(register_df)
        
        # Calculate temporal activity features
        temporal_features = self.calculate_temporal_features(register_df, binned_data)
        
        features[test_id] = temporal_features

    return features

def prepare_register_dataframe(self, register_tokens):
    """
    Prepare register dataframe with time information
    
    Parameters:
    register_tokens (list): List of register tokens
    
    Returns:
    pandas.DataFrame: Register dataframe with time information
    """
    # Convert to dataframe
    register_df = pd.DataFrame(register_tokens)
    
    # Skip if empty
    if len(register_df) == 0:
        return register_df
        
    # Ensure timestamp is numeric
    register_df['timestamp'] = register_df['timestamp'].astype(int)
    
    # Convert timestamps to relative time (seconds from start)
    min_ts = register_df['timestamp'].min()
    register_df['relative_time'] = (register_df['timestamp'] - min_ts) / 1e9  # Assuming ns
    
    return register_df

def create_time_bins(self, register_df):
    """
    Create time bins and count accesses
    
    Parameters:
    register_df (pandas.DataFrame): Register dataframe with time information
    
    Returns:
    dict: Time bin information and counts
    """
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
    
    return {
        'n_bins': n_bins,
        'read_counts': read_counts,
        'write_counts': write_counts
    }

def calculate_temporal_features(self, register_df, binned_data):
    """
    Calculate temporal activity features
    
    Parameters:
    register_df (pandas.DataFrame): Register dataframe with time information
    binned_data (dict): Time bin information and counts
    
    Returns:
    dict: Temporal activity features
    """
    # Extract bin data
    read_counts = binned_data['read_counts']
    write_counts = binned_data['write_counts']
    
    # Calculate test duration
    min_ts = register_df['timestamp'].min()
    max_ts = register_df['timestamp'].max()
    test_duration = (max_ts - min_ts) / 1e9  # seconds
    
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
    phase_features = self.identify_activity_phases(read_counts, write_counts)
    temporal_features['phase_analysis'] = phase_features

    # Calculate autocorrelation to identify periodic patterns
    if len(read_counts) > 10:
        temporal_features['periodicity'] = self.calculate_periodicity(read_counts, write_counts)
    
    return temporal_features

def identify_activity_phases(self, read_counts, write_counts):
    """
    Identify activity phases in time bins
    
    Parameters:
    read_counts (pandas.Series): Read counts by time bin
    write_counts (pandas.Series): Write counts by time bin
    
    Returns:
    dict: Phase analysis results
    """
    total_counts = read_counts + write_counts
    
    # Simple phase detection: identify periods of high vs low activity
    threshold = total_counts.mean() + total_counts.std()
    
    phases = []
    current_phase = None
    phase_start = 0
    
    for i, count in enumerate(total_counts):
        if count > threshold:
            if current_phase != 'high':
                if current_phase is not None:
                    phases.append({
                        'type': current_phase,
                        'start': phase_start,
                        'end': i - 1,
                        'duration': i - phase_start
                    })
                current_phase = 'high'
                phase_start = i
        else:
            if current_phase != 'low':
                if current_phase is not None:
                    phases.append({
                        'type': current_phase,
                        'start': phase_start,
                        'end': i - 1,
                        'duration': i - phase_start
                    })
                current_phase = 'low'
                phase_start = i
    
    # Add final phase
    if current_phase is not None:
        phases.append({
            'type': current_phase,
            'start': phase_start,
            'end': len(total_counts) - 1,
            'duration': len(total_counts) - phase_start
        })
    
    return {
        'phases': phases,
        'num_phases': len(phases),
        'high_activity_ratio': sum(1 for p in phases if p['type'] == 'high') / len(phases) if phases else 0
    }

def calculate_periodicity(self, read_counts, write_counts):
    """
    Calculate autocorrelation to identify periodic patterns
    
    Parameters:
    read_counts (pandas.Series): Read counts by time bin
    write_counts (pandas.Series): Write counts by time bin
    
    Returns:
    dict: Periodicity features
    """
    read_autocorr = self.calculate_autocorrelation(read_counts.values, 10)
    write_autocorr = self.calculate_autocorrelation(write_counts.values, 10)
    
    return {
        'read_autocorrelation': read_autocorr.tolist(),
        'write_autocorrelation': write_autocorr.tolist(),
        'has_periodicity': np.max(np.abs(read_autocorr[1:])) > 0.5 or np.max(
            np.abs(write_autocorr[1:])) > 0.5
    }

def calculate_autocorrelation(self, data, max_lag):
    """
    Calculate autocorrelation for a time series
    
    Parameters:
    data (numpy.ndarray): Time series data
    max_lag (int): Maximum lag for autocorrelation
    
    Returns:
    numpy.ndarray: Autocorrelation values
    """
    # Normalize data
    data = data - np.mean(data)
    if np.std(data) > 0:
        data = data / np.std(data)
    
    # Calculate autocorrelation
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[len(data)-1:]
    
    # Normalize and limit to max_lag
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    autocorr = autocorr[:min(max_lag+1, len(autocorr))]
    
    return autocorr
