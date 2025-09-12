# Complete extract_fsm_transition.py

import numpy as np
import logging
from collections import defaultdict, Counter

def extract_fsm_transition_features(self, test_records):
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
        fsm_transitions = self.group_fsm_transitions(tokens['fsm_states'])
        
        # Calculate FSM transition statistics
        fsm_features = self.calculate_fsm_statistics(fsm_transitions)
        
        features[test_id] = fsm_features

    return features

def group_fsm_transitions(self, fsm_tokens):
    """
    Group FSM transitions by FSM name
    
    Parameters:
    fsm_tokens (list): List of FSM state tokens
    
    Returns:
    dict: Dictionary mapping FSM names to transitions
    """
    # Specifically handling the FSM patterns from the tokenizer:
    # - Sleep State FSM transition: from_state -> to_state
    # - Generic FSM state transitions: FSM fsm_name: from_state -> to_state
    fsm_transitions = defaultdict(list)
    
    for token in fsm_tokens:
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
    
    return fsm_transitions

def calculate_fsm_statistics(self, fsm_transitions):
    """
    Calculate FSM transition statistics
    
    Parameters:
    fsm_transitions (dict): Dictionary mapping FSM names to transitions
    
    Returns:
    dict: Dictionary of FSM statistics
    """
    fsm_features = {}
    
    for fsm_name, transitions in fsm_transitions.items():
        # Skip if not enough transitions
        if len(transitions) < 2:
            continue

        # Sort transitions by timestamp
        sorted_transitions = sorted(transitions, key=lambda x: x['timestamp'])
        
        # Calculate state and transition statistics
        state_stats = self.calculate_state_statistics(sorted_transitions)
        
        # Identify cyclic behaviors
        cycle_stats = self.identify_cycles(state_stats['transition_matrix'], state_stats['unique_states'])
        
        # Combine statistics
        fsm_stats = {**state_stats, **cycle_stats}
        fsm_features[fsm_name] = fsm_stats
    
    return fsm_features

def calculate_state_statistics(self, sorted_transitions):
    """
    Calculate state and transition statistics
    
    Parameters:
    sorted_transitions (list): List of transitions sorted by timestamp
    
    Returns:
    dict: State statistics
    """
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
    stats = {
        'unique_states': unique_states,
        'state_counts': dict(state_counts),
        'transition_counts': {f"{from_}->{to_}": count for (from_, to_), count in
                              transition_counts.items()},
        'total_transitions': len(sorted_transitions),
        'unique_transitions': len(transition_counts),
        'num_states': len(unique_states),
        'start_state': sorted_transitions[0]['from_state'],
        'end_state': sorted_transitions[-1]['to_state'],
        'transition_entropy': self.calculate_entropy(transition_counts.values()),
        'transition_matrix': transition_matrix
    }
    
    return stats

def identify_cycles(self, transition_matrix, unique_states):
    """
    Identify cyclic behaviors in FSM transitions
    
    Parameters:
    transition_matrix (numpy.ndarray): State transition matrix
    unique_states (list): List of unique states
    
    Returns:
    dict: Cycle statistics
    """
    # Simple cycle detection: look for states that transition back to themselves
    self_transitions = []
    for i, state in enumerate(unique_states):
        if transition_matrix[i, i] > 0:
            self_transitions.append((state, int(transition_matrix[i, i])))
    
    # Look for simple cycles (2-state cycles)
    two_state_cycles = []
    for i in range(len(unique_states)):
        for j in range(i + 1, len(unique_states)):
            if transition_matrix[i, j] > 0 and transition_matrix[j, i] > 0:
                two_state_cycles.append((unique_states[i], unique_states[j]))
    
    return {
        'self_transitions': self_transitions,
        'two_state_cycles': two_state_cycles,
        'has_cycles': len(self_transitions) > 0 or len(two_state_cycles) > 0
    }

def calculate_entropy(self, values):
    """
    Calculate entropy for a collection of values
    
    Parameters:
    values (iterable): Collection of values
    
    Returns:
    float: Entropy value
    """
    # Convert to list and calculate probabilities
    values = list(values)
    total = sum(values)
    
    if total == 0:
        return 0.0
    
    # Calculate entropy
    entropy = 0
    for value in values:
        if value > 0:
            p = value / total
            entropy -= p * np.log2(p)
    
    return entropy


