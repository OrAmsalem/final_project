# Complete extract_error_context.py

import numpy as np
import logging
from collections import Counter

def extract_error_context_features(self, test_records):
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
        
        # Extract error messages
        error_messages = extract_error_messages(tokens)
        
        # Skip if no error messages
        if not error_messages:
            features[test_id] = {'error_count': 0}
            continue
        
        # Sort errors by timestamp
        sorted_errors = sorted(error_messages, key=lambda x: x['timestamp'])
        
        # Find register accesses around each error
        error_contexts = self.find_error_contexts(sorted_errors, tokens)
        
        # Calculate error context features
        error_features = self.calculate_error_features(sorted_errors, error_contexts)
        
        features[test_id] = error_features

    return features

def extract_error_messages (tokens):
    """
    Extract error messages from tokens
    
    Parameters:
    tokens (dict): Tokenized log data
    
    Returns:
    list: List of error messages
    """
    # Skip if no error tokens
    if 'errors' not in tokens or not tokens['errors']:
        return []
        
    # Extract error messages
    error_messages = []
    for token in tokens['errors']:
        if 'match' in token and token['match']:
            error_messages.append({
                'message': token['match'][0],
                'timestamp': int(token.get('timestamp', 0)),
                'raw': token.get('raw', '')
            })
            
    return error_messages

def find_error_contexts(self, sorted_errors, tokens):
    """
    Find register accesses around each error
    
    Parameters:
    sorted_errors (list): List of error messages sorted by timestamp
    tokens (dict): Tokenized log data
    
    Returns:
    list: List of error contexts
    """
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
        
    return error_contexts

def calculate_error_features(self, sorted_errors, error_contexts):
    """
    Calculate error context features
    
    Parameters:
    sorted_errors (list): List of error messages sorted by timestamp
    error_contexts (list): List of error contexts
    
    Returns:
    dict: Error context features
    """
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
    common_terms = self.extract_common_terms(error_text)
    error_features['common_terms'] = common_terms
    
    return error_features

def extract_common_terms(self, text):
    """
    Extract common terms from error text
    
    Parameters:
    text (str): Error text
    
    Returns:
    list: Common terms
    """
    # Simple term extraction: split by whitespace and count
    words = text.lower().split()
    word_counts = Counter(words)
    
    # Filter out very common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    filtered_counts = {word: count for word, count in word_counts.items() 
                      if word not in stop_words and len(word) > 2}
    
    # Get top terms
    common_terms = [term for term, count in 
                   sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    return common_terms
