import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

def ramp_extract_features(tokenizer_output: Dict[str, Any], 
                          passing_tests_dir: str,
                          output_dir: Optional[str] = None,
                          feature_types: Optional[List[str]] = None,
                          logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Extract features from passing tests using the tokenizer output dictionary.
    
    Parameters:
    -----------
    tokenizer_output : Dict[str, Any]
        Dictionary output from running the tokenizer
    passing_tests_dir : str
        Directory containing passing test logs
    output_dir : str, optional
        Directory to save extracted features and models
    feature_types : List[str], optional
        List of feature types to extract (None for all)
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing extracted features for each test
    """
    # Setup logger if not provided
    if logger is None:
        logger = logging.getLogger('ramp_extract_features')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.info(f"Starting feature extraction from passing tests in {passing_tests_dir}")
    
    # Initialize feature extractor
    from AIFeatureExtractor import AIFeatureExtractor  # Import locally to avoid import issues
    feature_extractor = AIFeatureExtractor(logger=logger)
    
    # Process passing tests directory
    test_files = {}
    try:
        # Get list of all test files in the directory
        path = Path(passing_tests_dir)
        test_files = {
            test_file.stem: str(test_file) 
            for test_file in path.glob("**/*.log*")
            if test_file.is_file()
        }
        
        logger.info(f"Found {len(test_files)} test files in {passing_tests_dir}")
    except Exception as e:
        logger.error(f"Error processing passing tests directory: {e}")
        return {}
    
    # Map tokenizer output to tests
    tests_with_tokens = {}
    for test_id, test_data in tokenizer_output.items():
        if test_id in test_files:
            tests_with_tokens[test_id] = {
                'tokens': test_data,
                'status': 'pass',  # These are from passing_tests_dir so marked as pass
                'file_path': test_files[test_id]
            }
    
    logger.info(f"Matched {len(tests_with_tokens)} tests with tokenizer output")
    
    # Extract features
    tests_with_features = feature_extractor.extract_features(
        tests_with_tokens, 
        feature_types=feature_types
    )
    
    # Save models and features if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save features
        features_file = os.path.join(output_dir, 'extracted_features.json')
        try:
            # Convert to serializable format
            serializable_features = {}
            for test_id, test_data in tests_with_features.items():
                serializable_features[test_id] = {
                    'features': test_data.get('features', {}),
                    'status': test_data.get('status', 'unknown')
                }
                
            with open(features_file, 'w') as f:
                json.dump(serializable_features, f, indent=2)
            
            logger.info(f"Saved extracted features to {features_file}")
        except Exception as e:
            logger.error(f"Error saving features: {e}")
        
        # Save models
        models_dir = os.path.join(output_dir, 'models')
        feature_extractor.save_models(models_dir)
    
    return tests_with_features


# Example usage:
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('feature_extraction')
    
    # Example parameters
    tokenizer_output = {}  # This would be loaded from the tokenizer run
    passing_tests_dir = "path/to/passing/tests"
    output_dir = "path/to/output"
    
    # Extract features
    result = ramp_extract_features(
        tokenizer_output=tokenizer_output,
        passing_tests_dir=passing_tests_dir,
        output_dir=output_dir,
        logger=logger
    )
    
    print(f"Extracted features for {len(result)} tests")