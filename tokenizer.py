"""
AI-enhanced log tokenization functionality for ptracker.log.gz
"""
import re
import gzip
import logging
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import AI token extractor
from models.ai_token_extractor import AITokenExtractor


class LogTokenizer:
    def __init__(self, use_ai=True, logger=None):
        """
        Initialize the log tokenizer

        Parameters:
        use_ai (bool): Whether to use AI for enhanced tokenization
        logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.use_ai = use_ai

        # Initialize AI token extractor if enabled
        if self.use_ai:
            self.ai_extractor = AITokenExtractor(logger=self.logger)

        # Pattern for ptracker.log format
        # [timestamp] action [address] size description value [comment]
        self.ptracker_pattern = r'\[(\d+)\]\s+(MEM_[RW])\s+\[(0x[0-9a-fA-F]+)\]\s+(\d+b)\s+([^\s].*?)\s+(0x[0-9a-fA-F]+)(?:\s+<--\s+(.+))?'

        # Additional patterns for other log entries like FSM transitions, etc.
        self.additional_patterns = [
            # Sleep state FSM transition
            r'Sleep\s+State\s+FSM\s+transition:\s+([\w\d]+)\s+->\s+([\w\d]+)',
            # Generic FSM state transitions
            r'FSM\s+([\w\d]+):\s+([\w\d]+)\s+->\s+([\w\d]+)',
            # Error messages
            r'ERROR\s*:\s*(.+)'
        ]

    def tokenize_logs(self, tests):
        """
        Tokenize log files for a collection of tests

        Parameters:
        tests (dict): Dictionary of test data

        Returns:
        dict: Updated test data with tokenized logs
        """
        self.logger.info(f"Tokenizing logs for {len(tests)} tests")

        for test_id, test_data in tests.items():
            # Read the log file if needed
            if not test_data.get('log_data'):
                log_data = self._read_log_file(test_data['log_path'])
                if log_data:
                    test_data['log_data'] = log_data
                else:
                    # Skip this test if log can't be read
                    self.logger.warning(f"Skipping tokenization for test {test_id}: unable to read log")
                    continue

            # Extract tokens from log content
            if 'log_data' in test_data and 'content' in test_data['log_data']:
                # Extract tokens using pattern matching
                tokens = self._extract_tokens(test_data['log_data']['content'])

                # If AI tokenization is enabled, enhance the tokens with AI-based analysis
                if self.use_ai:
                    # Create parsed logs for AI analysis
                    parsed_logs = self._create_parsed_logs(tokens)

                    # Extract additional tokens using AI
                    ai_tokens = self.ai_extractor.extract_tokens(parsed_logs)

                    # Enhance tokens with AI-derived information
                    tokens = self._enhance_tokens_with_ai(tokens, ai_tokens)

                test_data['tokens'] = tokens

        self.logger.info("Log tokenization complete")
        return tests

    def _read_log_file(self, log_path):
        """
        Read and parse a gzipped log file

        Parameters:
        log_path (str): Path to the gzipped log file

        Returns:
        dict: Raw log content and metadata
        """
        try:
            self.logger.debug(f"Reading log file: {log_path}")
            with gzip.open(log_path, 'rt', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            return {
                'content': content,
                'file_path': log_path
            }
        except Exception as e:
            self.logger.error(f"Error reading log file {log_path}: {e}")
            return None

    def _extract_tokens(self, content):
        """
        Extract tokens from log content based on patterns

        Parameters:
        content (str): Log file content

        Returns:
        dict: Extracted tokens organized by type
        """
        tokens = defaultdict(list)

        # Extract memory operations from ptracker log format
        for match in re.finditer(self.ptracker_pattern, content):
            timestamp, action, address, size, description, value, comment = match.groups()

            # Default to empty string if comment is None
            comment = comment or ""

            token_data = {
                'timestamp': timestamp,
                'action': action,
                'address': address,
                'size': size,
                'description': description.strip(),
                'value': value,
                'comment': comment.strip(),
                'position': match.start(),
                'raw': match.group(0)
            }

            # Categorize by read/write
            if action == "MEM_R":
                tokens['memory_reads'].append(token_data)
            elif action == "MEM_W":
                tokens['memory_writes'].append(token_data)

            # Also categorize by register access paths
            register_path = description.strip()
            tokens['registers'].append(token_data)

            # Track register arrays separately
            if '[' in register_path and ']' in register_path:
                tokens['register_arrays'].append(token_data)

            # Special categories based on register naming patterns
            if 'fsm' in register_path.lower() or 'state' in register_path.lower():
                tokens['fsm_related'].append(token_data)

            if 'status' in register_path.lower() or 'error' in register_path.lower():
                tokens['status_registers'].append(token_data)

        # Extract other patterns
        for pattern in self.additional_patterns:
            for match in re.finditer(pattern, content):
                token_data = {
                    'match': match.groups(),
                    'position': match.start(),
                    'raw': match.group(0)
                }

                # Find the nearest timestamp before this token
                timestamp_matches = list(re.finditer(r'\[(\d+)\]', content[:match.start()]))
                if timestamp_matches:
                    token_data['timestamp'] = timestamp_matches[-1].group(1)
                else:
                    token_data['timestamp'] = "0"

                # Categorize based on the pattern
                if "Sleep" in pattern or "FSM" in pattern:
                    tokens['fsm_states'].append(token_data)
                elif "ERROR" in pattern:
                    tokens['errors'].append(token_data)

        return tokens

    def _create_parsed_logs(self, tokens):
        """
        Create a list of parsed log entries for AI analysis

        Parameters:
        tokens (dict): Tokens extracted from the log

        Returns:
        list: List of parsed log entries
        """
        parsed_logs = []

        # Add register tokens
        for token in tokens.get('registers', []):
            parsed_logs.append(token)

        # Add other relevant tokens
        for token in tokens.get('fsm_states', []):
            if 'match' in token and token['match']:
                # Convert FSM state tokens to a standardized format
                if len(token['match']) == 3:  # Generic FSM format
                    fsm_name, from_state, to_state = token['match']
                elif len(token['match']) == 2:  # Sleep state FSM format
                    fsm_name = "SleepStateFSM"
                    from_state, to_state = token['match']
                else:
                    continue

                parsed_logs.append({
                    'timestamp': token.get('timestamp', '0'),
                    'action': 'FSM_TRANSITION',
                    'description': f"{fsm_name}_transition",
                    'value': f"{from_state}->{to_state}",
                    'from_state': from_state,
                    'to_state': to_state,
                    'fsm_name': fsm_name
                })

        return parsed_logs

    def _enhance_tokens_with_ai(self, tokens, ai_tokens):
        """
        Enhance tokens with AI-derived information

        Parameters:
        tokens (dict): Tokens extracted using pattern matching
        ai_tokens (dict): Tokens and patterns extracted using AI

        Returns:
        dict: Enhanced tokens
        """
        # Skip if AI tokens is empty
        if not ai_tokens:
            return tokens

        # Add AI-derived tokens
        tokens['ai_important_registers'] = ai_tokens.get('registers', [])
        tokens['ai_register_arrays'] = ai_tokens.get('register_arrays', [])
        tokens['ai_sequences'] = ai_tokens.get('sequences', [])
        tokens['ai_anomalies'] = ai_tokens.get('anomalies', [])

        # Tag important registers in the original token list
        important_registers = {reg['register']: reg['importance']
                               for reg in ai_tokens.get('registers', [])}

        for category in ['registers', 'memory_reads', 'memory_writes']:
            for token in tokens.get(category, []):
                if token['description'] in important_registers:
                    token['ai_importance'] = important_registers[token['description']]

        return tokens