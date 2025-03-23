#!/usr/bin/env python3
"""
AI Debug Assist - Main entry point

This module orchestrates the AI debug assist pipeline for hardware pre-silicon validation.
It coordinates log collection, tokenization, feature extraction, anomaly detection, and reporting.
"""
import os
import sys
import time
import argparse
import logging
import json
from datetime import datetime

# Import core modules
from utils.logger import setup_logger
from core.collector import LogCollector
from core.tokenizer import LogTokenizer
from core.feature_extractor import FeatureExtractor
from core.behavior_learner import BehaviorLearner
from core.anomaly_detector import AnomalyDetector
from core.analyzer import AnomalyAnalyzer
from reporting.report_generator import ReportGenerator


class AIDebugAssist:
    def __init__(self, regression_directory, output_directory, config=None):
        """
        Initialize the AI Debug Assist tool

        Parameters:
        regression_directory (str): Path to the directory containing test directories
        output_directory (str): Path to store output reports
        config (dict): Configuration settings (optional)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Setup logger
        self.logger = setup_logger("AI_Debug_Assist", output_directory)
        self.logger.info("Initializing AI Debug Assist")

        # Store configuration
        self.regression_directory = regression_directory
        self.output_directory = output_directory
        self.config = config or {}

        # Determine whether to use AI for tokenization
        self.use_ai_tokenizer = self.config.get('use_ai_tokenizer', True)

        # Initialize components
        self.collector = LogCollector(regression_directory, logger=self.logger)
        self.tokenizer = LogTokenizer(use_ai=self.use_ai_tokenizer, logger=self.logger)
        self.feature_extractor = FeatureExtractor(logger=self.logger)
        self.behavior_learner = BehaviorLearner(logger=self.logger)
        self.anomaly_detector = AnomalyDetector(logger=self.logger)
        self.analyzer = AnomalyAnalyzer(logger=self.logger)
        self.report_generator = ReportGenerator(output_directory, logger=self.logger)

        # Initialize data storage
        self.tests = {}
        self.passing_tests = {}
        self.failing_tests = {}
        self.anomalies = {}
        self.buckets = {}

    def run(self):
        """
        Run the complete AI debug assist pipeline

        Returns:
        str: Path to the generated reports directory
        """
        start_time = time.time()
        self.logger.info("Starting AI Debug Assist pipeline")

        # Step 1: Collect logs
        self.logger.info("Step 1: Collecting logs")
        self.tests, self.passing_tests, self.failing_tests = self.collector.collect_logs()

        if not self.failing_tests:
            self.logger.warning("No failing tests found. Nothing to debug.")
            return None

        # Step 2: Tokenize logs
        self.logger.info(f"Step 2: Tokenizing logs (AI-enhanced: {self.use_ai_tokenizer})")
        self.passing_tests = self.tokenizer.tokenize_logs(self.passing_tests)
        self.failing_tests = self.tokenizer.tokenize_logs(self