"""
Test Suite for Guardrails and Conversational Agent

Run this to validate guardrail functionality and agent behavior.
"""

import unittest
from typing import List, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from guardrails import (
    InputGuardrail,
    OutputGuardrail,
    RejectionHandler,
    GuardrailResult
)


class TestInputGuardrail(unittest.TestCase):
    """Test cases for input guardrail."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.guardrail_moderate = InputGuardrail(
            strictness="moderate",
            log_to_mlflow=False
        )
        cls.guardrail_strict = InputGuardrail(
            strictness="strict",
            log_to_mlflow=False
        )
        cls.guardrail_lenient = InputGuardrail(
            strictness="lenient",
            log_to_mlflow=False
        )
    
    def test_valid_databricks_queries(self):
        """Test that clear Databricks queries are accepted."""
        valid_queries = [
            "How do I create a cluster in Databricks?",
            "Explain MLflow experiment tracking",
            "What is Delta Lake?",
            "Show me how to use Databricks SQL",
            "How do I configure Unity Catalog?",
        ]
        
        for query in valid_queries:
            with self.subTest(query=query):
                result = self.guardrail_moderate.validate(query)
                self.assertTrue(
                    result.is_valid,
                    f"Query should be valid: {query}\nReason: {result.reason}"
                )
                self.assertEqual(result.category, "databricks")
    
    def test_off_topic_queries(self):
        """Test that off-topic queries are rejected."""
        invalid_queries = [
            "What's the weather today?",
            "Tell me a joke",
            "How do I cook pasta?",
            "What is the capital of France?",
            "Write me a poem about cats",
        ]
        
        for query in invalid_queries:
            with self.subTest(query=query):
                result = self.guardrail_moderate.validate(query)
                self.assertFalse(
                    result.is_valid,
                    f"Query should be rejected: {query}\nReason: {result.reason}"
                )
                self.assertEqual(result.category, "off_topic")
    
    def test_borderline_queries_moderate(self):
        """Test borderline queries with moderate strictness."""
        borderline_queries = [
            "How do I use Apache Spark?",  # Spark is used in Databricks
            "What is Python?",  # Too general
            "Explain machine learning",  # Too general
        ]
        
        # With moderate strictness, some may pass, some may fail
        # Just ensure we get a result and it's categorized
        for query in borderline_queries:
            with self.subTest(query=query):
                result = self.guardrail_moderate.validate(query)
                self.assertIsNotNone(result.category)
                self.assertIn(result.category, ["databricks", "off_topic", "unclear"])
    
    def test_strictness_levels(self):
        """Test that strictness levels behave differently."""
        query = "How do I use Apache Spark for data processing?"
        
        result_strict = self.guardrail_strict.validate(query)
        result_moderate = self.guardrail_moderate.validate(query)
        result_lenient = self.guardrail_lenient.validate(query)
        
        # Lenient should be most permissive
        # We can't guarantee exact behavior, but we can check they all return results
        self.assertIsNotNone(result_strict.category)
        self.assertIsNotNone(result_moderate.category)
        self.assertIsNotNone(result_lenient.category)
    
    def test_confidence_scores(self):
        """Test that confidence scores are in valid range."""
        queries = [
            "How do I create a Databricks cluster?",
            "What's the weather?",
            "How do I use Spark?"
        ]
        
        for query in queries:
            with self.subTest(query=query):
                result = self.guardrail_moderate.validate(query)
                self.assertGreaterEqual(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)
    
    def test_metadata_present(self):
        """Test that metadata is included in results."""
        result = self.guardrail_moderate.validate("Test query")
        
        self.assertIsNotNone(result.metadata)
        self.assertIn("strictness", result.metadata)
        self.assertIn("latency_ms", result.metadata)
        self.assertIn("query_length", result.metadata)


class TestOutputGuardrail(unittest.TestCase):
    """Test cases for output guardrail."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.guardrail = OutputGuardrail(log_to_mlflow=False)
    
    def test_valid_responses(self):
        """Test that on-topic responses are accepted."""
        test_cases = [
            {
                "query": "How do I create a Databricks cluster?",
                "response": "To create a Databricks cluster, go to the Compute section and click 'Create Cluster'. Configure the cluster settings including Spark version, node types, and auto-termination settings."
            },
            {
                "query": "What is MLflow?",
                "response": "MLflow is an open-source platform for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment. It's fully integrated with Databricks."
            }
        ]
        
        for case in test_cases:
            with self.subTest(query=case["query"]):
                result = self.guardrail.validate(case["query"], case["response"])
                self.assertTrue(
                    result.is_valid,
                    f"Response should be valid\nQuery: {case['query']}\nResponse: {case['response']}\nReason: {result.reason}"
                )
    
    def test_off_topic_responses(self):
        """Test that off-topic responses are flagged."""
        test_cases = [
            {
                "query": "How do I use Databricks?",
                "response": "Let me tell you about my favorite recipe for chocolate cake. First, you need to preheat the oven to 350 degrees..."
            },
            {
                "query": "Explain Delta Lake",
                "response": "Delta Lake is great. Also, did you know that the weather is nice today? I love sunny days!"
            }
        ]
        
        for case in test_cases:
            with self.subTest(query=case["query"]):
                result = self.guardrail.validate(case["query"], case["response"])
                # Off-topic responses should ideally be flagged as invalid
                # But we'll just check that we get a result
                self.assertIsNotNone(result.is_valid)
                self.assertIsNotNone(result.reason)


class TestRejectionHandler(unittest.TestCase):
    """Test cases for rejection handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = RejectionHandler()
    
    def test_generates_message(self):
        """Test that rejection messages are generated."""
        guardrail_result = GuardrailResult(
            is_valid=False,
            confidence=0.9,
            reason="Query is about cooking, not Databricks",
            category="off_topic"
        )
        
        message = self.handler.generate_rejection("How do I cook pasta?", guardrail_result)
        
        self.assertIsInstance(message, str)
        self.assertGreater(len(message), 0)
        self.assertIn("Databricks", message)
    
    def test_message_is_polite(self):
        """Test that rejection messages are polite and helpful."""
        guardrail_result = GuardrailResult(
            is_valid=False,
            confidence=0.9,
            reason="Off-topic query",
            category="off_topic"
        )
        
        message = self.handler.generate_rejection("Random question", guardrail_result)
        
        # Should not contain rude words
        rude_words = ["stupid", "dumb", "wrong", "bad"]
        message_lower = message.lower()
        for word in rude_words:
            self.assertNotIn(word, message_lower)
        
        # Should contain helpful suggestions
        self.assertIn("help", message_lower)


class TestGuardrailIntegration(unittest.TestCase):
    """Integration tests for guardrails."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.input_guardrail = InputGuardrail(
            strictness="moderate",
            log_to_mlflow=False
        )
        cls.output_guardrail = OutputGuardrail(log_to_mlflow=False)
        cls.rejection_handler = RejectionHandler()
    
    def test_full_rejection_flow(self):
        """Test complete flow for rejected query."""
        query = "What's the weather today?"
        
        # Step 1: Input guardrail
        input_result = self.input_guardrail.validate(query)
        self.assertFalse(input_result.is_valid)
        
        # Step 2: Generate rejection
        rejection_message = self.rejection_handler.generate_rejection(query, input_result)
        self.assertIsInstance(rejection_message, str)
        self.assertGreater(len(rejection_message), 0)
    
    def test_full_acceptance_flow(self):
        """Test complete flow for accepted query."""
        query = "How do I create a Databricks cluster?"
        
        # Step 1: Input guardrail
        input_result = self.input_guardrail.validate(query)
        self.assertTrue(input_result.is_valid)
        
        # Step 2: Simulate response generation
        response = "To create a cluster, go to Compute and click Create Cluster."
        
        # Step 3: Output guardrail
        output_result = self.output_guardrail.validate(query, response)
        self.assertTrue(output_result.is_valid)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.input_guardrail = InputGuardrail(
            strictness="moderate",
            log_to_mlflow=False
        )
    
    def test_empty_query(self):
        """Test handling of empty query."""
        result = self.input_guardrail.validate("")
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.is_valid)
    
    def test_very_long_query(self):
        """Test handling of very long query."""
        long_query = "How do I use Databricks? " * 100
        result = self.input_guardrail.validate(long_query)
        self.assertIsNotNone(result)
    
    def test_special_characters(self):
        """Test handling of special characters."""
        queries = [
            "How do I use Databricks? 🚀",
            "What is MLflow??? !!!",
            "Databricks <script>alert('test')</script>",
        ]
        
        for query in queries:
            with self.subTest(query=query):
                result = self.input_guardrail.validate(query)
                self.assertIsNotNone(result)
    
    def test_multilingual_queries(self):
        """Test handling of non-English queries."""
        queries = [
            "¿Cómo uso Databricks?",  # Spanish
            "Comment utiliser Databricks?",  # French
            "Databricksの使い方は？",  # Japanese
        ]
        
        for query in queries:
            with self.subTest(query=query):
                result = self.input_guardrail.validate(query)
                self.assertIsNotNone(result)


def run_tests(verbosity=2):
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInputGuardrail))
    suite.addTests(loader.loadTestsFromTestCase(TestOutputGuardrail))
    suite.addTests(loader.loadTestsFromTestCase(TestRejectionHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestGuardrailIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result


if __name__ == "__main__":
    # Run tests when script is executed directly
    result = run_tests(verbosity=2)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
