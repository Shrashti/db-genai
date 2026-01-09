"""
Test Suite for RAG Evaluation Framework

Tests for ground truth generation, RAGAS metrics, and tool calling evaluation.
"""

import pytest
import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ground_truth_generator import GroundTruthGenerator, EvaluationSample
from ragas_evaluator import RAGASEvaluator, EvaluationResults


class TestGroundTruthGenerator:
    """Tests for ground truth data generation."""
    
    @pytest.fixture
    def sample_csv_path(self, tmp_path):
        """Create a temporary CSV file for testing."""
        csv_content = """id\turl\tcontent
1\thttps://docs.databricks.com/test1\tDatabricks is a unified analytics platform. It provides collaborative notebooks and scalable compute.
2\thttps://docs.databricks.com/test2\tTo create a cluster, go to the Compute page and click Create Cluster. Configure the cluster settings.
3\thttps://docs.databricks.com/test3\tUnity Catalog is a unified governance solution for data and AI assets on Databricks."""
        
        csv_file = tmp_path / "test_docs.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)
    
    def test_generator_initialization(self, sample_csv_path):
        """Test that generator initializes correctly."""
        generator = GroundTruthGenerator(
            csv_path=sample_csv_path,
            llm_endpoint="databricks-qwen3-next-80b-a3b-instruct"
        )
        
        assert generator is not None
        assert len(generator.docs_df) == 3
        assert 'id' in generator.docs_df.columns
        assert 'url' in generator.docs_df.columns
        assert 'content' in generator.docs_df.columns
    
    def test_sample_documents_random(self, sample_csv_path):
        """Test random document sampling."""
        generator = GroundTruthGenerator(csv_path=sample_csv_path)
        sampled = generator.sample_documents(num_samples=2, strategy="random")
        
        assert len(sampled) == 2
        assert all(col in sampled.columns for col in ['id', 'url', 'content'])
    
    def test_sample_documents_diverse(self, sample_csv_path):
        """Test diverse document sampling."""
        generator = GroundTruthGenerator(csv_path=sample_csv_path)
        sampled = generator.sample_documents(num_samples=2, strategy="diverse")
        
        assert len(sampled) <= 2  # May be less if not enough diverse samples
        assert all(col in sampled.columns for col in ['id', 'url', 'content'])
    
    def test_question_classification(self, sample_csv_path):
        """Test question type classification."""
        generator = GroundTruthGenerator(csv_path=sample_csv_path)
        
        assert generator.classify_question_type("How do I create a cluster?") == "how-to"
        assert generator.classify_question_type("What is Unity Catalog?") == "conceptual"
        assert generator.classify_question_type("Show me an API example") == "api"
        assert generator.classify_question_type("Configure SSO settings") == "configuration"
        assert generator.classify_question_type("What are the benefits?") == "factual"
    
    def test_difficulty_estimation(self, sample_csv_path):
        """Test difficulty estimation."""
        generator = GroundTruthGenerator(csv_path=sample_csv_path)
        
        short_q = "What is Databricks?"
        short_doc = "Databricks is a platform."
        assert generator.estimate_difficulty(short_q, short_doc) == "easy"
        
        long_q = "How do I configure advanced cluster settings with custom init scripts?"
        long_doc = "A" * 1000  # Long documentation
        assert generator.estimate_difficulty(long_q, long_doc) in ["medium", "hard"]
    
    def test_evaluation_sample_structure(self):
        """Test EvaluationSample dataclass structure."""
        sample = EvaluationSample(
            user_input="Test question?",
            retrieved_contexts=["Context 1"],
            response="Test response",
            reference="Test reference",
            doc_url="https://test.com",
            doc_id="123",
            question_type="factual",
            difficulty="easy"
        )
        
        assert sample.user_input == "Test question?"
        assert len(sample.retrieved_contexts) == 1
        assert sample.question_type == "factual"
        assert sample.difficulty == "easy"
    
    def test_export_to_json(self, sample_csv_path, tmp_path):
        """Test JSON export functionality."""
        generator = GroundTruthGenerator(csv_path=sample_csv_path)
        
        samples = [
            EvaluationSample(
                user_input="Q1?",
                retrieved_contexts=["C1"],
                response="R1",
                reference="Ref1",
                doc_url="url1",
                doc_id="1",
                question_type="factual",
                difficulty="easy"
            )
        ]
        
        output_path = tmp_path / "test_output.json"
        generator.export_to_json(samples, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]['user_input'] == "Q1?"
    
    def test_export_to_csv(self, sample_csv_path, tmp_path):
        """Test CSV export functionality."""
        generator = GroundTruthGenerator(csv_path=sample_csv_path)
        
        samples = [
            EvaluationSample(
                user_input="Q1?",
                retrieved_contexts=["C1"],
                response="R1",
                reference="Ref1",
                doc_url="url1",
                doc_id="1",
                question_type="factual",
                difficulty="easy"
            )
        ]
        
        output_path = tmp_path / "test_output.csv"
        generator.export_to_csv(samples, str(output_path))
        
        assert output_path.exists()
        
        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df.iloc[0]['user_input'] == "Q1?"


class TestRAGASEvaluator:
    """Tests for RAGAS evaluation."""
    
    @pytest.fixture
    def sample_evaluation_data(self):
        """Create sample evaluation data."""
        return [
            {
                "user_input": "What is Databricks?",
                "retrieved_contexts": ["Databricks is a unified analytics platform."],
                "response": "Databricks is a unified analytics platform for data and AI.",
                "reference": "Databricks is a unified analytics platform.",
                "question_type": "conceptual"
            },
            {
                "user_input": "How do I create a cluster?",
                "retrieved_contexts": ["To create a cluster, go to Compute page."],
                "response": "Go to the Compute page and click Create Cluster.",
                "reference": "Navigate to Compute page and create cluster.",
                "question_type": "how-to"
            }
        ]
    
    def test_evaluator_initialization(self):
        """Test that evaluator initializes correctly."""
        evaluator = RAGASEvaluator(
            llm_endpoint="databricks-qwen3-next-80b-a3b-instruct",
            log_to_mlflow=False
        )
        
        assert evaluator is not None
        assert len(evaluator.metrics) == 5  # 5 RAGAS metrics
    
    def test_prepare_ragas_dataset(self, sample_evaluation_data):
        """Test RAGAS dataset preparation."""
        evaluator = RAGASEvaluator(log_to_mlflow=False)
        dataset = evaluator.prepare_ragas_dataset(sample_evaluation_data)
        
        assert dataset is not None
        assert len(dataset.samples) == 2
        assert dataset.samples[0].user_input == "What is Databricks?"
    
    def test_evaluation_results_structure(self):
        """Test EvaluationResults dataclass structure."""
        results = EvaluationResults(
            faithfulness_score=0.85,
            answer_relevancy_score=0.90,
            context_precision_score=0.80,
            context_recall_score=0.75,
            answer_correctness_score=0.88,
            average_score=0.836,
            num_samples=10,
            timestamp="2024-01-01T00:00:00",
            detailed_results=pd.DataFrame()
        )
        
        assert results.faithfulness_score == 0.85
        assert results.average_score == 0.836
        assert results.num_samples == 10


class TestToolCallingEvaluation:
    """Tests for tool calling evaluation."""
    
    def test_tool_call_tracking(self):
        """Test that tool calls are properly tracked."""
        # Mock agent result
        agent_result = {
            "response": "Test response",
            "metadata": {
                "tool_calls": 2,
                "tools_used": ["generic_doc_retriever", "api_docs_retriever"],
                "latency_ms": 1500
            },
            "was_rejected": False
        }
        
        assert agent_result["metadata"]["tool_calls"] == 2
        assert len(agent_result["metadata"]["tools_used"]) == 2
        assert "generic_doc_retriever" in agent_result["metadata"]["tools_used"]
    
    def test_tool_selection_accuracy(self):
        """Test tool selection matches expected tools."""
        test_cases = [
            {
                "question_type": "how-to",
                "expected_tools": ["tutorial_retriever", "generic_doc_retriever"],
                "actual_tools": ["tutorial_retriever"]
            },
            {
                "question_type": "api",
                "expected_tools": ["api_docs_retriever"],
                "actual_tools": ["api_docs_retriever"]
            }
        ]
        
        for case in test_cases:
            # Check if at least one expected tool was used
            has_match = any(
                tool in case["expected_tools"] 
                for tool in case["actual_tools"]
            )
            assert has_match or len(case["actual_tools"]) > 0


class TestIntegration:
    """Integration tests for the complete evaluation pipeline."""
    
    def test_sample_data_loading(self):
        """Test loading sample evaluation data."""
        sample_data_path = Path(__file__).parent.parent / "ground-truth data" / "sample_evaluation_data.json"
        
        if sample_data_path.exists():
            with open(sample_data_path) as f:
                data = json.load(f)
            
            assert len(data) > 0
            assert "user_input" in data[0]
            assert "reference" in data[0]
            assert "question_type" in data[0]
    
    def test_evaluation_data_format(self):
        """Test that evaluation data has required RAGAS format."""
        required_fields = ["user_input", "retrieved_contexts", "response", "reference"]
        
        sample = {
            "user_input": "Test question?",
            "retrieved_contexts": ["Context"],
            "response": "Answer",
            "reference": "Reference answer"
        }
        
        for field in required_fields:
            assert field in sample


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
