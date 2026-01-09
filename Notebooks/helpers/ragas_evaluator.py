"""
RAGAS Evaluator for RAG System Evaluation

This module implements RAGAS (RAG Assessment) metrics to evaluate the quality
of the Databricks documentation RAG system.

Metrics implemented:
- Faithfulness: Is the response grounded in retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are relevant contexts ranked higher?
- Context Recall: Were all relevant contexts retrieved?
- Factual Correctness: Does the answer match the reference?
"""

import pandas as pd
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import mlflow
from datetime import datetime

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset, SingleTurnSample
from databricks_langchain import ChatDatabricks


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    context_recall_score: float
    answer_correctness_score: float
    average_score: float
    num_samples: int
    timestamp: str
    detailed_results: pd.DataFrame


class RAGASEvaluator:
    """
    Evaluates RAG system using RAGAS metrics.
    
    Provides comprehensive evaluation of retrieval quality, answer quality,
    and overall system performance.
    """
    
    def __init__(
        self,
        llm_endpoint: str = "databricks-qwen3-next-80b-a3b-instruct",
        log_to_mlflow: bool = True
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            llm_endpoint: Databricks LLM endpoint for evaluation
            log_to_mlflow: Whether to log results to MLflow
        """
        self.llm_endpoint = llm_endpoint
        self.log_to_mlflow = log_to_mlflow
        
        print(f"🔧 Initializing RAGAS Evaluator...")
        print(f"   LLM Endpoint: {llm_endpoint}")
        
        # Initialize LLM for RAGAS
        llm = ChatDatabricks(endpoint=llm_endpoint)
        self.evaluator_llm = LangchainLLMWrapper(llm)
        
        # Configure RAGAS metrics with the LLM
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ]
        
        print(f"✅ RAGAS Evaluator initialized with {len(self.metrics)} metrics")
    
    def prepare_ragas_dataset(
        self,
        evaluation_data: List[Dict[str, Any]]
    ) -> EvaluationDataset:
        """
        Convert evaluation data to RAGAS EvaluationDataset format.
        
        Args:
            evaluation_data: List of dicts with keys:
                - user_input: The question
                - retrieved_contexts: List of retrieved context strings
                - response: The generated answer
                - reference: Ground truth answer (optional)
                
        Returns:
            RAGAS EvaluationDataset object
        """
        print(f"\n📦 Preparing RAGAS dataset from {len(evaluation_data)} samples...")
        
        samples = []
        for item in evaluation_data:
            # Create SingleTurnSample for each evaluation item
            sample = SingleTurnSample(
                user_input=item['user_input'],
                retrieved_contexts=item.get('retrieved_contexts', []),
                response=item['response'],
                reference=item.get('reference', item['response'])  # Use response as reference if not provided
            )
            samples.append(sample)
        
        dataset = EvaluationDataset(samples=samples)
        print(f"✅ Created RAGAS dataset with {len(samples)} samples")
        
        return dataset
    
    def evaluate(
        self,
        evaluation_data: List[Dict[str, Any]],
        run_name: Optional[str] = None
    ) -> EvaluationResults:
        """
        Run RAGAS evaluation on the provided data.
        
        Args:
            evaluation_data: List of evaluation samples
            run_name: Optional name for the evaluation run
            
        Returns:
            EvaluationResults object with scores and detailed results
        """
        print(f"\n{'='*80}")
        print(f"RUNNING RAGAS EVALUATION")
        print(f"{'='*80}")
        print(f"Samples: {len(evaluation_data)}")
        print(f"Metrics: {len(self.metrics)}")
        
        if run_name is None:
            run_name = f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare dataset
        dataset = self.prepare_ragas_dataset(evaluation_data)
        
        # Run evaluation
        print(f"\n🔍 Running RAGAS evaluation...")
        print(f"   This may take a few minutes...")
        
        try:
            # Run RAGAS evaluate function
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.evaluator_llm
            )
            
            print(f"✅ Evaluation complete!")
            
            # Extract scores
            scores_df = result.to_pandas()
            
            # Calculate average scores for each metric
            faithfulness_score = scores_df['faithfulness'].mean() if 'faithfulness' in scores_df else 0.0
            answer_relevancy_score = scores_df['answer_relevancy'].mean() if 'answer_relevancy' in scores_df else 0.0
            context_precision_score = scores_df['context_precision'].mean() if 'context_precision' in scores_df else 0.0
            context_recall_score = scores_df['context_recall'].mean() if 'context_recall' in scores_df else 0.0
            answer_correctness_score = scores_df['answer_correctness'].mean() if 'answer_correctness' in scores_df else 0.0
            
            # Calculate overall average
            metric_scores = [
                faithfulness_score,
                answer_relevancy_score,
                context_precision_score,
                context_recall_score,
                answer_correctness_score
            ]
            average_score = sum(metric_scores) / len(metric_scores)
            
            # Create results object
            results = EvaluationResults(
                faithfulness_score=faithfulness_score,
                answer_relevancy_score=answer_relevancy_score,
                context_precision_score=context_precision_score,
                context_recall_score=context_recall_score,
                answer_correctness_score=answer_correctness_score,
                average_score=average_score,
                num_samples=len(evaluation_data),
                timestamp=datetime.now().isoformat(),
                detailed_results=scores_df
            )
            
            # Print results
            self._print_results(results)
            
            # Log to MLflow if enabled
            if self.log_to_mlflow:
                self._log_to_mlflow(results, run_name)
            
            return results
        
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            raise
    
    def _print_results(self, results: EvaluationResults):
        """Print evaluation results in a formatted way."""
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"\n📊 Aggregate Scores (0.0 - 1.0):")
        print(f"   Faithfulness:        {results.faithfulness_score:.3f}")
        print(f"   Answer Relevancy:    {results.answer_relevancy_score:.3f}")
        print(f"   Context Precision:   {results.context_precision_score:.3f}")
        print(f"   Context Recall:      {results.context_recall_score:.3f}")
        print(f"   Answer Correctness:  {results.answer_correctness_score:.3f}")
        print(f"   {'─'*40}")
        print(f"   Average Score:       {results.average_score:.3f}")
        print(f"\n📈 Evaluation Details:")
        print(f"   Samples Evaluated:   {results.num_samples}")
        print(f"   Timestamp:           {results.timestamp}")
        print(f"{'='*80}")
    
    def _log_to_mlflow(self, results: EvaluationResults, run_name: str):
        """Log evaluation results to MLflow."""
        print(f"\n📝 Logging results to MLflow...")
        
        try:
            with mlflow.start_run(run_name=run_name):
                # Log aggregate metrics
                mlflow.log_metric("faithfulness", results.faithfulness_score)
                mlflow.log_metric("answer_relevancy", results.answer_relevancy_score)
                mlflow.log_metric("context_precision", results.context_precision_score)
                mlflow.log_metric("context_recall", results.context_recall_score)
                mlflow.log_metric("answer_correctness", results.answer_correctness_score)
                mlflow.log_metric("average_score", results.average_score)
                mlflow.log_metric("num_samples", results.num_samples)
                
                # Log parameters
                mlflow.log_param("llm_endpoint", self.llm_endpoint)
                mlflow.log_param("num_metrics", len(self.metrics))
                mlflow.log_param("timestamp", results.timestamp)
                
                # Save detailed results as CSV artifact
                results_csv = "evaluation_results.csv"
                results.detailed_results.to_csv(results_csv, index=False)
                mlflow.log_artifact(results_csv)
                
                # Create and log summary JSON
                summary = {
                    "faithfulness": results.faithfulness_score,
                    "answer_relevancy": results.answer_relevancy_score,
                    "context_precision": results.context_precision_score,
                    "context_recall": results.context_recall_score,
                    "answer_correctness": results.answer_correctness_score,
                    "average_score": results.average_score,
                    "num_samples": results.num_samples,
                    "timestamp": results.timestamp
                }
                
                with open("evaluation_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
                mlflow.log_artifact("evaluation_summary.json")
                
                print(f"✅ Results logged to MLflow")
        
        except Exception as e:
            print(f"⚠️  MLflow logging error: {str(e)}")
    
    def evaluate_by_category(
        self,
        evaluation_data: List[Dict[str, Any]],
        category_key: str = 'question_type'
    ) -> Dict[str, EvaluationResults]:
        """
        Evaluate samples grouped by category.
        
        Args:
            evaluation_data: List of evaluation samples
            category_key: Key to group by (e.g., 'question_type', 'difficulty')
            
        Returns:
            Dict mapping category to EvaluationResults
        """
        print(f"\n📊 Evaluating by category: {category_key}")
        
        # Group samples by category
        categories = {}
        for sample in evaluation_data:
            category = sample.get(category_key, 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(sample)
        
        print(f"   Found {len(categories)} categories")
        
        # Evaluate each category
        results_by_category = {}
        for category, samples in categories.items():
            print(f"\n   Evaluating '{category}' ({len(samples)} samples)...")
            results = self.evaluate(samples, run_name=f"ragas_eval_{category}")
            results_by_category[category] = results
        
        return results_by_category
    
    def compare_results(
        self,
        results_list: List[EvaluationResults],
        labels: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple evaluation results.
        
        Args:
            results_list: List of EvaluationResults
            labels: Labels for each result set
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for label, results in zip(labels, results_list):
            comparison_data.append({
                'Label': label,
                'Faithfulness': results.faithfulness_score,
                'Answer Relevancy': results.answer_relevancy_score,
                'Context Precision': results.context_precision_score,
                'Context Recall': results.context_recall_score,
                'Answer Correctness': results.answer_correctness_score,
                'Average': results.average_score,
                'Samples': results.num_samples
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(f"\n📊 Comparison Results:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df


def main():
    """Example usage of RAGAS evaluator."""
    
    # Example evaluation data
    evaluation_data = [
        {
            "user_input": "How do I create a cluster in Databricks?",
            "retrieved_contexts": [
                "To create a cluster in Databricks, go to the Compute page and click Create Cluster. Configure the cluster settings including cluster name, Databricks runtime version, and node types."
            ],
            "response": "To create a cluster, navigate to the Compute page in your Databricks workspace and click the 'Create Cluster' button. You'll need to specify a cluster name, select a Databricks runtime version, and choose your node types.",
            "reference": "Go to Compute page, click Create Cluster, configure name, runtime, and node types.",
            "question_type": "how-to"
        },
        {
            "user_input": "What is Unity Catalog?",
            "retrieved_contexts": [
                "Unity Catalog is a unified governance solution for data and AI assets on Databricks. It provides centralized access control, auditing, lineage, and data discovery across all workspaces."
            ],
            "response": "Unity Catalog is Databricks' unified governance solution that provides centralized management of data and AI assets, including access control, auditing, and lineage tracking.",
            "reference": "Unity Catalog is a unified governance solution for data and AI on Databricks with centralized access control, auditing, and lineage.",
            "question_type": "conceptual"
        }
    ]
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(
        llm_endpoint="databricks-qwen3-next-80b-a3b-instruct",
        log_to_mlflow=True
    )
    
    # Run evaluation
    results = evaluator.evaluate(evaluation_data)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Average score: {results.average_score:.3f}")


if __name__ == "__main__":
    main()
