"""
Simple RAG Evaluation System

A straightforward approach to evaluate RAG quality using RAGAS metrics.
Perfect for demonstrating in interviews.

Usage:
    python simple_rag_eval.py
"""

import pandas as pd
import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from databricks_langchain import ChatDatabricks
import mlflow


class SimpleRAGEvaluator:
    """
    Simple RAG evaluator - easy to understand and explain.
    
    Three steps:
    1. Load test questions
    2. Get RAG responses
    3. Evaluate with RAGAS
    """
    
    def __init__(self, llm_endpoint="databricks-qwen3-next-80b-a3b-instruct"):
        """Initialize with LLM endpoint."""
        self.llm_endpoint = llm_endpoint
        llm = ChatDatabricks(endpoint=llm_endpoint)
        self.evaluator_llm = LangchainLLMWrapper(llm)
        
        # Just 4 key metrics - easy to remember
        self.metrics = [
            faithfulness,          # Is answer grounded in context?
            answer_relevancy,      # Is answer relevant to question?
            context_precision,     # Are good contexts ranked high?
            context_recall         # Did we retrieve all relevant info?
        ]
    
    def evaluate_rag_system(self, test_data):
        """
        Evaluate RAG system on test data.
        
        Args:
            test_data: List of dicts with:
                - question: The user question
                - answer: RAG system's answer
                - contexts: Retrieved contexts
                - ground_truth: Expected answer
        
        Returns:
            Dict with metric scores
        """
        print(f"\n🔍 Evaluating {len(test_data)} samples...")
        
        # Convert to RAGAS format
        samples = []
        for item in test_data:
            samples.append(SingleTurnSample(
                user_input=item['question'],
                response=item['answer'],
                retrieved_contexts=item['contexts'],
                reference=item['ground_truth']
            ))
        
        dataset = EvaluationDataset(samples=samples)
        
        # Run evaluation
        result = evaluate(dataset=dataset, metrics=self.metrics, llm=self.evaluator_llm)
        scores_df = result.to_pandas()
        
        # Calculate averages
        scores = {
            'faithfulness': scores_df['faithfulness'].mean(),
            'answer_relevancy': scores_df['answer_relevancy'].mean(),
            'context_precision': scores_df['context_precision'].mean(),
            'context_recall': scores_df['context_recall'].mean()
        }
        scores['average'] = sum(scores.values()) / len(scores)
        
        # Print results
        print("\n📊 Results:")
        for metric, score in scores.items():
            print(f"   {metric}: {score:.3f}")
        
        return scores


def load_sample_questions():
    """Load 10 simple test questions - easy to review."""
    return [
        {
            "question": "How do I create a cluster in Databricks?",
            "ground_truth": "Go to Compute page, click Create Cluster, configure settings.",
            "contexts": ["To create a cluster, navigate to Compute and click Create Cluster."],
            "answer": ""  # Will be filled by RAG system
        },
        {
            "question": "What is Unity Catalog?",
            "ground_truth": "Unity Catalog is a unified governance solution for data and AI.",
            "contexts": ["Unity Catalog provides centralized governance for data and AI assets."],
            "answer": ""
        },
        {
            "question": "How do I read a Delta table?",
            "ground_truth": "Use spark.read.format('delta').load('/path') or spark.table('table_name')",
            "contexts": ["Read Delta tables with spark.read.format('delta').load() or spark.table()"],
            "answer": ""
        },
        {
            "question": "What is Photon?",
            "ground_truth": "Photon is a vectorized query engine for faster SQL performance.",
            "contexts": ["Photon is a vectorized engine that speeds up SQL queries 2-3x."],
            "answer": ""
        },
        {
            "question": "How do I schedule a notebook?",
            "ground_truth": "Create a job in Workflows, add notebook task, set schedule.",
            "contexts": ["Schedule notebooks by creating a job and adding a trigger."],
            "answer": ""
        }
    ]


def run_rag_and_evaluate(agent, test_questions):
    """
    Simple workflow: Run RAG, then evaluate.
    
    Args:
        agent: Your RAG agent
        test_questions: List of test questions
    
    Returns:
        Evaluation scores
    """
    print("\n" + "="*60)
    print("SIMPLE RAG EVALUATION")
    print("="*60)
    
    # Step 1: Get RAG responses
    print("\n📝 Step 1: Getting RAG responses...")
    for i, item in enumerate(test_questions):
        result = agent.query(item['question'])
        item['answer'] = result['response']
        print(f"   {i+1}. {item['question'][:50]}... ✓")
    
    # Step 2: Evaluate
    print("\n📊 Step 2: Evaluating with RAGAS...")
    evaluator = SimpleRAGEvaluator()
    scores = evaluator.evaluate_rag_system(test_questions)
    
    # Step 3: Log to MLflow
    print("\n💾 Step 3: Logging to MLflow...")
    with mlflow.start_run(run_name="simple_rag_eval"):
        for metric, score in scores.items():
            mlflow.log_metric(metric, score)
    
    print("\n✅ Done! Check MLflow for results.")
    return scores


if __name__ == "__main__":
    # Example usage
    print("Simple RAG Evaluation System")
    print("="*60)
    print("\nTo use:")
    print("1. Load your RAG agent")
    print("2. Load test questions")
    print("3. Run: scores = run_rag_and_evaluate(agent, questions)")
    print("\nThat's it! 3 steps, 4 metrics, easy to explain.")
