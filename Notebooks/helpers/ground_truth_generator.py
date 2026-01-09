"""
Ground Truth Data Generator for RAG Evaluation

This module generates high-quality question-answer pairs from Databricks documentation
for use in RAGAS-based evaluation of the RAG system.

Features:
- Loads and samples documentation from CSV
- Generates realistic questions using LLM
- Creates reference answers
- Exports RAGAS-compatible datasets
"""

import pandas as pd
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import random
from pathlib import Path
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class EvaluationSample:
    """Represents a single evaluation sample in RAGAS format."""
    user_input: str  # The question
    retrieved_contexts: List[str]  # Relevant documentation chunks
    response: str  # Expected/reference answer
    reference: str  # Ground truth answer
    doc_url: str  # Source documentation URL
    doc_id: str  # Document ID
    question_type: str  # Type of question (factual, how-to, api, etc.)
    difficulty: str  # easy, medium, hard


class GroundTruthGenerator:
    """
    Generates ground truth evaluation datasets from Databricks documentation.
    
    Samples documentation entries and uses an LLM to generate realistic
    questions and reference answers for RAG evaluation.
    """
    
    def __init__(
        self,
        csv_path: str,
        llm_endpoint: str = "databricks-qwen3-next-80b-a3b-instruct",
        random_seed: int = 42
    ):
        """
        Initialize the ground truth generator.
        
        Args:
            csv_path: Path to the documentation CSV file
            llm_endpoint: Databricks LLM endpoint for question generation
            random_seed: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.llm = ChatDatabricks(endpoint=llm_endpoint)
        random.seed(random_seed)
        
        print(f"📚 Loading documentation from: {csv_path}")
        self.docs_df = pd.read_csv(csv_path, sep='\t')
        print(f"✅ Loaded {len(self.docs_df)} documentation entries")
        
        # Question generation prompt
        self.question_gen_prompt = """You are an expert at creating realistic questions from documentation.

Given a piece of Databricks documentation, generate 1-2 high-quality questions that a user might ask.

Requirements:
1. Questions should be natural and realistic
2. Questions should be answerable from the provided documentation
3. Vary question types: factual, how-to, conceptual, API-related
4. Make questions specific and clear
5. Return ONLY the questions, one per line, no numbering

Documentation:
{documentation}

Generate 1-2 questions:"""

        self.answer_gen_prompt = """You are a Databricks documentation expert.

Given a question and relevant documentation, provide a clear, accurate reference answer.

Requirements:
1. Answer should be based ONLY on the provided documentation
2. Be concise but complete
3. Include key details and steps if applicable
4. Do not add information not in the documentation

Question: {question}

Documentation:
{documentation}

Reference Answer:"""
    
    def sample_documents(
        self,
        num_samples: int = 50,
        strategy: str = "diverse"
    ) -> pd.DataFrame:
        """
        Sample documents from the full dataset.
        
        Args:
            num_samples: Number of documents to sample
            strategy: Sampling strategy - "random", "diverse", or "stratified"
            
        Returns:
            DataFrame with sampled documents
        """
        print(f"\n🎲 Sampling {num_samples} documents using '{strategy}' strategy...")
        
        if strategy == "random":
            sampled = self.docs_df.sample(n=num_samples, random_state=42)
        
        elif strategy == "diverse":
            # Sample from different URL patterns to get diverse content
            # Group by URL domain/path to ensure diversity
            self.docs_df['url_prefix'] = self.docs_df['url'].str.extract(r'(https://[^/]+/[^/]+/[^/]+)')
            
            # Get unique URL prefixes
            url_groups = self.docs_df.groupby('url_prefix')
            
            # Sample proportionally from each group
            samples_per_group = max(1, num_samples // len(url_groups))
            sampled_list = []
            
            for _, group in url_groups:
                n = min(samples_per_group, len(group))
                sampled_list.append(group.sample(n=n, random_state=42))
            
            sampled = pd.concat(sampled_list)
            
            # If we need more samples, randomly sample the remainder
            if len(sampled) < num_samples:
                remaining = num_samples - len(sampled)
                additional = self.docs_df[~self.docs_df.index.isin(sampled.index)].sample(
                    n=remaining, random_state=42
                )
                sampled = pd.concat([sampled, additional])
            
            # If we have too many, randomly select num_samples
            if len(sampled) > num_samples:
                sampled = sampled.sample(n=num_samples, random_state=42)
            
            sampled = sampled.drop(columns=['url_prefix'])
        
        else:  # stratified by content length
            # Stratify by content length (short, medium, long)
            self.docs_df['content_length'] = self.docs_df['content'].str.len()
            self.docs_df['length_category'] = pd.qcut(
                self.docs_df['content_length'],
                q=3,
                labels=['short', 'medium', 'long']
            )
            
            sampled = self.docs_df.groupby('length_category', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), num_samples // 3), random_state=42)
            )
            
            sampled = sampled.drop(columns=['content_length', 'length_category'])
        
        print(f"✅ Sampled {len(sampled)} documents")
        return sampled.reset_index(drop=True)
    
    def generate_questions(
        self,
        documentation: str,
        num_questions: int = 2
    ) -> List[str]:
        """
        Generate questions from a documentation chunk using LLM.
        
        Args:
            documentation: Documentation text
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Truncate very long documentation to avoid token limits
        max_chars = 2000
        if len(documentation) > max_chars:
            documentation = documentation[:max_chars] + "..."
        
        prompt = self.question_gen_prompt.format(documentation=documentation)
        
        try:
            messages = [
                SystemMessage(content="You are a question generation expert."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            questions_text = response.content.strip()
            
            # Parse questions (one per line)
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            # Remove numbering if present
            questions = [q.lstrip('0123456789.-) ') for q in questions]
            
            # Limit to requested number
            questions = questions[:num_questions]
            
            return questions
        
        except Exception as e:
            print(f"⚠️  Error generating questions: {str(e)}")
            return []
    
    def generate_reference_answer(
        self,
        question: str,
        documentation: str
    ) -> str:
        """
        Generate a reference answer for a question using LLM.
        
        Args:
            question: The question
            documentation: Relevant documentation
            
        Returns:
            Reference answer
        """
        # Truncate very long documentation
        max_chars = 2000
        if len(documentation) > max_chars:
            documentation = documentation[:max_chars] + "..."
        
        prompt = self.answer_gen_prompt.format(
            question=question,
            documentation=documentation
        )
        
        try:
            messages = [
                SystemMessage(content="You are a Databricks documentation expert."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
        
        except Exception as e:
            print(f"⚠️  Error generating answer: {str(e)}")
            return ""
    
    def classify_question_type(self, question: str) -> str:
        """Classify the type of question."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['how to', 'how do', 'how can', 'steps to']):
            return 'how-to'
        elif any(word in question_lower for word in ['what is', 'what are', 'define', 'explain']):
            return 'conceptual'
        elif any(word in question_lower for word in ['api', 'method', 'function', 'parameter', 'class']):
            return 'api'
        elif any(word in question_lower for word in ['example', 'code', 'sample', 'implement']):
            return 'code-example'
        elif any(word in question_lower for word in ['configure', 'setup', 'install', 'create']):
            return 'configuration'
        else:
            return 'factual'
    
    def estimate_difficulty(self, question: str, documentation: str) -> str:
        """Estimate question difficulty based on complexity."""
        # Simple heuristic based on question and doc length
        question_words = len(question.split())
        doc_words = len(documentation.split())
        
        if question_words > 15 or doc_words > 500:
            return 'hard'
        elif question_words > 8 or doc_words > 200:
            return 'medium'
        else:
            return 'easy'
    
    def generate_dataset(
        self,
        num_samples: int = 50,
        questions_per_doc: int = 1,
        sampling_strategy: str = "diverse"
    ) -> List[EvaluationSample]:
        """
        Generate a complete evaluation dataset.
        
        Args:
            num_samples: Number of documentation samples to use
            questions_per_doc: Number of questions to generate per document
            sampling_strategy: Document sampling strategy
            
        Returns:
            List of EvaluationSample objects
        """
        print(f"\n{'='*80}")
        print(f"GENERATING EVALUATION DATASET")
        print(f"{'='*80}")
        print(f"Target samples: {num_samples}")
        print(f"Questions per doc: {questions_per_doc}")
        
        # Sample documents
        sampled_docs = self.sample_documents(num_samples, sampling_strategy)
        
        evaluation_samples = []
        
        print(f"\n🤖 Generating questions and answers...")
        for idx, row in sampled_docs.iterrows():
            doc_id = str(row['id'])
            url = row['url']
            content = row['content']
            
            print(f"\n--- Processing document {idx + 1}/{len(sampled_docs)} (ID: {doc_id}) ---")
            
            # Generate questions
            questions = self.generate_questions(content, questions_per_doc)
            
            if not questions:
                print(f"⚠️  No questions generated, skipping...")
                continue
            
            print(f"✓ Generated {len(questions)} question(s)")
            
            for q_idx, question in enumerate(questions):
                print(f"  Q{q_idx + 1}: {question[:80]}...")
                
                # Generate reference answer
                reference_answer = self.generate_reference_answer(question, content)
                
                if not reference_answer:
                    print(f"  ⚠️  No answer generated, skipping question...")
                    continue
                
                print(f"  ✓ Generated reference answer ({len(reference_answer)} chars)")
                
                # Classify question
                q_type = self.classify_question_type(question)
                difficulty = self.estimate_difficulty(question, content)
                
                # Create evaluation sample
                sample = EvaluationSample(
                    user_input=question,
                    retrieved_contexts=[content],  # The source documentation
                    response=reference_answer,  # This will be replaced with actual agent response
                    reference=reference_answer,  # Ground truth answer
                    doc_url=url,
                    doc_id=doc_id,
                    question_type=q_type,
                    difficulty=difficulty
                )
                
                evaluation_samples.append(sample)
                print(f"  ✓ Added sample (type: {q_type}, difficulty: {difficulty})")
        
        print(f"\n{'='*80}")
        print(f"✅ Generated {len(evaluation_samples)} evaluation samples")
        print(f"{'='*80}")
        
        # Print summary statistics
        self._print_dataset_summary(evaluation_samples)
        
        return evaluation_samples
    
    def _print_dataset_summary(self, samples: List[EvaluationSample]):
        """Print summary statistics of the generated dataset."""
        print(f"\n📊 Dataset Summary:")
        print(f"   Total samples: {len(samples)}")
        
        # Question type distribution
        type_counts = {}
        for sample in samples:
            type_counts[sample.question_type] = type_counts.get(sample.question_type, 0) + 1
        
        print(f"\n   Question Types:")
        for q_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {q_type}: {count}")
        
        # Difficulty distribution
        diff_counts = {}
        for sample in samples:
            diff_counts[sample.difficulty] = diff_counts.get(sample.difficulty, 0) + 1
        
        print(f"\n   Difficulty Levels:")
        for difficulty, count in sorted(diff_counts.items()):
            print(f"     - {difficulty}: {count}")
    
    def export_to_json(
        self,
        samples: List[EvaluationSample],
        output_path: str
    ):
        """
        Export evaluation samples to JSON format.
        
        Args:
            samples: List of evaluation samples
            output_path: Path to save JSON file
        """
        print(f"\n💾 Exporting to JSON: {output_path}")
        
        # Convert to dict format
        samples_dict = [asdict(sample) for sample in samples]
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(samples)} samples to {output_path}")
    
    def export_to_csv(
        self,
        samples: List[EvaluationSample],
        output_path: str
    ):
        """
        Export evaluation samples to CSV format.
        
        Args:
            samples: List of evaluation samples
            output_path: Path to save CSV file
        """
        print(f"\n💾 Exporting to CSV: {output_path}")
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(sample) for sample in samples])
        
        # Convert list columns to strings for CSV
        df['retrieved_contexts'] = df['retrieved_contexts'].apply(
            lambda x: x[0] if x else ""
        )
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False)
        
        print(f"✅ Exported {len(samples)} samples to {output_path}")
    
    def export_ragas_format(
        self,
        samples: List[EvaluationSample],
        output_path: str
    ):
        """
        Export in RAGAS-compatible format (simplified).
        
        Args:
            samples: List of evaluation samples
            output_path: Path to save JSON file
        """
        print(f"\n💾 Exporting RAGAS format: {output_path}")
        
        # Convert to RAGAS format (only essential fields)
        ragas_samples = []
        for sample in samples:
            ragas_samples.append({
                "user_input": sample.user_input,
                "retrieved_contexts": sample.retrieved_contexts,
                "response": sample.response,
                "reference": sample.reference
            })
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ragas_samples, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(ragas_samples)} samples in RAGAS format")


def main():
    """Example usage of the ground truth generator."""
    
    # Initialize generator
    generator = GroundTruthGenerator(
        csv_path="ground-truth data/databricks_docs.csv",
        llm_endpoint="databricks-qwen3-next-80b-a3b-instruct"
    )
    
    # Generate 50 evaluation samples
    samples = generator.generate_dataset(
        num_samples=50,
        questions_per_doc=1,
        sampling_strategy="diverse"
    )
    
    # Export in multiple formats
    generator.export_to_json(samples, "ground-truth data/evaluation_dataset.json")
    generator.export_to_csv(samples, "ground-truth data/evaluation_dataset.csv")
    generator.export_ragas_format(samples, "ground-truth data/ragas_dataset.json")
    
    print(f"\n✅ Ground truth generation complete!")
    print(f"   Generated {len(samples)} evaluation samples")
    print(f"   Exported to 3 formats (JSON, CSV, RAGAS)")


if __name__ == "__main__":
    main()
