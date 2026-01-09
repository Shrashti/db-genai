# RAG Evaluation - Simple Guide

## What is this?

A simple way to measure if your RAG system gives good answers. Uses 4 metrics from RAGAS.

## The 4 Metrics (Easy to Remember)

1. **Faithfulness** - Is the answer based on what was retrieved? (No hallucinations)
2. **Answer Relevancy** - Does the answer actually answer the question?
3. **Context Precision** - Did we retrieve the right documents?
4. **Context Recall** - Did we get all the relevant information?

Each metric is scored 0.0 to 1.0 (higher is better).

## How to Use (3 Steps)

### Step 1: Prepare Test Questions

Create 5-10 test questions with expected answers:

```python
test_questions = [
    {
        "question": "How do I create a cluster?",
        "ground_truth": "Go to Compute, click Create Cluster",
        "contexts": ["Documentation about cluster creation..."],
        "answer": ""  # RAG will fill this
    },
    # Add more...
]
```

### Step 2: Run Your RAG System

```python
from conversational_agent import create_databricks_agent

agent = create_databricks_agent(
    vector_search_endpoint="your_endpoint",
    vector_search_index="your_index"
)

# Get answers from RAG
for item in test_questions:
    result = agent.query(item['question'])
    item['answer'] = result['response']
```

### Step 3: Evaluate

```python
from simple_rag_eval import SimpleRAGEvaluator

evaluator = SimpleRAGEvaluator()
scores = evaluator.evaluate_rag_system(test_questions)

# Results:
# faithfulness: 0.850
# answer_relevancy: 0.920
# context_precision: 0.780
# context_recall: 0.810
# average: 0.840
```

## What Good Scores Look Like

- **0.8+** = Excellent
- **0.6-0.8** = Good, room for improvement
- **Below 0.6** = Needs work

## For Interviews

**"How do you evaluate your RAG system?"**

> "I use RAGAS with 4 key metrics: Faithfulness checks for hallucinations, Answer Relevancy ensures we answer the question, Context Precision and Recall measure retrieval quality. I test on 10 sample questions and track scores in MLflow. Scores above 0.8 indicate production-ready quality."

Simple, clear, confident.

## Files You Need

- `simple_rag_eval.py` - The evaluator (150 lines, easy to read)
- `sample_questions.json` - Your test questions (10 examples)
- `conversational_agent.py` - Your RAG system (already exists)

That's it. No complexity.

## Quick Demo

```python
# Complete example in 10 lines
from simple_rag_eval import SimpleRAGEvaluator, load_sample_questions, run_rag_and_evaluate
from conversational_agent import create_databricks_agent

# Setup
agent = create_databricks_agent("endpoint", "index")
questions = load_sample_questions()

# Run evaluation
scores = run_rag_and_evaluate(agent, questions)

# Done! Scores are in MLflow
```

## Next Steps

1. Create your 10 test questions
2. Run evaluation once to get baseline
3. Make improvements to your RAG system
4. Re-run to see if scores improved

Keep it simple. Focus on the 4 metrics. Track in MLflow.
