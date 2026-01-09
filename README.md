# db-genai

RAG system for Databricks documentation with evaluation

## Quick Overview

1. **Ingestion** - Load and chunk Databricks docs
2. **Agent** - RAG chatbot with guardrails  
3. **Evaluation** - Measure quality with RAGAS

## Project Structure

```
Notebooks/
├── 01-Ingestion.ipynb           # Document ingestion
├── 02-Knowledge-Base.ipynb      # Vector index setup
├── 03-Guardrails-Agent.ipynb    # RAG agent demo
├── 04-RAG-Evaluation.ipynb      # Advanced evaluation
├── EVALUATION_GUIDE.md          # How to evaluate (start here)
├── sample_questions.json        # 10 test questions
├── helpers/                     # Python modules
│   ├── conversational_agent.py  # RAG agent
│   ├── guardrails.py            # Validation
│   ├── ingestion_pipeline.py    # Document processing
│   ├── simple_rag_eval.py       # Simple evaluation ⭐
│   ├── ground_truth_generator.py # Advanced evaluation
│   └── ragas_evaluator.py       # Advanced evaluation
├── test/                        # Test suite
└── ground-truth data/           # Documentation CSV

⭐ = Start here for evaluation
```

## Quick Start

### 1. Evaluate Your RAG System (Simple)

## RAG Evaluation - Simple Approach

### 4 Metrics (0.0 to 1.0, higher is better)

1. **Faithfulness** - No hallucinations (answer based on retrieved docs)
2. **Answer Relevancy** - Actually answers the question
3. **Context Precision** - Retrieved the right documents
4. **Context Recall** - Got all relevant information

### Quick Start (3 Steps)

```python
# Step 1: Load test questions
import json
with open('sample_questions.json') as f:
    questions = json.load(f)

# Step 2: Get RAG answers
import sys
sys.path.insert(0, './helpers')
from conversational_agent import create_databricks_agent

agent = create_databricks_agent("endpoint", "index")

for q in questions:
    result = agent.query(q['question'])
    q['answer'] = result['response']

# Step 3: Evaluate
from simple_rag_eval import SimpleRAGEvaluator
evaluator = SimpleRAGEvaluator()
scores = evaluator.evaluate_rag_system(questions)

# Results: faithfulness: 0.85, answer_relevancy: 0.92, etc.
```

### What Good Scores Mean

- **0.8+** = Production ready
- **0.6-0.8** = Good, can improve
- **Below 0.6** = Needs work

## For Interviews

**Q: How do you evaluate your RAG system?**

> "I use RAGAS with 4 metrics: Faithfulness prevents hallucinations, Answer Relevancy ensures we answer the question, Context Precision and Recall measure retrieval quality. I test on 10 sample questions and track scores in MLflow. Scores above 0.8 mean production-ready quality."

**Files to show:**
1. `EVALUATION_GUIDE.md` - Simple explanation
2. `helpers/simple_rag_eval.py` - Clean code (150 lines)
3. `sample_questions.json` - Test data (10 questions)

Simple, clear, confident.
