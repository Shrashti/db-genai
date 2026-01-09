# Helper Scripts

Python modules used by the notebooks.

## Core RAG System

- **conversational_agent.py** - RAG agent with retrieval tools and guardrails
- **guardrails.py** - Input/output validation and rejection handling
- **ingestion_pipeline.py** - Document processing and chunking

## Evaluation System

### Simple (Recommended for interviews)
- **simple_rag_eval.py** - Easy evaluation with 4 RAGAS metrics (150 lines)

### Advanced (Optional)
- **ground_truth_generator.py** - Generate 50 Q&A pairs from documentation
- **ragas_evaluator.py** - Full RAGAS implementation with category analysis

## Usage

Import these modules in your notebooks:

```python
# In notebooks
import sys
sys.path.insert(0, './helpers')

from conversational_agent import create_databricks_agent
from simple_rag_eval import SimpleRAGEvaluator
```

## File Sizes

- `simple_rag_eval.py` - ~150 lines (simple, interview-ready)
- `conversational_agent.py` - ~650 lines (production RAG agent)
- `guardrails.py` - ~500 lines (validation logic)
- `ingestion_pipeline.py` - ~800 lines (document processing)
- `ground_truth_generator.py` - ~450 lines (advanced evaluation)
- `ragas_evaluator.py` - ~350 lines (advanced evaluation)
