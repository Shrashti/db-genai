# Quick Reference: Guardrails & Conversational Agent

## 🚀 Quick Start (Copy & Paste)

### 1. Basic Setup
```python
from conversational_agent import create_databricks_agent

# Create agent (uses your existing vector search setup)
agent = create_databricks_agent(
    vector_search_endpoint="databricks_doc_index",
    vector_search_index="workspace.default.databricks_index",
    guardrail_strictness="moderate"  # strict/moderate/lenient
)
```

### 2. Single Query
```python
# Ask a question
result = agent.query("How do I create a Databricks cluster?")

# Check if rejected
if result["was_rejected"]:
    print("Query was rejected:", result["response"])
else:
    print("Answer:", result["response"])
    print(f"Tools used: {result['metadata']['tools_used']}")
```

### 3. Multi-Turn Conversation
```python
conversation_id = "my_session"

# Question 1
result1 = agent.query(
    "What is Delta Lake?",
    conversation_id=conversation_id
)

# Question 2 (remembers context)
result2 = agent.query(
    "How do I create a Delta table?",
    conversation_id=conversation_id,
    include_history=True
)

# Question 3 (remembers both previous questions)
result3 = agent.query(
    "Show me a code example",
    conversation_id=conversation_id,
    include_history=True
)
```

### 4. View Conversation History
```python
history = agent.get_conversation_history(conversation_id)

for i, turn in enumerate(history, 1):
    print(f"\nTurn {i}:")
    print(f"User: {turn['user']}")
    print(f"Assistant: {turn['assistant'][:100]}...")
```

---

## 🛡️ Guardrail Configuration

### Strictness Levels
```python
# STRICT - Only clear Databricks queries
agent = create_databricks_agent(..., guardrail_strictness="strict")
# ✅ "How do I create a Databricks cluster?"
# ❌ "How do I use Spark?"  (too general)

# MODERATE - Balanced (RECOMMENDED)
agent = create_databricks_agent(..., guardrail_strictness="moderate")
# ✅ "How do I create a Databricks cluster?"
# ✅ "How do I use Spark?"  (Spark is used in Databricks)
# ❌ "What is Python?"  (too general)

# LENIENT - Broad interpretation
agent = create_databricks_agent(..., guardrail_strictness="lenient")
# ✅ Most data engineering/ML queries
# ❌ Only completely off-topic queries
```

### Disable Guardrails (for testing)
```python
from conversational_agent import DatabricksDocAgent

agent = DatabricksDocAgent(
    retrieval_tools=your_tools,
    enable_input_guardrail=False,   # Skip input validation
    enable_output_guardrail=False   # Skip output validation
)
```

---

## 📊 Response Format

### Successful Response
```python
{
    "response": "To create a cluster, go to Compute...",
    "conversation_id": "session_123",
    "was_rejected": False,
    "metadata": {
        "latency_ms": 1250,
        "tool_calls": 1,
        "tools_used": ["generic_doc_retriever"],
        "conversation_turn": 3
    }
}
```

### Rejected Response
```python
{
    "response": "I'm specifically designed to help with Databricks...",
    "conversation_id": "session_123",
    "was_rejected": True,
    "metadata": {
        "latency_ms": 180,
        "tool_calls": 0  # No wasted tool calls!
    }
}
```

---

## 📈 MLflow Tracking

### Automatic Logging
```python
import mlflow

mlflow.set_experiment("/Users/your-email/databricks-agent")

with mlflow.start_run(run_name="production_query"):
    result = agent.query("How do I use MLflow?")
    
# Automatically logged:
# - input_guardrail_confidence
# - input_guardrail_category
# - output_guardrail_confidence
# - num_tool_calls
# - tools_used
```

### Get Metrics Summary
```python
metrics = agent.get_metrics_summary()

print(f"Total Queries: {metrics['total_queries']}")
print(f"Rejected: {metrics['rejected_queries']}")
print(f"Rejection Rate: {metrics['rejection_rate']:.1%}")
print(f"Avg Confidence: {metrics['avg_input_confidence']:.2f}")
```

---

## 🧪 Testing

### Test Individual Guardrails
```python
from guardrails import InputGuardrail

guardrail = InputGuardrail(strictness="moderate", log_to_mlflow=False)

# Test a query
result = guardrail.validate("How do I create a cluster?")

print(f"Valid: {result.is_valid}")
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reason: {result.reason}")
```

### Run Test Suite
```bash
cd Notebooks
python test_guardrails.py
```

---

## 🚢 Deployment

### As MLflow Model
```python
import mlflow

class DatabricksDocEndpoint:
    def __init__(self):
        self.agent = create_databricks_agent(...)
    
    def predict(self, model_input):
        return self.agent.query(
            user_query=model_input["query"],
            conversation_id=model_input.get("conversation_id")
        )

# Log model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=DatabricksDocEndpoint()
    )
```

### As REST API (FastAPI)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
agent = create_databricks_agent(...)

class QueryRequest(BaseModel):
    query: str
    conversation_id: str = None

@app.post("/query")
def query_endpoint(request: QueryRequest):
    return agent.query(
        user_query=request.query,
        conversation_id=request.conversation_id
    )

# Run: uvicorn api:app --reload
```

---

## 🔍 Common Use Cases

### Use Case 1: Customer Support Bot
```python
# Strict mode - only answer Databricks questions
agent = create_databricks_agent(..., guardrail_strictness="strict")

# Handle customer query
result = agent.query(
    user_query=customer_question,
    conversation_id=customer_id
)

if result["was_rejected"]:
    # Escalate to human support
    escalate_to_human(customer_id, customer_question)
else:
    # Send automated response
    send_response(customer_id, result["response"])
```

### Use Case 2: Internal Documentation Assistant
```python
# Moderate mode - allow related questions
agent = create_databricks_agent(..., guardrail_strictness="moderate")

# Multi-turn conversation with employee
conversation_id = f"employee_{employee_id}"

while True:
    question = get_employee_question()
    if question.lower() == "exit":
        break
    
    result = agent.query(question, conversation_id, include_history=True)
    display_response(result["response"])
```

### Use Case 3: Automated Testing
```python
# Test suite for documentation coverage
test_queries = [
    "How do I create a cluster?",
    "Explain MLflow tracking",
    "What is Unity Catalog?",
    # ... more queries
]

with mlflow.start_run(run_name="doc_coverage_test"):
    for query in test_queries:
        result = agent.query(query)
        
        # Log if query was rejected (might indicate missing docs)
        if result["was_rejected"]:
            mlflow.log_param(f"rejected_{query[:30]}", True)
```

---

## ⚙️ Advanced Configuration

### Custom Agent with Specific Tools
```python
from conversational_agent import DatabricksDocAgent
from databricks_langchain import VectorSearchRetrieverTool

# Create only the tools you need
api_retriever = VectorSearchRetrieverTool(
    endpoint_name="databricks_doc_index",
    index_name="workspace.default.databricks_index",
    columns=["chunk_id", "text", "url"],
    tool_name="api_docs_retriever",
    tool_description="Retrieves API documentation",
    filters={"doc_type": "api_reference"},
    num_results=5
)

# Create agent with custom configuration
agent = DatabricksDocAgent(
    retrieval_tools=[api_retriever],  # Only API docs
    llm_endpoint="databricks-qwen3-next-80b-a3b-instruct",
    guardrail_strictness="strict",
    max_conversation_history=5,  # Keep only last 5 turns
    log_to_mlflow=True
)
```

### Custom Rejection Messages
```python
from guardrails import RejectionHandler

class CustomRejectionHandler(RejectionHandler):
    def generate_rejection(self, query, guardrail_result):
        return f"Sorry! I can only answer Databricks questions. Try asking about clusters, MLflow, or Delta Lake."

# Use in your agent
# (requires modifying DatabricksDocAgent to accept custom handler)
```

---

## 🐛 Troubleshooting

### Problem: High Rejection Rate
```python
# Solution 1: Use lenient mode
agent = create_databricks_agent(..., guardrail_strictness="lenient")

# Solution 2: Check what's being rejected
metrics = agent.get_metrics_summary()
print(f"Rejection rate: {metrics['rejection_rate']:.1%}")

# Review rejected queries in MLflow
```

### Problem: Slow Responses
```python
# Solution 1: Disable output guardrail (saves ~100-200ms)
agent = DatabricksDocAgent(
    ...,
    enable_output_guardrail=False
)

# Solution 2: Check latency breakdown
result = agent.query("test query")
print(f"Total latency: {result['metadata']['latency_ms']:.0f}ms")
```

### Problem: False Rejections
```python
# Check guardrail decision
result = agent.query("How do I use Spark?")

if result["was_rejected"] and result["guardrail_input"]:
    print(f"Category: {result['guardrail_input'].category}")
    print(f"Confidence: {result['guardrail_input'].confidence}")
    print(f"Reason: {result['guardrail_input'].reason}")
    
# If false rejection, use lenient mode or disable guardrail
```

---

## 📚 Example Queries

### ✅ Will Be Accepted (Moderate Mode)
- "How do I create a cluster in Databricks?"
- "Explain MLflow experiment tracking"
- "What is Delta Lake and how does it work?"
- "Show me code examples for Databricks SQL"
- "How do I configure Unity Catalog?"
- "What are Databricks workflows?"
- "How do I use Apache Spark in Databricks?"

### ❌ Will Be Rejected (Moderate Mode)
- "What's the weather today?"
- "Tell me a joke"
- "How do I cook pasta?"
- "What is the capital of France?"
- "Explain quantum physics"
- "What is Python?" (too general)

---

## 📖 Full Documentation

- **Implementation Details**: [GUARDRAILS_README.md](file:///Users/shrgupta5/Personal/Databricks/assessment/db-genai/Notebooks/GUARDRAILS_README.md)
- **Complete Walkthrough**: [walkthrough.md](file:///Users/shrgupta5/.gemini/antigravity/brain/da1c2ae4-6c89-4694-bedc-761cdf758001/walkthrough.md)
- **Interactive Examples**: [03-Guardrails-Agent.ipynb](file:///Users/shrgupta5/Personal/Databricks/assessment/db-genai/Notebooks/03-Guardrails-Agent.ipynb)

---

## 💡 Tips

1. **Start with moderate strictness** - best balance for most use cases
2. **Use conversation IDs** - enables multi-turn conversations
3. **Monitor rejection rate** - high rate may indicate wrong strictness level
4. **Log to MLflow** - essential for debugging and optimization
5. **Test edge cases** - empty queries, special characters, etc.
6. **Review rejected queries** - may reveal missing documentation

---

## 🎯 Next Steps

1. Open [03-Guardrails-Agent.ipynb](file:///Users/shrgupta5/Personal/Databricks/assessment/db-genai/Notebooks/03-Guardrails-Agent.ipynb)
2. Run the cells to see guardrails in action
3. Try your own queries in Part 5 (Interactive Testing)
4. Adjust strictness based on your needs
5. Deploy to production!
