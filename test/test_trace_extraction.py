
import sys
import os
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add parent directory to path to import helpers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Notebooks", "helpers")))

# Mock dependencies before importing conversational_agent
sys.modules["mlflow"] = MagicMock()
sys.modules["databricks_langchain"] = MagicMock()
sys.modules["langchain_core.messages"] = MagicMock()
sys.modules["langgraph.prebuilt"] = MagicMock()

# Now import the module to test
from conversational_agent import DatabricksDocAgent

def test_trace_extraction():
    print("Testing trace extraction logic...")
    
    # Mock the agent and its components
    mock_llm_endpoint = "test-endpoint"
    mock_tools = []
    
    # Instantiate agent (dependencies are mocked)
    agent = DatabricksDocAgent(
        retrieval_tools=mock_tools,
        llm_endpoint=mock_llm_endpoint,
        enable_input_guardrail=False, # Disable to skip guardrail logic
        enable_output_guardrail=False,
        log_to_mlflow=False
    )
    
    # Mock the internal LangChain agent's invoke method
    # We need to simulate the message structure that the agent receives
    
    # 1. AIMessage with tool calls
    mock_ai_msg = MagicMock()
    mock_ai_msg.content = ""
    mock_ai_msg.tool_calls = [{
        "id": "call_123",
        "name": "generic_doc_retriever",
        "args": {"query": "how to create cluster"}
    }]
    
    # 2. ToolMessage with result
    mock_tool_msg = MagicMock()
    type(mock_tool_msg).__name__ = "ToolMessage"
    mock_tool_msg.name = "generic_doc_retriever"
    mock_tool_msg.tool_call_id = "call_123"
    mock_tool_msg.content = json.dumps([{"title": "Cluster Creation", "url": "doc.com/cluster"}])
    
    # 3. Final AIMessage response
    mock_final_msg = MagicMock()
    mock_final_msg.content = "To create a cluster, go to Compute..."
    
    # Setup the return value of agent.invoke
    agent.agent = MagicMock()
    agent.agent.invoke.return_value = {
        "messages": [
            MagicMock(), # System prompt (ignored by logic usually)
            MagicMock(), # User query (ignored)
            mock_ai_msg,
            mock_tool_msg,
            mock_final_msg
        ]
    }
    
    # Run query
    result = agent.query("How do I create a cluster?")
    
    # Verify trace in metadata
    trace = result["metadata"].get("trace")
    
    if not trace:
        print("❌ FAILED: No trace found in metadata")
        return
        
    print(f"✅ Found {len(trace)} trace items")
    
    item = trace[0]
    print(f"Item: {item}")
    
    assert item["tool"] == "generic_doc_retriever"
    assert item["inputs"] == {"query": "how to create cluster"}
    assert item["outputs"][0]["title"] == "Cluster Creation"
    
    print("✅ Trace extraction verification passed!")

if __name__ == "__main__":
    try:
        test_trace_extraction()
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
