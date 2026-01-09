"""
Test script to verify DatabricksDocAgent logging and tool calling.

This script demonstrates:
1. Comprehensive logging at each step
2. Tool call tracking and reporting
3. MLflow integration
4. Guardrail validation
"""

from conversational_agent import DatabricksDocAgent
from databricks_langchain import VectorSearchRetrieverTool
import mlflow

# Configuration
VECTOR_SEARCH_ENDPOINT = "your_vector_search_endpoint"
VECTOR_SEARCH_INDEX = "your_vector_search_index"
LLM_ENDPOINT = "databricks-qwen3-next-80b-a3b-instruct"

def create_mock_retrieval_tools():
    """
    Create mock retrieval tools for testing.
    Replace with actual vector search tools in production.
    """
    # This is a placeholder - replace with actual tools
    class MockRetrieverTool:
        def __init__(self, name, description):
            self.name = name
            self.description = description
        
        def invoke(self, query):
            return f"Mock result for {self.name}: {query}"
    
    return [
        MockRetrieverTool("generic_doc_retriever", "Retrieves generic Databricks documentation"),
        MockRetrieverTool("api_docs_retriever", "Retrieves API reference documentation"),
        MockRetrieverTool("tutorial_retriever", "Retrieves tutorial and how-to guides"),
        MockRetrieverTool("code_examples_retriever", "Retrieves documentation with code examples"),
    ]


def test_agent_with_logging():
    """Test the agent with comprehensive logging."""
    
    print("=" * 80)
    print("INITIALIZING DATABRICKS DOC AGENT")
    print("=" * 80)
    
    # Create retrieval tools
    retrieval_tools = create_mock_retrieval_tools()
    
    # Initialize agent
    agent = DatabricksDocAgent(
        retrieval_tools=retrieval_tools,
        llm_endpoint=LLM_ENDPOINT,
        guardrail_strictness="moderate",
        enable_input_guardrail=True,
        enable_output_guardrail=True,
        max_conversation_history=10,
        log_to_mlflow=True
    )
    
    print("\n✅ Agent initialized successfully")
    print(f"   - Retrieval tools: {len(retrieval_tools)}")
    print(f"   - Input guardrail: Enabled")
    print(f"   - Output guardrail: Enabled")
    print(f"   - MLflow logging: Enabled")
    
    # Test queries
    test_queries = [
        {
            "query": "How do I use MLflow with Databricks jobs?",
            "description": "Valid Databricks query - should be accepted"
        },
        {
            "query": "What's the weather like today?",
            "description": "Off-topic query - should be rejected"
        },
        {
            "query": "How do I create a Delta table?",
            "description": "Valid Databricks query - should be accepted"
        }
    ]
    
    # Start MLflow run
    with mlflow.start_run(run_name="agent_logging_test"):
        conversation_id = "test_conv_1"
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n\n{'#' * 80}")
            print(f"TEST CASE {i}: {test_case['description']}")
            print(f"{'#' * 80}")
            print(f"Query: {test_case['query']}")
            
            # Query the agent
            result = agent.query(
                user_query=test_case['query'],
                conversation_id=conversation_id,
                include_history=True
            )
            
            # Print summary
            print(f"\n{'=' * 80}")
            print(f"TEST CASE {i} SUMMARY")
            print(f"{'=' * 80}")
            print(f"Was Rejected: {result['was_rejected']}")
            print(f"Tool Calls: {result['metadata']['tool_calls']}")
            print(f"Tools Used: {result['metadata']['tools_used']}")
            print(f"Latency: {result['metadata']['latency_ms']:.0f}ms")
            
            if result.get('guardrail_input'):
                gr = result['guardrail_input']
                print(f"\nInput Guardrail:")
                print(f"  Valid: {gr.is_valid}")
                print(f"  Confidence: {gr.confidence}")
                print(f"  Category: {gr.category}")
            
            print(f"\n{'=' * 80}\n")
        
        # Print conversation history
        print(f"\n{'=' * 80}")
        print(f"CONVERSATION HISTORY")
        print(f"{'=' * 80}")
        
        history = agent.get_conversation_history(conversation_id)
        for i, turn in enumerate(history, 1):
            print(f"\nTurn {i}:")
            print(f"  Timestamp: {turn['timestamp']}")
            print(f"  User: {turn['user'][:100]}...")
            print(f"  Rejected: {turn['was_rejected']}")
            print(f"  Tool Calls: {len(turn['tool_calls'])}")
            print(f"  Tools: {turn['tool_calls']}")
        
        # Print metrics summary
        print(f"\n{'=' * 80}")
        print(f"METRICS SUMMARY")
        print(f"{'=' * 80}")
        
        metrics = agent.get_metrics_summary()
        print(f"Total Queries: {metrics.get('total_queries', 0)}")
        print(f"Accepted Queries: {metrics.get('accepted_queries', 0)}")
        print(f"Rejected Queries: {metrics.get('rejected_queries', 0)}")
        
        # Export metrics to MLflow
        agent.export_metrics_to_mlflow()
        
        print(f"\n✅ All metrics exported to MLflow")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DATABRICKS DOC AGENT - LOGGING & TOOL CALL TEST")
    print("=" * 80)
    print("\nThis test demonstrates:")
    print("  1. Comprehensive step-by-step logging")
    print("  2. Tool call tracking and reporting")
    print("  3. Guardrail validation (input/output)")
    print("  4. MLflow integration")
    print("  5. Conversation memory management")
    print("\n" + "=" * 80 + "\n")
    
    test_agent_with_logging()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nCheck the console output above to see:")
    print("  - Detailed logging at each step")
    print("  - Tool call tracking")
    print("  - Guardrail decisions")
    print("  - MLflow logging confirmation")
    print("\n" + "=" * 80 + "\n")
