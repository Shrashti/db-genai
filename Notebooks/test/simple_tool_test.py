"""
Simple Tool Calling Test Script

This script tests if the agent is actually calling tools.
It will show you exactly what's happening in the agent's message flow.
"""

import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("SIMPLE TOOL CALLING TEST")
print("=" * 80)
print()

# Step 1: Check if we can import the agent
print("Step 1: Importing agent...")
try:
    from conversational_agent import DatabricksDocAgent
    print("✅ Agent imported successfully")
except Exception as e:
    print(f"❌ Failed to import agent: {e}")
    sys.exit(1)

# Step 2: Create mock tools for testing
print("\nStep 2: Creating mock retrieval tools...")

class MockRetrieverTool:
    """Mock tool that simulates a retrieval tool."""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        print(f"  Created mock tool: {name}")
    
    def __call__(self, query):
        return f"[MOCK RESULT from {self.name}]: Documentation about {query}"
    
    def invoke(self, query):
        return self.__call__(query)

# Create mock tools
mock_tools = [
    MockRetrieverTool(
        "generic_doc_retriever",
        "Retrieves generic Databricks documentation"
    ),
    MockRetrieverTool(
        "api_docs_retriever", 
        "Retrieves API reference documentation"
    )
]

print(f"✅ Created {len(mock_tools)} mock tools")

# Step 3: Initialize agent
print("\nStep 3: Initializing agent...")
print("  LLM endpoint: databricks-qwen3-next-80b-a3b-instruct")
print("  Guardrails: Disabled for testing")
print("  MLflow logging: Disabled for testing")

try:
    agent = DatabricksDocAgent(
        retrieval_tools=mock_tools,
        llm_endpoint="databricks-qwen3-next-80b-a3b-instruct",
        enable_input_guardrail=False,  # Disable to focus on tool calling
        enable_output_guardrail=False,
        log_to_mlflow=False
    )
    print("✅ Agent initialized")
except Exception as e:
    print(f"❌ Failed to initialize agent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test with a simple query
print("\n" + "=" * 80)
print("Step 4: Testing with query")
print("=" * 80)
print()

test_query = "What is Delta Lake?"
print(f"Query: {test_query}")
print()

try:
    result = agent.query(
        user_query=test_query,
        conversation_id="simple_test_1"
    )
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()
    
    # Check tool calls
    tool_calls = result['metadata']['tool_calls']
    tools_used = result['metadata']['tools_used']
    
    print(f"Tool Calls: {tool_calls}")
    print(f"Tools Used: {tools_used}")
    print()
    
    if tool_calls > 0:
        print("✅ SUCCESS! Tools were called!")
        print(f"   {tool_calls} tool call(s) detected")
        print(f"   Tools: {tools_used}")
    else:
        print("❌ FAILURE! No tools were called!")
        print()
        print("This means the LLM is answering without using retrieval tools.")
        print()
        print("Possible reasons:")
        print("  1. LLM endpoint doesn't support function calling")
        print("  2. Tools aren't bound to the LLM properly")
        print("  3. System prompt isn't strong enough")
        print("  4. Agent configuration issue")
    
    print()
    print("Response preview:")
    print("-" * 80)
    print(result['response'][:500] + "...")
    
except Exception as e:
    print(f"❌ Query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print()
print("Check the console output above, especially:")
print("  - 'Step 3a: Analyzing Agent Response' section")
print("  - Look for '✓ Has tool_calls attribute'")
print("  - Count of messages (should be 5+ if tools used)")
print()
