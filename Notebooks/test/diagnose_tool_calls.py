"""
Diagnostic Script: Why Are Tool Calls Empty?

This script helps diagnose why tool calls are showing as 0.
Run this to see detailed information about the agent's behavior.
"""

import sys
from typing import Any

def diagnose_tool_calls():
    """
    Diagnose why tool calls might be empty.
    """
    print("=" * 80)
    print("TOOL CALL DIAGNOSTIC SCRIPT")
    print("=" * 80)
    print()
    
    # Common reasons for empty tool calls
    reasons = [
        {
            "reason": "1. System prompt doesn't enforce tool usage",
            "check": "Does the system prompt say 'ALWAYS use the retrieval tools'?",
            "solution": "Update system prompt to require tool usage"
        },
        {
            "reason": "2. LLM endpoint doesn't support function calling",
            "check": "Is the LLM endpoint capable of function calling?",
            "solution": "Use a model that supports function calling (e.g., GPT-4, Claude, or Databricks Foundation Models with tool support)"
        },
        {
            "reason": "3. Tools not properly passed to agent",
            "check": "Are retrieval_tools properly configured?",
            "solution": "Verify tools are LangChain-compatible and properly initialized"
        },
        {
            "reason": "4. Agent configuration issue",
            "check": "Is create_react_agent configured correctly?",
            "solution": "Check if agent needs explicit tool binding"
        },
        {
            "reason": "5. Query is too simple",
            "check": "Does the query actually need retrieval?",
            "solution": "Test with a query that clearly needs documentation lookup"
        },
        {
            "reason": "6. LLM decides tools aren't needed",
            "check": "Is the LLM choosing not to use tools?",
            "solution": "Make system prompt more directive about tool usage"
        }
    ]
    
    print("COMMON REASONS FOR EMPTY TOOL CALLS:\n")
    for item in reasons:
        print(f"❓ {item['reason']}")
        print(f"   Check: {item['check']}")
        print(f"   Solution: {item['solution']}")
        print()
    
    print("=" * 80)
    print("DIAGNOSTIC STEPS")
    print("=" * 80)
    print()
    
    print("Step 1: Check the console logs from your query")
    print("-" * 80)
    print("Look for this section:")
    print("""
--- Step 3a: Analyzing Agent Response ---
Total messages in agent result: X

Message 0: SystemMessage
Message 1: HumanMessage
Message 2: AIMessage
  ✓ Has tool_calls attribute: 2 calls    <-- LOOK FOR THIS
    - Tool: generic_doc_retriever
    - Tool: api_docs_retriever
""")
    print()
    print("If you see '✓ Has tool_calls attribute', tools ARE being called!")
    print("If you DON'T see this, tools are NOT being called.")
    print()
    
    print("Step 2: Check what messages the agent returned")
    print("-" * 80)
    print("Count the messages:")
    print("  - 2-3 messages = No tools used (System + Human + AI response)")
    print("  - 5+ messages = Tools likely used (includes ToolMessage objects)")
    print()
    
    print("Step 3: Verify your tools are configured")
    print("-" * 80)
    print("Check that you see:")
    print("  'Calling agent with 4 available tools'  <-- Should be > 0")
    print()
    
    print("Step 4: Test with a complex query")
    print("-" * 80)
    print("Try these queries that REQUIRE retrieval:")
    print("  - 'What are the specific API parameters for creating a Databricks cluster?'")
    print("  - 'Show me the exact syntax for Delta Lake MERGE operations'")
    print("  - 'What are the configuration options for MLflow autologging?'")
    print()
    
    print("=" * 80)
    print("QUICK FIXES TO TRY")
    print("=" * 80)
    print()
    
    fixes = [
        {
            "fix": "Fix 1: Update System Prompt",
            "code": """
# In conversational_agent.py, update system_prompt to:
self.system_prompt = \"\"\"You are a Databricks documentation expert assistant.

CRITICAL: You MUST use the retrieval tools for EVERY query to ensure accuracy.

INSTRUCTIONS:
1. ALWAYS call the appropriate retrieval tool(s) first
2. Use the retrieved information to answer the question
3. Include citations with URLs from the retrieved documents
4. Format citations as numbered references [1], [2], etc.
...
\"\"\"
"""
        },
        {
            "fix": "Fix 2: Bind Tools to LLM",
            "code": """
# In conversational_agent.py, update agent creation:
self.llm_with_tools = self.llm.bind_tools(self.retrieval_tools)
self.agent = create_react_agent(
    self.llm_with_tools,
    self.retrieval_tools
)
"""
        },
        {
            "fix": "Fix 3: Use Different Agent Type",
            "code": """
# Try using a different agent configuration:
from langgraph.prebuilt import create_react_agent

self.agent = create_react_agent(
    self.llm,
    self.retrieval_tools,
    state_modifier=self.system_prompt  # Pass as state_modifier
)
"""
        },
        {
            "fix": "Fix 4: Verify LLM Supports Tools",
            "code": """
# Test if your LLM supports function calling:
from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(endpoint="databricks-qwen3-next-80b-a3b-instruct")

# Check if it has bind_tools method
if hasattr(llm, 'bind_tools'):
    print("✅ LLM supports tool binding")
else:
    print("❌ LLM may not support function calling")
"""
        }
    ]
    
    for item in fixes:
        print(f"🔧 {item['fix']}")
        print(item['code'])
        print()
    
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Run your agent query and check the console logs")
    print("2. Look for the 'Step 3a: Analyzing Agent Response' section")
    print("3. Count the messages - if only 2-3, tools aren't being called")
    print("4. Try Fix 1 (update system prompt) first")
    print("5. If that doesn't work, try Fix 2 (bind tools to LLM)")
    print("6. Share the console output from 'Step 3a' for more help")
    print()
    print("=" * 80)


if __name__ == "__main__":
    diagnose_tool_calls()
