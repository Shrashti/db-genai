"""
Conversational Agent with Guardrails for Databricks Documentation Q&A

This module provides a production-ready conversational agent that:
- Validates queries are Databricks-related
- Maintains conversation history
- Provides citations from documentation
- Tracks metrics and performance
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import mlflow
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from databricks_langchain import ChatDatabricks

from guardrails import (
    InputGuardrail,
    OutputGuardrail,
    RejectionHandler,
    GuardrailMetrics,
    GuardrailResult
)


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    timestamp: str
    user_query: str
    agent_response: str
    tool_calls: List[str] = field(default_factory=list)
    guardrail_input: Optional[GuardrailResult] = None
    guardrail_output: Optional[GuardrailResult] = None
    was_rejected: bool = False


class ConversationMemory:
    """
    Manages conversation history for multi-turn conversations.
    
    Supports both in-memory and persistent storage.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of turns to keep in memory
        """
        self.max_history = max_history
        self.conversations: Dict[str, List[ConversationTurn]] = {}
    
    def add_turn(
        self,
        conversation_id: str,
        turn: ConversationTurn
    ):
        """Add a conversation turn."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append(turn)
        
        # Trim history if needed
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = (
                self.conversations[conversation_id][-self.max_history:]
            )
    
    def get_history(
        self,
        conversation_id: str,
        num_turns: Optional[int] = None
    ) -> List[ConversationTurn]:
        """Get conversation history."""
        history = self.conversations.get(conversation_id, [])
        
        if num_turns is not None:
            history = history[-num_turns:]
        
        return history
    
    def get_messages_for_llm(
        self,
        conversation_id: str,
        num_turns: int = 5
    ) -> List[Dict]:
        """
        Get conversation history formatted for LLM.
        
        Returns list of message dicts with role and content.
        """
        history = self.get_history(conversation_id, num_turns)
        messages = []
        
        for turn in history:
            if not turn.was_rejected:
                messages.append({
                    "role": "user",
                    "content": turn.user_query
                })
                messages.append({
                    "role": "assistant",
                    "content": turn.agent_response
                })
        
        return messages
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_all_conversations(self) -> List[str]:
        """Get list of all conversation IDs."""
        return list(self.conversations.keys())


class DatabricksDocAgent:
    """
    Production-ready conversational agent for Databricks documentation Q&A.
    
    Features:
    - Input/output guardrails
    - Conversation memory
    - Citation support
    - MLflow tracking
    - Rejection handling
    """
    
    def __init__(
        self,
        retrieval_tools: List[Any],
        llm_endpoint: str = "databricks-qwen3-next-80b-a3b-instruct",
        guardrail_strictness: str = "moderate",
        enable_input_guardrail: bool = True,
        enable_output_guardrail: bool = True,
        max_conversation_history: int = 10,
        log_to_mlflow: bool = True
    ):
        """
        Initialize the Databricks documentation agent.
        
        Args:
            retrieval_tools: List of retrieval tools (from vector search)
            llm_endpoint: Databricks LLM endpoint
            guardrail_strictness: "strict", "moderate", or "lenient"
            enable_input_guardrail: Whether to validate input queries
            enable_output_guardrail: Whether to validate output responses
            max_conversation_history: Max turns to keep in memory
            log_to_mlflow: Whether to log to MLflow
        """
        self.llm = ChatDatabricks(endpoint=llm_endpoint)
        self.retrieval_tools = retrieval_tools
        self.log_to_mlflow = log_to_mlflow
        
        # Initialize guardrails
        self.enable_input_guardrail = enable_input_guardrail
        self.enable_output_guardrail = enable_output_guardrail
        
        if enable_input_guardrail:
            self.input_guardrail = InputGuardrail(
                llm_endpoint=llm_endpoint,
                strictness=guardrail_strictness,
                log_to_mlflow=log_to_mlflow
            )
        
        if enable_output_guardrail:
            self.output_guardrail = OutputGuardrail(
                llm_endpoint=llm_endpoint,
                log_to_mlflow=log_to_mlflow
            )
        
        self.rejection_handler = RejectionHandler()
        self.memory = ConversationMemory(max_history=max_conversation_history)
        self.metrics = GuardrailMetrics()
        
        # System prompt for the agent - Enhanced for better tool usage
        self.system_prompt = """You are a Databricks documentation expert assistant.

🔴 CRITICAL REQUIREMENT: You MUST use the retrieval tools for EVERY query to ensure accuracy.
Never answer from memory alone - always retrieve current documentation first.

MANDATORY WORKFLOW FOR EVERY QUERY:
1. ALWAYS call at least one retrieval tool FIRST (choose the most appropriate)
2. Wait for and read the retrieved documentation carefully
3. Formulate your answer based ONLY on the retrieved information
4. ALWAYS include citations with URLs for every piece of information

TOOL SELECTION GUIDE:
- generic_doc_retriever: General Databricks concepts, features, overviews, getting started
- api_docs_retriever: API references, method signatures, parameters, SDK documentation
- tutorial_retriever: How-to guides, step-by-step instructions, tutorials
- code_examples_retriever: Code samples, implementation examples, working code

RESPONSE REQUIREMENTS:
1. Use retrieved information to answer clearly and accurately
2. Include inline citations [1], [2] for every fact or claim
3. List all source URLs at the end under "Sources:"
4. If retrieved docs don't have the answer, say so explicitly
5. Stay focused on Databricks-related topics only

Response format:
<Your detailed answer with inline citations [1], [2]>

Sources:
[1] <URL from first source>
[2] <URL from second source>

⚠️ IMPORTANT: You have access to retrieval tools. Use them BEFORE answering!"""
        
        # CRITICAL FIX: Bind tools to LLM before creating agent
        # This is required for ChatDatabricks to know about and invoke tools
        print(f"🔧 Binding {len(self.retrieval_tools)} tools to LLM...")
        llm_with_tools = self.llm.bind_tools(self.retrieval_tools)
        print(f"✅ Tools successfully bound to LLM")
        
        # Create the agent with LLM that has tools bound
        # Note: System prompt will be injected in the query method
        self.agent = create_react_agent(
            llm_with_tools,  # Use LLM with bound tools (CRITICAL!)
            self.retrieval_tools
        )
        
        print(f"✅ Agent initialized with {len(self.retrieval_tools)} tools")
        print(f"   System prompt length: {len(self.system_prompt)} chars")
        print(f"   LLM endpoint: {llm_endpoint}")
    
    def query(
        self,
        user_query: str,
        conversation_id: Optional[str] = None,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query with guardrails and conversation memory.
        
        Args:
            user_query: User's question
            conversation_id: Optional conversation ID for multi-turn
            include_history: Whether to include conversation history
            
        Returns:
            Dict with response, metadata, and guardrail results
        """
        start_time = datetime.now()
        
        # Generate conversation ID if not provided
        if conversation_id is None:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*80}")
        print(f"QUERY PROCESSING START")
        print(f"{'='*80}")
        print(f"Conversation ID: {conversation_id}")
        print(f"User Query: {user_query}")
        print(f"Include History: {include_history}")
        
        # Step 1: Input Guardrail
        print(f"\n--- Step 1: Input Guardrail Validation ---")
        if self.enable_input_guardrail:
            input_result = self.input_guardrail.validate(user_query)
            self.metrics.log_input_check(user_query, input_result)
            
            print(f"Input Valid: {input_result.is_valid}")
            print(f"Confidence: {input_result.confidence}")
            print(f"Category: {input_result.category}")
            
            if not input_result.is_valid:
                print(f"❌ Query REJECTED: {input_result.reason}")
                
                # Generate rejection response
                rejection_message = self.rejection_handler.generate_rejection(
                    user_query,
                    input_result
                )
                
                self.metrics.log_rejection(user_query, input_result.reason)
                
                # Record rejected turn
                turn = ConversationTurn(
                    timestamp=datetime.now().isoformat(),
                    user_query=user_query,
                    agent_response=rejection_message,
                    guardrail_input=input_result,
                    was_rejected=True
                )
                self.memory.add_turn(conversation_id, turn)
                
                return {
                    "response": rejection_message,
                    "conversation_id": conversation_id,
                    "was_rejected": True,
                    "guardrail_input": input_result,
                    "metadata": {
                        "latency_ms": (datetime.now() - start_time).total_seconds() * 1000,
                        "tool_calls": 0,
                        "tools_used": [],
                        "conversation_turn": len(self.memory.get_history(conversation_id)),
                        "trace": []
                    }
                }
            else:
                print(f"✅ Query ACCEPTED")
        else:
            print(f"Input guardrail disabled")
            input_result = None
        
        # Step 2: Build messages with conversation history
        print(f"\n--- Step 2: Building Message Context ---")
        
        # Inject system prompt as the first message
        messages = [SystemMessage(content=self.system_prompt)]
        print(f"Added system prompt ({len(self.system_prompt)} chars)")
        
        if include_history and conversation_id:
            history_messages = self.memory.get_messages_for_llm(conversation_id)
            print(f"Including {len(history_messages)} historical messages")
            for msg in history_messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        else:
            print(f"No conversation history included")
        
        # Add current query
        messages.append(HumanMessage(content=user_query))
        print(f"Total messages to agent: {len(messages)} (including system prompt)")
        
        # Step 3: Invoke agent
        print(f"\n--- Step 3: Invoking Agent ---")
        print(f"Calling agent with {len(self.retrieval_tools)} available tools")
        agent_result = self.agent.invoke({"messages": messages})
        
        print(f"\n--- Step 3a: Analyzing Agent Response ---")
        print(f"Total messages in agent result: {len(agent_result['messages'])}")
        
        # Extract response and tool calls with detailed logging and tracing
        final_response = agent_result["messages"][-1].content
        
        tool_calls = []
        tool_call_details = []
        
        # New: Track traces for visualization
        # Map tool_call_id -> {name, args, output}
        trace_map = {}
        
        for i, msg in enumerate(agent_result["messages"]):
            msg_type = type(msg).__name__
            print(f"\nMessage {i}: {msg_type}")
            
            # Check for tool_calls attribute (AIMessage with tool calls)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"  ✓ Has tool_calls attribute: {len(msg.tool_calls)} calls")
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        # Dictionary format
                        t_id = tc.get('id', '')
                        t_name = tc.get('name', 'unknown')
                        t_args = tc.get('args', {})
                    else:
                        # ToolCall object format
                        t_id = getattr(tc, 'id', '')
                        t_name = getattr(tc, 'name', 'unknown')
                        t_args = getattr(tc, 'args', {})
                    
                    tool_calls.append(t_name)
                    tool_call_details.append({
                        'name': t_name,
                        'args': t_args,
                        'id': t_id
                    })
                    
                    # Initialize trace entry
                    if t_id:
                        trace_map[t_id] = {
                            "tool": t_name,
                            "inputs": t_args,
                            "outputs": None
                        }
                        
                    print(f"    - Tool: {t_name}")
            
            # Check for ToolMessage (tool execution results)
            if msg_type == 'ToolMessage':
                tool_name = getattr(msg, 'name', 'unknown_tool')
                tool_call_id = getattr(msg, 'tool_call_id', None)
                content = getattr(msg, 'content', '')
                
                print(f"  ✓ Tool execution result for: {tool_name}")
                if tool_name not in tool_calls:
                    tool_calls.append(tool_name)
                
                # Update trace with output
                if tool_call_id and tool_call_id in trace_map:
                    # Try to parse content if it's JSON
                    try:
                        parsed_content = json.loads(content)
                        trace_map[tool_call_id]["outputs"] = parsed_content
                    except:
                        # If not JSON, use raw string (truncated if too long)
                        trace_map[tool_call_id]["outputs"] = content[:500] + "..." if len(content) > 500 else content

        # Convert trace_map to list
        traces = list(trace_map.values())
        
        print(f"\n--- Tool Call Summary ---")
        print(f"Total tool calls: {len(tool_calls)}")
        print(f"Tools used: {list(set(tool_calls))}")
        if tool_call_details:
            print(f"\nDetailed tool calls:")
            for detail in tool_call_details:
                print(f"  - {detail['name']}: {detail['args']}")
        
        # Step 4: Output Guardrail
        print(f"\n--- Step 4: Output Guardrail Validation ---")
        if self.enable_output_guardrail:
            output_result = self.output_guardrail.validate(user_query, final_response)
            self.metrics.log_output_check(output_result)
            
            print(f"Output Valid: {output_result.is_valid}")
            
            if not output_result.is_valid:
                print(f"⚠️  Output validation failed: {output_result.reason}")
                # Response failed validation - could regenerate or reject
                # For now, we'll add a disclaimer
                final_response = (
                    f"{final_response}\n\n"
                    f"⚠️ Note: This response may contain inaccuracies. "
                    f"Please verify with official Databricks documentation."
                )
            else:
                print(f"✅ Output validated successfully")
        else:
            print(f"Output guardrail disabled")
            output_result = None
        
        # Step 5: Record conversation turn
        print(f"\n--- Step 5: Recording Conversation Turn ---")
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            agent_response=final_response,
            tool_calls=tool_calls,
            guardrail_input=input_result,
            guardrail_output=output_result,
            was_rejected=False
        )
        self.memory.add_turn(conversation_id, turn)
        print(f"Turn recorded with {len(tool_calls)} tool calls")
        
        # Step 6: Log to MLflow
        print(f"\n--- Step 6: MLflow Logging ---")
        if self.log_to_mlflow:
            print(f"Logging to MLflow...")
            self._log_to_mlflow(user_query, final_response, tool_calls, turn)
            print(f"✅ MLflow logging complete")
        else:
            print(f"MLflow logging disabled")
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        print(f"\n{'='*80}")
        print(f"QUERY RESULT")
        print(f"{'='*80}")
        print(f"")
        print(f"Was Rejected: {False}")
        print(f"Conversation ID: {conversation_id}")
        print(f"")
        print(f"Latency: {latency_ms:.0f}ms")
        print(f"Tool Calls: {len(tool_calls)}")
        print(f"Tools Used: {list(set(tool_calls))}")
        
        if input_result:
            print(f"\nInput Guardrail:")
            print(f"  Valid: {input_result.is_valid}")
            print(f"  Confidence: {input_result.confidence}")
            print(f"  Category: {input_result.category}")
        
        print(f"\n{'='*80}")
        print(f"RESPONSE")
        print(f"{'='*80}")
        print(final_response)
        
        return {
            "response": final_response,
            "conversation_id": conversation_id,
            "was_rejected": False,
            "guardrail_input": input_result,
            "guardrail_output": output_result,
            "metadata": {
                "latency_ms": latency_ms,
                "tool_calls": len(tool_calls),
                "tools_used": list(set(tool_calls)),
                "conversation_turn": len(self.memory.get_history(conversation_id)),
                "trace": traces
            }
        }
    
    def _log_to_mlflow(
        self,
        query: str,
        response: str,
        tool_calls: List[str],
        turn: ConversationTurn
    ):
        """Log query and response to MLflow."""
        try:
            print(f"  - Logging query (truncated): {query[:100]}")
            mlflow.log_param("query", query[:100])  # Truncate long queries
            
            print(f"  - Logging tool call count: {len(tool_calls)}")
            mlflow.log_metric("num_tool_calls", len(tool_calls))
            
            if tool_calls:
                tools_used_str = ", ".join(set(tool_calls))
                print(f"  - Logging tools used: {tools_used_str}")
                mlflow.log_param("tools_used", tools_used_str)
                
                # Log individual tool calls
                for i, tool in enumerate(tool_calls):
                    mlflow.log_param(f"tool_call_{i}", tool)
            else:
                print(f"  - No tools were called")
                mlflow.log_param("tools_used", "none")
            
            print(f"  - Logging response text ({len(response)} chars)")
            mlflow.log_text(response, "response.txt")
            
            if turn.guardrail_input:
                input_valid = 1 if turn.guardrail_input.is_valid else 0
                print(f"  - Logging input validation: {input_valid}")
                mlflow.log_metric("input_valid", input_valid)
                mlflow.log_param("input_category", turn.guardrail_input.category or "unknown")
                mlflow.log_metric("input_confidence", turn.guardrail_input.confidence)
            
            if turn.guardrail_output:
                output_valid = 1 if turn.guardrail_output.is_valid else 0
                print(f"  - Logging output validation: {output_valid}")
                mlflow.log_metric("output_valid", output_valid)
                
        except Exception as e:
            print(f"  ⚠️  MLflow logging error: {str(e)}")
            # Don't fail if MLflow logging fails
    
    def get_conversation_history(
        self,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get formatted conversation history.
        
        Returns list of turns with user queries and agent responses.
        """
        turns = self.memory.get_history(conversation_id)
        
        return [
            {
                "timestamp": turn.timestamp,
                "user": turn.user_query,
                "assistant": turn.agent_response,
                "was_rejected": turn.was_rejected,
                "tool_calls": turn.tool_calls
            }
            for turn in turns
        ]
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation history."""
        self.memory.clear_conversation(conversation_id)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of guardrail metrics."""
        return self.metrics.get_summary()
    
    def export_metrics_to_mlflow(self):
        """Export all metrics to MLflow."""
        self.metrics.export_to_mlflow()


def create_databricks_agent(
    vector_search_endpoint: str,
    vector_search_index: str,
    llm_endpoint: str = "databricks-qwen3-next-80b-a3b-instruct",
    guardrail_strictness: str = "moderate",
    **kwargs
) -> DatabricksDocAgent:
    """
    Factory function to create a Databricks documentation agent.
    
    Args:
        vector_search_endpoint: Databricks vector search endpoint name
        vector_search_index: Vector search index name
        llm_endpoint: LLM endpoint for generation
        guardrail_strictness: "strict", "moderate", or "lenient"
        **kwargs: Additional arguments for DatabricksDocAgent
        
    Returns:
        Configured DatabricksDocAgent instance
    """
    from databricks_langchain import VectorSearchRetrieverTool
    
    # Create retrieval tools with enhanced descriptions
    # Tool descriptions are critical - they guide the LLM on when to use each tool
    generic_retriever = VectorSearchRetrieverTool(
        endpoint_name=vector_search_endpoint,
        index_name=vector_search_index,
        columns=["chunk_id", "doc_id", "text", "url"],
        tool_name="generic_doc_retriever",
        tool_description=(
            "Search general Databricks documentation for concepts, features, overviews, "
            "getting started guides, and product information. Use this for broad questions "
            "about what Databricks is, how features work, or general explanations."
        ),
        filters={"doc_type": "general"},
        num_results=5,
        disable_notice=True
    )
    
    api_retriever = VectorSearchRetrieverTool(
        endpoint_name=vector_search_endpoint,
        index_name=vector_search_index,
        columns=["chunk_id", "doc_id", "text", "url", "doc_type"],
        tool_name="api_docs_retriever",
        tool_description=(
            "Search API reference documentation for method signatures, parameters, "
            "return types, SDK documentation, and API usage. Use this when users ask "
            "about specific APIs, methods, classes, or programmatic interfaces."
        ),
        filters={"doc_type": "api_reference"},
        num_results=5,
        disable_notice=True
    )
    
    tutorial_retriever = VectorSearchRetrieverTool(
        endpoint_name=vector_search_endpoint,
        index_name=vector_search_index,
        columns=["chunk_id", "doc_id", "text", "url", "doc_type"],
        tool_name="tutorial_retriever",
        tool_description=(
            "Search tutorial and how-to guides for step-by-step instructions, "
            "walkthroughs, and practical guides. Use this when users ask 'how to' "
            "do something or need procedural guidance."
        ),
        filters={"doc_type": "tutorial"},
        num_results=5,
        disable_notice=True
    )
    
    code_retriever = VectorSearchRetrieverTool(
        endpoint_name=vector_search_endpoint,
        index_name=vector_search_index,
        columns=["chunk_id", "doc_id", "text", "url", "doc_type", "has_code"],
        tool_name="code_examples_retriever",
        tool_description=(
            "Search documentation with code examples, sample implementations, "
            "and working code snippets. Use this when users need code examples, "
            "implementation patterns, or want to see how to use something in code."
        ),
        filters={"has_code": "true"},
        num_results=5,
        disable_notice=True
    )
    
    retrieval_tools = [
        generic_retriever,
        api_retriever,
        tutorial_retriever,
        code_retriever
    ]
    
    return DatabricksDocAgent(
        retrieval_tools=retrieval_tools,
        llm_endpoint=llm_endpoint,
        guardrail_strictness=guardrail_strictness,
        **kwargs
    )
