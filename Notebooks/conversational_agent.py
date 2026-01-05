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
        
        # System prompt for the agent
        self.system_prompt = """You are a Databricks documentation expert assistant.

IMPORTANT INSTRUCTIONS:
1. Use the retrieval tools to find accurate information from Databricks documentation
2. Answer questions clearly, concisely, and accurately
3. ALWAYS include citations with URLs for every piece of information you provide
4. Format citations as numbered references [1], [2], etc.
5. List all source URLs at the end of your response
6. If you're unsure or don't have enough information, say so
7. Stay focused on Databricks-related topics only

Response format:
<Your detailed answer with inline citations [1], [2]>

Sources:
[1] <URL from first source>
[2] <URL from second source>"""
        
        # Create the agent
        self.agent = create_react_agent(
            self.llm,
            self.retrieval_tools
        )
    
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
        
        # Step 1: Input Guardrail
        if self.enable_input_guardrail:
            input_result = self.input_guardrail.validate(user_query)
            self.metrics.log_input_check(user_query, input_result)
            
            if not input_result.is_valid:
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
                        "tool_calls": 0
                    }
                }
        else:
            input_result = None
        
        # Step 2: Build messages with conversation history
        messages = [SystemMessage(content=self.system_prompt)]
        
        if include_history and conversation_id:
            history_messages = self.memory.get_messages_for_llm(conversation_id)
            for msg in history_messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add current query
        messages.append(HumanMessage(content=user_query))
        
        # Step 3: Invoke agent
        agent_result = self.agent.invoke({"messages": messages})
        
        # Extract response and tool calls
        final_response = agent_result["messages"][-1].content
        
        tool_calls = []
        for msg in agent_result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls.extend([tc['name'] for tc in msg.tool_calls])
        
        # Step 4: Output Guardrail
        if self.enable_output_guardrail:
            output_result = self.output_guardrail.validate(user_query, final_response)
            self.metrics.log_output_check(output_result)
            
            if not output_result.is_valid:
                # Response failed validation - could regenerate or reject
                # For now, we'll add a disclaimer
                final_response = (
                    f"{final_response}\n\n"
                    f"⚠️ Note: This response may contain inaccuracies. "
                    f"Please verify with official Databricks documentation."
                )
        else:
            output_result = None
        
        # Step 5: Record conversation turn
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
        
        # Step 6: Log to MLflow
        if self.log_to_mlflow:
            self._log_to_mlflow(user_query, final_response, tool_calls, turn)
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
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
                "conversation_turn": len(self.memory.get_history(conversation_id))
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
            mlflow.log_param("query", query[:100])  # Truncate long queries
            mlflow.log_metric("num_tool_calls", len(tool_calls))
            mlflow.log_param("tools_used", ", ".join(set(tool_calls)))
            mlflow.log_text(response, "response.txt")
            
            if turn.guardrail_input:
                mlflow.log_metric("input_valid", 1 if turn.guardrail_input.is_valid else 0)
            
            if turn.guardrail_output:
                mlflow.log_metric("output_valid", 1 if turn.guardrail_output.is_valid else 0)
        except Exception:
            pass  # Don't fail if MLflow logging fails
    
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
    
    # Create retrieval tools with different filters
    generic_retriever = VectorSearchRetrieverTool(
        endpoint_name=vector_search_endpoint,
        index_name=vector_search_index,
        columns=["chunk_id", "doc_id", "text", "url"],
        tool_name="generic_doc_retriever",
        tool_description="Retrieves generic Databricks documentation.",
        filters={"doc_type": "general"},
        num_results=5,
        disable_notice=True
    )
    
    api_retriever = VectorSearchRetrieverTool(
        endpoint_name=vector_search_endpoint,
        index_name=vector_search_index,
        columns=["chunk_id", "doc_id", "text", "url", "doc_type"],
        tool_name="api_docs_retriever",
        tool_description="Retrieves API reference documentation.",
        filters={"doc_type": "api_reference"},
        num_results=5,
        disable_notice=True
    )
    
    tutorial_retriever = VectorSearchRetrieverTool(
        endpoint_name=vector_search_endpoint,
        index_name=vector_search_index,
        columns=["chunk_id", "doc_id", "text", "url", "doc_type"],
        tool_name="tutorial_retriever",
        tool_description="Retrieves tutorial and how-to guides.",
        filters={"doc_type": "tutorial"},
        num_results=5,
        disable_notice=True
    )
    
    code_retriever = VectorSearchRetrieverTool(
        endpoint_name=vector_search_endpoint,
        index_name=vector_search_index,
        columns=["chunk_id", "doc_id", "text", "url", "doc_type", "has_code"],
        tool_name="code_examples_retriever",
        tool_description="Retrieves documentation with code examples.",
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
