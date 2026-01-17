"""
Wanderbricks Agent Core with Human Review Workflow

This module provides the main agent implementation with support for:
- Dynamic NL-to-SQL query generation
- Human-in-the-loop review workflow
- LangGraph-based state management
"""

import mlflow
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, field

try:
    from langchain.agents import create_agent
    LANGGRAPH_AVAILABLE = False  # Using legacy create_agent
except ImportError:
    LANGGRAPH_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    pass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from langchain_community.chat_models import ChatDatabricks
except ImportError:
    ChatDatabricks = None

from .tools import get_wanderbricks_tools, get_fast_path_tools, set_human_review_enabled
from .guardrails import TravelTopicGuardrail


# System prompt with schema awareness
SYSTEM_PROMPT = """You are a helpful travel assistant for Wanderbricks, a premium vacation rental platform.

You have access to the following tools:

1. **search_properties**: Search for properties by location, price, and type.
   - Use for simple property searches with known parameters.

2. **get_amenities**: Get amenities for a specific property.
   - Use when user asks about what a property offers.

3. **book_property**: Create a booking for a property.
   - Use when user confirms they want to book.

4. **natural_language_query**: Execute any complex query against our database.
   - Use for complex questions that don't fit the above tools.
   - Examples: "top-rated hosts", "properties with pools", "booking statistics"
   - This tool may request human review before execution.

5. **get_table_schema**: Get database schema information.
   - Use when you need to understand the data structure.

GUIDELINES:
- For simple property searches, prefer search_properties (faster).
- For complex analytical queries, use natural_language_query.
- If a query involves JOINs or aggregations, use natural_language_query.
- Always explain what data you found in a friendly, helpful manner.
- If a user asks about topics unrelated to travel, politely refuse.
"""


@dataclass
class AgentState:
    """State for the agent workflow."""
    messages: List[Any] = field(default_factory=list)
    current_query: Optional[str] = None
    pending_review: bool = False
    review_request_id: Optional[str] = None
    iteration: int = 0


class WanderbricksAgent:
    """
    Wanderbricks Agent with Human Review Workflow.
    
    This agent supports:
    - Tool-calling for property search, amenities, and bookings
    - Dynamic NL-to-SQL for complex queries
    - Human-in-the-loop review for generated SQL
    """
    
    def __init__(self, 
                 model_name: str = "databricks-meta-llama-3-70b-instruct",
                 llm=None,
                 human_review_enabled: bool = True,
                 use_fast_path_only: bool = False):
        """
        Initialize the Wanderbricks agent.
        
        Args:
            model_name: Name of the Databricks model endpoint.
            llm: Pre-configured LLM instance (optional).
            human_review_enabled: Whether to enable human review workflow.
            use_fast_path_only: If True, only use fast-path tools (no dynamic SQL).
        """
        self.model_name = model_name
        self.human_review_enabled = human_review_enabled
        self.use_fast_path_only = use_fast_path_only
        
        # Configure human review
        set_human_review_enabled(human_review_enabled)
        
        # Get tools
        if use_fast_path_only:
            self.tools = get_fast_path_tools()
        else:
            self.tools = get_wanderbricks_tools(include_dynamic=True)
        
        # Configure LLM
        if llm:
            self.llm = llm
        elif ChatDatabricks:
            self.llm = ChatDatabricks(endpoint=model_name, temperature=0.1)
        else:
            raise RuntimeError("No LLM available. Install langchain-community or provide an LLM.")
        
        # Initialize guardrail
        self.guardrail = TravelTopicGuardrail()
        
        # Build the agent
        self.agent = self._build_agent()
        
    def _build_agent(self):
        """Build the agent based on available libraries."""
        if LANGGRAPH_AVAILABLE:
            return self._build_langgraph_agent()
        else:
            return self._build_legacy_agent()
    
    def _build_legacy_agent(self):
        """Build agent using legacy create_agent API."""
        try:
            from langchain.agents import create_agent
            return create_agent(self.llm, self.tools, system_prompt=SYSTEM_PROMPT)
        except ImportError:
            # Fallback to simple tool-calling
            return self._build_simple_agent()
    
    def _build_simple_agent(self):
        """Build a simple agent without LangChain agent framework."""
        # This is a minimal implementation for environments without full LangChain
        class SimpleAgent:
            def __init__(self, llm, tools, system_prompt):
                self.llm = llm
                self.tools = {t.name: t for t in tools}
                self.system_prompt = system_prompt
                
            def invoke(self, input_dict):
                messages = input_dict.get("messages", [])
                # Simple invocation - just pass to LLM with tools
                response = self.llm.invoke(messages)
                return {"messages": messages + [response]}
        
        return SimpleAgent(self.llm, self.tools, SYSTEM_PROMPT)
    
    def _build_langgraph_agent(self):
        """Build agent using LangGraph for advanced workflow control."""
        # Define state schema
        from typing import TypedDict, Annotated
        from langgraph.graph.message import add_messages
        
        class State(TypedDict):
            messages: Annotated[list, add_messages]
            pending_review: bool
            review_data: Optional[Dict]
        
        # Create the graph
        graph = StateGraph(State)
        
        # Define nodes
        def call_model(state: State):
            """Call the LLM."""
            messages = state["messages"]
            response = self.llm.invoke(messages)
            return {"messages": [response], "pending_review": False}
        
        def should_continue(state: State):
            """Decide whether to continue or end."""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END
        
        # Add nodes
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        
        return graph.compile()
    
    def run(self, input_query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Run the agent with a user query.
        
        Args:
            input_query: User's natural language query.
            context: Optional context (e.g., conversation history).
            
        Returns:
            Agent's response as a string.
        """
        # 1. Guardrail Check
        check_result = self.guardrail.check(input_query)
        if not check_result["allowed"]:
            return f"I cannot answer that. {check_result['reason']}"
            
        # 2. Agent Execution
        try:
            # Prepare messages
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=input_query)
            ]
            
            # Include context if provided
            if context and "history" in context:
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + context["history"] + [HumanMessage(content=input_query)]
            
            # Invoke the agent
            result = self.agent.invoke({"messages": messages})
            
            # Extract response
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                return str(last_message)
            return str(result)
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def run_with_review(self, input_query: str) -> Dict[str, Any]:
        """
        Run the agent with explicit review workflow tracking.
        
        Returns a dict with:
        - response: The agent's response
        - pending_review: Whether a review is pending
        - review_request: Review request details if pending
        """
        # Check if there's a pending review from a previous query
        from .tools import get_review_node
        review_node = get_review_node()
        pending = review_node.get_pending_reviews()
        
        if pending:
            return {
                "response": "There is a pending SQL review. Please complete it first.",
                "pending_review": True,
                "review_requests": [r.to_dict() for r in pending]
            }
        
        # Run the query
        response = self.run(input_query)
        
        # Check for new pending reviews
        pending = review_node.get_pending_reviews()
        
        return {
            "response": response,
            "pending_review": len(pending) > 0,
            "review_requests": [r.to_dict() for r in pending] if pending else None
        }


class WanderbricksAgentWithHistory(WanderbricksAgent):
    """
    Extended agent with conversation history management.
    """
    
    def __init__(self, *args, max_history: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history: List[Any] = []
        self.max_history = max_history
    
    def run(self, input_query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Run with automatic history management."""
        # Add context from history
        context = context or {}
        context["history"] = self.conversation_history[-self.max_history:]
        
        # Get response
        response = super().run(input_query, context)
        
        # Update history
        self.conversation_history.append(HumanMessage(content=input_query))
        self.conversation_history.append(AIMessage(content=response))
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# Factory functions for different agent configurations

def create_wanderbricks_agent(
    model_name: str = "databricks-meta-llama-3-70b-instruct",
    human_review: bool = True,
    fast_path_only: bool = False
) -> WanderbricksAgent:
    """
    Create a Wanderbricks agent with the specified configuration.
    
    Args:
        model_name: Databricks model endpoint name.
        human_review: Enable human review workflow.
        fast_path_only: Only use fast-path tools (no dynamic SQL).
        
    Returns:
        Configured WanderbricksAgent instance.
    """
    return WanderbricksAgent(
        model_name=model_name,
        human_review_enabled=human_review,
        use_fast_path_only=fast_path_only
    )


def create_agent_with_history(
    model_name: str = "databricks-meta-llama-3-70b-instruct",
    human_review: bool = True,
    max_history: int = 10
) -> WanderbricksAgentWithHistory:
    """
    Create a Wanderbricks agent with conversation history.
    
    Args:
        model_name: Databricks model endpoint name.
        human_review: Enable human review workflow.
        max_history: Maximum conversation turns to remember.
        
    Returns:
        Configured WanderbricksAgentWithHistory instance.
    """
    return WanderbricksAgentWithHistory(
        model_name=model_name,
        human_review_enabled=human_review,
        max_history=max_history
    )


# Backward compatibility
def create_agent():
    """Create a default agent (backward compatible)."""
    return WanderbricksAgent()
