import mlflow
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from .tools import get_wanderbricks_tools
from .guardrails import TravelTopicGuardrail

class WanderbricksAgent:
    def __init__(self, model_name="databricks-meta-llama-3-70b-instruct", llm=None):
        self.model_name = model_name
        self.tools = get_wanderbricks_tools()
        if llm:
            self.llm = llm
        else:
            self.llm = ChatDatabricks(endpoint=model_name, temperature=0.1)
        self.guardrail = TravelTopicGuardrail()
        self.agent_executor = self._build_agent()
        
    def _build_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful travel assistant for Wanderbricks. "
                       "You help users search for properties, check amenities, and make bookings. "
                       "Use the provided tools to fetch information. "
                       "If a user asks about topics unrelated to travel, politely refuse."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, return_intermediate_steps=True)
    
    def run(self, input_query: str):
        # 1. Guardrail Check
        check_result = self.guardrail.check(input_query)
        if not check_result["allowed"]:
            return f"I cannot answer that. {check_result['reason']}"
            
        # 2. Agent Execution
        try:
            # MLflow Tracing is handled automatically if enabled in the notebook
            result = self.agent_executor.invoke({"input": input_query})
            return result["output"]
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Example usage function to be called from notebook
def create_agent():
    return WanderbricksAgent()
