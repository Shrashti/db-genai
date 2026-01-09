# Wanderbricks Agent Architecture

## Overview
The Wanderbricks agent is designed as a **Tool-Calling (ReAct) Agent**. This architecture allows the Large Language Model (LLM) to "reason" about a user's intent and structurally "act" by invoking specialized tools that interact with the Databricks Unity Catalog.

## High-Level Architecture

```mermaid
graph LR
    User[User Query] --> Guardrail{Guardrail Check}
    Guardrail -->|Unsafe/Off-topic| Block[Refusal Message]
    Guardrail -->|Safe| Agent[Agent Core (LLM)]
    Agent -->|Decide Tool| Tools[Tool Execution Layer]
    Tools -->|SQL Query| UC[(Unity Catalog)]
    UC -->|Data| Tools
    Tools -->|Result| Agent
    Agent -->|Final Answer| User
```

## Core Components

### 1. The Brain (LLM)
*   **Role**: Reasoning engine.
*   **Implementation**: Databricks Model Serving (e.g., DBRX or Llama 3).
*   **Function**: Parses natural language (e.g., "Find a villa in Bali under $500"), maps it to the appropriate tool (`search_properties`), and extracts parameters (`location='Bali'`, `max_price=500`).

### 2. The Orchestrator
*   **Role**: Control flow management.
*   **Implementation**: **LangChain**.
*   **Function**: Manages the "ReAct" loop:
    1.  **Thought**: What does the user want?
    2.  **Action**: Call the specific Python function (Tool).
    3.  **Observation**: Receive the tool validation/output.
    4.  **Response**: Synthesize the final answer.

### 3. The Tools (Action Layer)
Specialized Python functions that translate intent into SQL execution against Unity Catalog.
*   **`search_properties`**:
    *   *Input*: Location, Price, Type.
    *   *Action*: Generates `SELECT * FROM wanderbricks.properties WHERE ...`
*   **`get_amenities`**:
    *   *Input*: Property ID.
    *   *Action*: Performs a `JOIN` between `property_amenities` and `amenities` tables.
*   **`book_property`**:
    *   *Input*: Property ID, User Email, Dates.
    *   *Action*: Executes an `INSERT` statement into the `wanderbricks.bookings` table.

### 4. Safety Layer (Guardrails)
*   **Role**: Pre-emptive filtering.
*   **Implementation**: Custom `TravelTopicGuardrail` class.
*   **Function**: Checks input *before* it reaches the expensive LLM. Ensures queries are travel-related and free of malicious intent (e.g., "Ignore system prompt").

### 5. Observability
*   **Role**: Monitoring and Debugging.
*   **Implementation**: **MLflow Tracing**.
*   **Function**: Automatically logs every step of the chain (System Prompt -> Tool Call -> SQL Execution -> Response) to the MLflow Experiment Tracking server.

## Design Justification (Interview Talking Points)

*   **Why Tool-Calling?**: Allows the agent to be grounded in real-time enterprise data (SQL) rather than hallucinating from training data.
*   **Why Unity Catalog?**: Provides a unified governance layer. The agent uses standard JDBC/SQL interfaces, respecting existing ACLs and data lineage.
*   **Why Guardrails?**: Essential for enterprise deployment to prevent brand risk (answering inappropriate topics) and reduce costs (blockingjunk requests early).

## Deep Dive: How "NL-to-SQL" Works Here

It is important to clarify that this agent uses **Tool Calling (Parameter Extraction)** rather than raw "Text-to-SQL" generation. This approach is safer and more reliable for enterprise applications.

### The 3-Step Translation Process

#### Step 1: Contextual Resolution (The "Reasoning")
The LLM looks at the user's Natural Language and the **Conversation History**.
*   **User**: "Does *it* have a pool?"
*   **LLM Context**: Sees the previous search result returned `{'id': 'prop_101', 'name': 'Cozy Loft', ...}`.
*   **Reasoning**: The LLM infers that "it" refers to `prop_101`.

#### Step 2: Parameter Extraction (The "Function Call")
Instead of writing SQL code directly, the LLM decides to call a specific Python function with structured arguments.
*   **LLM Output**: `tool_call: get_amenities(property_id="prop_101")`
*   *Note: The LLM does not know about the JOINs or table schemas at this stage. It only knows it needs to "get amenities" for a specific ID.*

#### Step 3: Deterministic SQL Generation (The "execution")
The Python tool (`get_amenities` in `tools.py`) receives the arguments and inserts them into a **pre-written, optimized SQL template**.

```python
# specific tool logic in tools.py
def get_amenities(property_id):
    # The developer hardcoded this complex JOIN to ensure correctness
    sql = f"""
    SELECT a.name 
    FROM wanderbricks.property_amenities pa 
    JOIN wanderbricks.amenities a ON pa.amenity_id = a.amenity_id 
    WHERE pa.property_id = '{property_id}'
    """
    return spark.sql(sql)
```

### Why this approach?
1.  **Security**: Prevents "SQL Injection" via LLM hallucinations because the SQL structure is hardcoded.
2.  **Optimization**: You (the engineer) write the efficient `JOIN` logic once, rather than hoping the LLM figures out the schema relationships every time.
3.  **Accuracy**: The LLM only needs to extract parameters (like City Name or Price), which it is very good at, rather than writing complex SQL syntax.

## Behind the Scenes: How Extraction Works (The JSON Schema)

You might wonder: *How does the LLM know to extract "New York" as `location` and "300" as `max_price`?*

The secret is the **Tool Definition (JSON Schema)**.

### 1. The Interface (Python)
When we define the tool in Python using `@tool`, we provide type hints and a docstring.
```python
@tool
def search_properties(location: str, max_price: Optional[float] = None):
    """Search for properties in a specific city with a price limit."""
    ...
```

### 2. The Translation (JSON Schema)
LangChain translates this Python function into a JSON Schema that is sent to the LLM's system prompt. It looks like this:

```json
{
  "name": "search_properties",
  "description": "Search for properties in a specific city with a price limit.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city name to search in (e.g., 'New York')"
      },
      "max_price": {
        "type": "number",
        "description": "Optional maximum price per night"
      }
    },
    "required": ["location"]
  }
}
```

### 3. The LLM's Job (Slot Filling)
The LLM reads this schema and treats it like a form it needs to fill out.
*   **User Input**: "I want a place in **Paris** under **200** euros."
*   **LLM "Thinking"**: 
    *   I see a tool `search_properties`.
    *   The schema asks for `location` (string). found "Paris".
    *   The schema asks for `max_price` (number). found "200".
*   **Output**: Generates a structured JSON object: `{"location": "Paris", "max_price": 200}`.

This structured output is what our Python code receives to build the SQL query.
