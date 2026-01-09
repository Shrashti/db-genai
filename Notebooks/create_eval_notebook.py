import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Wanderbricks Agent: Interpretation & Mapping Evaluation\n",
    "\n",
    "This notebook evaluates how accurately the agent translates Natural Language queries into:\n",
    "1.  **Tool Calls** (Interpretation: Did it pick the right tool? Did it extract the right parameters?)\n",
    "2.  **SQL Generation** (Mapping: Do those parameters produce the correct Unity Catalog SQL?)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install langchain mlflow databricks-sdk langchain-community"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "dbutils.library.restartPython()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import pandas as pd\n",
    "\n",
    "# Ensure modules are loadable\n",
    "sys.path.append(os.getcwd())\n",
    "\n",
    "from wanderbricks.agent_core import create_agent\n",
    "from wanderbricks.tools import _generate_search_sql, _generate_amenities_sql, _generate_booking_sql\n",
    "\n",
    "agent = create_agent()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define Test Cases\n",
    "# Format: (User Query, Expected Tool Name, Expected Params (partial))\n",
    "test_cases = [\n",
    "    (\n",
    "        \"Find me a villa in Bali under $500\", \n",
    "        \"search_properties\", \n",
    "        {\"location\": \"Bali\", \"max_price\": 500, \"property_type\": \"villa\"}\n",
    "    ),\n",
    "    (\n",
    "        \"Does prop_101 have a pool?\", \n",
    "        \"get_amenities\", \n",
    "        {\"property_id\": \"prop_101\"}\n",
    "    ),\n",
    "    (\n",
    "        \"Book prop_101 for user@example.com from 2024-01-01 to 2024-01-05\", \n",
    "        \"book_property\", \n",
    "        {\"property_id\": \"prop_101\", \"user_email\": \"user@example.com\"}\n",
    "    )\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = []\n",
    "\n",
    "print(\"Starting Evaluation...\\n\")\n",
    "\n",
    "for query, expected_tool, expected_params in test_cases:\n",
    "    print(f\"🔹 Query: {query}\")\n",
    "    \n",
    "    # Run Agent and capture intermediate steps\n",
    "    try:\n",
    "        response = agent.agent_executor.invoke({\"input\": query})\n",
    "        steps = response.get(\"intermediate_steps\", [])\n",
    "        \n",
    "        if not steps:\n",
    "            print(\"   ❌ No tool called.\")\n",
    "            results.append({\"query\": query, \"status\": \"FAILED\", \"reason\": \"No tool called\"})\n",
    "            continue\n",
    "            \n",
    "        # The first step typically contains the tool call\n",
    "        # step is a tuple: (tool_agent_action, tool_output)\n",
    "        action, output = steps[0]\n",
    "        tool_name = action.tool\n",
    "        tool_input = action.tool_input\n",
    "        \n",
    "        # Check Tool Selection\n",
    "        tool_match = (tool_name == expected_tool)\n",
    "        \n",
    "        # Check Parameter Extraction\n",
    "        param_match = True\n",
    "        for k, v in expected_params.items():\n",
    "            if k not in tool_input or tool_input[k] != v:\n",
    "                # Allow loose match for numbers (500 vs 500.0) or case-insensitive string if needed\n",
    "                if str(tool_input.get(k)) != str(v):\n",
    "                    param_match = False\n",
    "                    print(f\"   ⚠️ Param mismatch: Expected {k}={v}, got {tool_input.get(k)}\")\n",
    "\n",
    "        # Check SQL Generation (Mapping)\n",
    "        sql_generated = \"\"\n",
    "        if tool_name == \"search_properties\":\n",
    "            sql_generated = _generate_search_sql(**tool_input)\n",
    "        elif tool_name == \"get_amenities\":\n",
    "            sql_generated = _generate_amenities_sql(**tool_input)\n",
    "        elif tool_name == \"book_property\":\n",
    "            sql_generated = _generate_booking_sql(booking_id=\"TST\", **tool_input)\n",
    "            \n",
    "        print(f\"   ✅ Tool: {tool_name}\")\n",
    "        print(f\"   ✅ Params: {tool_input}\")\n",
    "        print(f\"   📝 Generated SQL: \\n{sql_generated}\")\n",
    "        \n",
    "        results.append({\n",
    "            \"query\": query,\n",
    "            \"tool_selected\": tool_name,\n",
    "            \"tool_correct\": tool_match,\n",
    "            \"params_extracted\": tool_input,\n",
    "            \"params_correct\": param_match,\n",
    "            \"generated_sql\": sql_generated\n",
    "        })\n",
    "        \n",
    "    except Exception as e:\n",
    "        print(f\"   ❌ Error: {e}\")\n",
    "        results.append({\"query\": query, \"status\": \"ERROR\", \"reason\": str(e)})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Summary Report\n",
    "df = pd.DataFrame(results)\n",
    "display(df)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

file_path = "Notebooks/06-Agent-Evaluation.ipynb"
with open(file_path, "w") as f:
    json.dump(notebook_content, f, indent=1)

print(f"Created {file_path}")
