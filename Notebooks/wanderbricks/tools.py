from langchain.tools import tool
from typing import List, Optional, Dict, Any
import json
import random

# Import our custom modules
try:
    from .sql_generator import SQLGenerator, SQLResult, get_schema_context
    from .human_review import HumanReviewNode, ReviewResult, ReviewAction
    SQL_GENERATOR_AVAILABLE = True
except ImportError:
    SQL_GENERATOR_AVAILABLE = False
    print("[Warning] sql_generator or human_review modules not available. Using legacy mode.")

try:
    from .multi_step_sql import MultiStepSQLGenerator, SmartNLToSQL, QueryPlan
    MULTI_STEP_AVAILABLE = True
except ImportError:
    MULTI_STEP_AVAILABLE = False
    print("[Warning] multi_step_sql module not available.")

# Mock database simulation
# In a real scenario, these would be Delta/Unity Catalog tables
MOCK_PROPERTIES = [
    {
        "id": "prop_101",
        "name": "Cozy Downtown Loft",
        "location": "New York",
        "price_per_night": 150,
        "type": "apartment",
        "score": 4.5
    },
    {
        "id": "prop_102",
        "name": "Beachfront Villa",
        "location": "Miami",
        "price_per_night": 450,
        "type": "villa",
        "score": 4.8
    },
    {
        "id": "prop_103",
        "name": "Mountain Cabin",
        "location": "Denver",
        "price_per_night": 200,
        "type": "cabin",
        "score": 4.2
    },
    {
        "id": "prop_104",
        "name": "Luxury Condo",
        "location": "New York",
        "price_per_night": 350,
        "type": "apartment",
        "score": 4.7
    }
]

MOCK_AMENITIES = {
    "prop_101": ["wifi", "kitchen", "ac", "washer"],
    "prop_102": ["wifi", "pool", "beach_access", "ac", "kitchen"],
    "prop_103": ["wifi", "fireplace", "hiking_trials", "kitchen"],
    "prop_104": ["wifi", "gym", "doorman", "elevator", "ac"]
}

MOCK_BOOKINGS = []

# Global state for human review workflow
_review_node: Optional[HumanReviewNode] = None
_human_review_enabled: bool = True

def set_human_review_enabled(enabled: bool):
    """Enable or disable human review workflow."""
    global _human_review_enabled
    _human_review_enabled = enabled

def get_review_node() -> HumanReviewNode:
    """Get or create the human review node singleton."""
    global _review_node
    if _review_node is None:
        _review_node = HumanReviewNode(auto_approve_threshold=0.95)
    return _review_node

# --- Legacy SQL Generation Helpers (Kept for backward compatibility) ---

def _generate_search_sql(location: str, max_price: Optional[float] = None, property_type: Optional[str] = None) -> str:
    query = f"SELECT * FROM wanderbricks.properties p JOIN wanderbricks.cities c ON p.city_id = c.city_id WHERE c.city ILIKE '%{location}%'"
    if max_price:
        query += f" AND p.base_price <= {max_price}"
    if property_type:
        query += f" AND p.property_type ILIKE '{property_type}'"
    query += " LIMIT 100"
    return query.strip()

def _generate_amenities_sql(property_id: str) -> str:
    query = f"""
    SELECT a.name, a.category
    FROM wanderbricks.property_amenities pa 
    JOIN wanderbricks.amenities a ON pa.amenity_id = a.amenity_id 
    WHERE pa.property_id = '{property_id}'
    """
    return query.strip()

def _generate_booking_sql(booking_id: str, property_id: str, dates: str, user_email: str) -> str:
    query = f"""
    INSERT INTO wanderbricks.bookings (booking_id, property_id, user_email, dates, status, created_at)
    VALUES ('{booking_id}', '{property_id}', '{user_email}', '{dates}', 'confirmed', current_timestamp())
    """
    return query.strip()

import os

def _get_spark_session():
    """Try to get the global spark session if running in Databricks."""
    try:
        import IPython
        return IPython.get_ipython().user_ns.get('spark')
    except:
        return None

def _run_sql_query(query: str, as_json: bool = True):
    """
    Execute SQL using the best available method:
    1. Internal Spark Session (if in Databricks Notebook)
    2. Databricks SQL Connector (if local and configured)
    3. Failure (returns None to trigger fallback)
    """
    # Strategy 1: Internal Spark
    spark = _get_spark_session()
    if spark:
        try:
            print("[Info] Executing via Internal Spark Session")
            df = spark.sql(query)
            return df.toPandas().to_json(orient='records')
        except Exception as e:
            print(f"[Error] Spark SQL failed: {e}")
            return None

    # Strategy 2: Databricks SQL Connector (Remote)
    # Requires: pip install databricks-sql-connector
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    
    if host and token and http_path:
        try:
            from databricks import sql
            print("[Info] Executing via Databricks SQL Connector (Remote)")
            with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    if query.strip().upper().startswith("SELECT"):
                        # Get columns and data for JSON serialization
                        columns = [desc[0] for desc in cursor.description]
                        data = cursor.fetchall()
                        # Simple list of dicts
                        results = [dict(zip(columns, row)) for row in data]
                        return json.dumps(results, default=str)
                    else:
                        # For INSERT/UPDATE
                        return json.dumps({"status": "success", "rows_affected": cursor.rowcount})
        except ImportError:
            print("[Warning] databricks-sql-connector not installed.")
        except Exception as e:
            print(f"[Error] SQL Connector failed: {e}")
    
    return None

@tool
def search_properties(location: str, max_price: Optional[float] = None, property_type: Optional[str] = None) -> str:
    """
    Search for properties in the Wanderbricks inventory.
    Args:
        location: The city name to search in (e.g., 'New York', 'Miami').
        max_price: Optional maximum price per night.
        property_type: Optional type of property (e.g., 'apartment', 'villa').
    """
    # 1. Generate SQL
    sql_query = _generate_search_sql(location, max_price, property_type)
    print(f"[Tool: search_properties] Executing SQL: {sql_query}")
    
    # 2. Try Real Execution
    real_data = _run_sql_query(sql_query)
    if real_data:
        return real_data

    # 3. Fallback to Mock Data (Local Testing)
    print("[Tool: search_properties] No connection found. Returning MOCK data.")
    results = []
    
    for prop in MOCK_PROPERTIES:
        if location.lower() not in prop["location"].lower():
            continue
        if max_price is not None and prop["price_per_night"] > max_price:
            continue
        if property_type is not None and property_type.lower() != prop["type"].lower():
            continue
        results.append(prop)
        
    if not results:
        return json.dumps({"message": "No properties found matching criteria."})
    return json.dumps(results, indent=2)

@tool
def get_amenities(property_id: str) -> str:
    """
    Retrieve the list of amenities for a specific property.
    """
    # 1. Generate SQL
    sql_query = _generate_amenities_sql(property_id)
    print(f"[Tool: get_amenities] Executing SQL: {sql_query}")
    
    # 2. Try Real Execution
    real_data = _run_sql_query(sql_query)
    if real_data:
        return real_data

    # 3. Fallback to Mock Data
    print("[Tool: get_amenities] No connection found. Returning MOCK data.")
    if property_id in MOCK_AMENITIES:
        return json.dumps({"property_id": property_id, "amenities": MOCK_AMENITIES[property_id]})
    else:
        return json.dumps({"error": f"Property ID {property_id} not found."})

@tool
def book_property(property_id: str, dates: str, user_email: str) -> str:
    """
    Create a new booking entry in the bookings table.
    """
    booking_id = f"bk_{random.randint(1000, 9999)}"
    
    # 1. Generate SQL
    sql_query = _generate_booking_sql(booking_id, property_id, dates, user_email)
    print(f"[Tool: book_property] Executing SQL: {sql_query}")
    
    # 2. Try Real Execution
    real_data = _run_sql_query(sql_query)
    if real_data:
         return json.dumps({
            "status": "success",
            "message": f"Booking confirmed for {property_id}.",
            "booking_id": booking_id,
            "db_result": real_data
        })

    # 3. Fallback to Mock Data
    print("[Tool: book_property] No connection found. Returning MOCK data.")
    # Check if property exists in our mock data
    prop_exists = any(p["id"] == property_id for p in MOCK_PROPERTIES)
    if not prop_exists:
        return json.dumps({"status": "failed", "reason": "Invalid property ID"})
    
    booking = {
        "booking_id": booking_id,
        "property_id": property_id,
        "dates": dates,
        "user_email": user_email,
        "status": "confirmed"
    }
    MOCK_BOOKINGS.append(booking)
    return json.dumps({
        "status": "success",
        "message": f"Booking confirmed for {property_id}.",
        "booking_details": booking
    })

# =============================================================================
# NEW: Dynamic NL-to-SQL Tool with Human Review
# =============================================================================

@tool
def natural_language_query(query: str, skip_review: bool = False) -> str:
    """
    Execute any natural language query against the Wanderbricks database.
    
    This tool dynamically generates SQL from natural language queries using
    the full database schema. For complex or sensitive queries, it will
    ask for human review before execution.
    
    Args:
        query: Natural language description of what data you want.
               Examples:
               - "Show me top-rated hosts in France"
               - "Find properties with pools under $200/night"
               - "What are the most booked properties?"
               - "List all amenities available"
        skip_review: If True, skip human review (for high-confidence queries).
    
    Returns:
        JSON string with query results or error message.
    """
    if not SQL_GENERATOR_AVAILABLE:
        return json.dumps({
            "error": "Dynamic SQL generation not available. Use specific tools instead.",
            "available_tools": ["search_properties", "get_amenities", "book_property"]
        })
    
    print(f"[Tool: natural_language_query] Processing: {query}")
    
    # 1. Generate SQL using the schema-aware generator
    generator = SQLGenerator()
    
    try:
        result = generator.generate(query)
    except Exception as e:
        return json.dumps({
            "error": f"SQL generation failed: {str(e)}",
            "suggestion": "Try using specific tools like search_properties or get_amenities"
        })
    
    # 2. Check if SQL is valid
    if not result.is_valid:
        return json.dumps({
            "error": "Generated SQL failed validation",
            "validation_errors": result.validation_errors,
            "attempted_sql": result.sql
        })
    
    print(f"[Tool: natural_language_query] Generated SQL: {result.sql}")
    print(f"[Tool: natural_language_query] Confidence: {result.confidence:.0%}")
    print(f"[Tool: natural_language_query] Tables: {result.tables_used}")
    
    # 3. Human Review Workflow (if enabled)
    final_sql = result.sql
    
    if _human_review_enabled and not skip_review:
        review_node = get_review_node()
        
        # Skip review for high-confidence queries
        if not review_node.should_skip_review(result.confidence):
            print("[Tool: natural_language_query] Requesting human review...")
            
            review_request = review_node.create_review_request(
                original_query=query,
                generated_sql=result.sql,
                explanation=result.explanation,
                tables_used=result.tables_used,
                confidence=result.confidence
            )
            
            review_result = review_node.review(review_request)
            
            if not review_result.should_execute:
                if review_result.action == ReviewAction.REGENERATE and review_result.feedback:
                    # Try to regenerate with feedback
                    print(f"[Tool: natural_language_query] Regenerating with feedback: {review_result.feedback}")
                    refined_result = generator.refine(result, review_result.feedback)
                    
                    if refined_result.is_valid:
                        # One more review for the refined query
                        review_request2 = review_node.create_review_request(
                            original_query=query,
                            generated_sql=refined_result.sql,
                            explanation=refined_result.explanation,
                            tables_used=refined_result.tables_used,
                            confidence=refined_result.confidence,
                            iteration=1
                        )
                        review_result2 = review_node.review(review_request2)
                        
                        if review_result2.should_execute:
                            final_sql = review_result2.final_sql
                        else:
                            return json.dumps({
                                "status": "cancelled",
                                "message": "Query was cancelled after refinement",
                                "generated_sql": refined_result.sql
                            })
                    else:
                        return json.dumps({
                            "error": "Refined SQL failed validation",
                            "validation_errors": refined_result.validation_errors
                        })
                else:
                    return json.dumps({
                        "status": "cancelled",
                        "message": "Query was cancelled by user",
                        "generated_sql": result.sql
                    })
            else:
                final_sql = review_result.final_sql
    
    # 4. Execute the SQL
    print(f"[Tool: natural_language_query] Executing: {final_sql}")
    real_data = _run_sql_query(final_sql)
    
    if real_data:
        return json.dumps({
            "status": "success",
            "query": query,
            "sql_executed": final_sql,
            "explanation": result.explanation,
            "data": json.loads(real_data)
        }, indent=2)
    
    # 5. Fallback - return the SQL for manual execution
    return json.dumps({
        "status": "no_connection",
        "message": "No database connection available. Here is the generated SQL for manual execution.",
        "query": query,
        "generated_sql": final_sql,
        "explanation": result.explanation,
        "tables_used": result.tables_used,
        "confidence": result.confidence
    }, indent=2)


@tool 
def get_table_schema(table_name: Optional[str] = None) -> str:
    """
    Get the schema information for Wanderbricks tables.
    
    Args:
        table_name: Optional specific table name. If not provided, returns all tables.
                   Valid tables: properties, cities, hosts, users, bookings, 
                   amenities, property_amenities, property_images, employees
    
    Returns:
        JSON string with schema information.
    """
    if not SQL_GENERATOR_AVAILABLE:
        return json.dumps({
            "error": "Schema information not available in legacy mode.",
            "tables": ["properties", "cities", "hosts", "users", "bookings", 
                      "amenities", "property_amenities", "property_images", "employees"]
        })
    
    from .sql_generator import WANDERBRICKS_SCHEMA
    
    if table_name:
        table_name_lower = table_name.lower().replace("wanderbricks.", "")
        if table_name_lower in WANDERBRICKS_SCHEMA:
            return json.dumps({
                "table": table_name_lower,
                "schema": WANDERBRICKS_SCHEMA[table_name_lower]
            }, indent=2)
        else:
            return json.dumps({
                "error": f"Table '{table_name}' not found",
                "available_tables": list(WANDERBRICKS_SCHEMA.keys())
            })
    else:
        # Return overview of all tables
        overview = {}
        for name, info in WANDERBRICKS_SCHEMA.items():
            overview[name] = {
                "full_name": info["table_name"],
                "description": info["description"],
                "columns": list(info["columns"].keys())
            }
        return json.dumps(overview, indent=2)


@tool
def smart_query(query: str, show_steps: bool = True) -> str:
    """
    Execute a natural language query using multi-step SQL generation.
    
    This tool uses an intelligent workflow that:
    1. Analyzes your query to identify what context is needed
    2. Runs intermediate SQL queries to gather valid values (e.g., property types, city names)
    3. Uses that context to generate an accurate final query
    4. Optionally shows you each step for transparency
    
    Use this for complex queries that need database exploration first.
    
    Args:
        query: Natural language description of what data you want.
               Examples:
               - "looking for apartment in new york"
               - "properties with pools in Miami under $300"
               - "top-rated hosts with verified status"
        show_steps: If True, displays each step of the query plan (default: True)
    
    Returns:
        JSON string with the query plan, steps, and final results.
    """
    if not MULTI_STEP_AVAILABLE:
        return json.dumps({
            "error": "Multi-step SQL generation not available.",
            "suggestion": "Use natural_language_query tool instead."
        })
    
    print(f"[Tool: smart_query] Processing with multi-step workflow: {query}")
    
    # Create the smart generator with SQL executor
    smart_gen = SmartNLToSQL(
        sql_executor=_run_sql_query,
        human_review_enabled=_human_review_enabled
    )
    
    try:
        result = smart_gen.process(query, force_multi_step=True)
        
        # If we have a final SQL, try to execute it
        if result.get("final_sql") and result.get("status") == "completed":
            final_result = _run_sql_query(result["final_sql"])
            if final_result:
                result["data"] = json.loads(final_result)
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "error": f"Multi-step query failed: {str(e)}",
            "query": query
        })


@tool
def explore_column_values(table_name: str, column_name: str, filter_pattern: Optional[str] = None) -> str:
    """
    Explore distinct values in a specific column.
    
    Use this to discover valid values before constructing a query.
    
    Args:
        table_name: Name of the table (e.g., 'properties', 'cities')
        column_name: Name of the column to explore
        filter_pattern: Optional pattern to filter values (uses ILIKE)
    
    Returns:
        JSON with distinct values found.
    """
    # Validate table name
    valid_tables = ['properties', 'cities', 'hosts', 'users', 'bookings', 
                   'amenities', 'property_amenities', 'property_images', 'employees']
    
    if table_name.lower() not in valid_tables:
        return json.dumps({
            "error": f"Invalid table: {table_name}",
            "valid_tables": valid_tables
        })
    
    full_table = f"wanderbricks.{table_name.lower()}"
    
    if filter_pattern:
        sql = f"SELECT DISTINCT {column_name} FROM {full_table} WHERE {column_name} ILIKE '%{filter_pattern}%' LIMIT 50"
    else:
        sql = f"SELECT DISTINCT {column_name} FROM {full_table} LIMIT 50"
    
    print(f"[Tool: explore_column_values] SQL: {sql}")
    
    result = _run_sql_query(sql)
    
    if result:
        return json.dumps({
            "table": table_name,
            "column": column_name,
            "filter": filter_pattern,
            "values": json.loads(result)
        }, indent=2)
    
    # Fallback for no connection
    return json.dumps({
        "status": "no_connection",
        "sql": sql,
        "message": "Execute this SQL to explore values"
    })


def get_wanderbricks_tools(include_dynamic: bool = True, include_multi_step: bool = True):
    """
    Get all available Wanderbricks tools.
    
    Args:
        include_dynamic: If True, includes the natural_language_query tool
                        for dynamic SQL generation.
        include_multi_step: If True, includes multi-step query tools.
    
    Returns:
        List of tool functions.
    """
    # Core tools (always available)
    tools = [search_properties, get_amenities, book_property]
    
    # Add dynamic tools if SQL generator is available
    if include_dynamic and SQL_GENERATOR_AVAILABLE:
        tools.extend([natural_language_query, get_table_schema])
    
    # Add multi-step tools if available
    if include_multi_step and MULTI_STEP_AVAILABLE:
        tools.extend([smart_query, explore_column_values])
    
    return tools


def get_fast_path_tools():
    """Get only the fast-path tools (no dynamic SQL generation)."""
    return [search_properties, get_amenities, book_property]


def get_all_tools():
    """Get all tools including dynamic, multi-step, and schema tools."""
    return get_wanderbricks_tools(include_dynamic=True, include_multi_step=True)
