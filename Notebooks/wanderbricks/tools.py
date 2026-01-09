from langchain.tools import tool
from typing import List, Optional
import json
import random

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

# --- SQL Generation Helpers (Exposed for Evaluation) ---

def _generate_search_sql(location: str, max_price: Optional[float] = None, property_type: Optional[str] = None) -> str:
    query = f"SELECT * FROM wanderbricks.properties p JOIN wanderbricks.cities c ON p.city_id = c.city_id WHERE c.name ILIKE '%{location}%'"
    if max_price:
        query += f" AND p.price_per_night <= {max_price}"
    if property_type:
        query += f" AND p.property_type ILIKE '{property_type}'"
    return query.strip()

def _generate_amenities_sql(property_id: str) -> str:
    query = f"""
    SELECT a.name 
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

def get_wanderbricks_tools():
    return [search_properties, get_amenities, book_property]
