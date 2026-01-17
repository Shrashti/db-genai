"""
SQL Generator Module for Wanderbricks NL-to-SQL Layer

This module provides dynamic, schema-aware SQL generation from natural language queries.
It uses the table schemas and ERD documentation to generate accurate queries.
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import os

# Schema definitions as structured data (derived from DATA_SCHEMAS.md)
WANDERBRICKS_SCHEMA = {
    "properties": {
        "table_name": "wanderbricks.properties",
        "description": "Main property listings table containing all vacation rental properties",
        "columns": {
            "property_id": {"type": "bigint", "description": "Unique identifier for the property", "is_pk": True},
            "host_id": {"type": "bigint", "description": "Reference to the property owner/host", "fk": "hosts.host_id"},
            "city_id": {"type": "bigint", "description": "Reference to the city location", "fk": "cities.city_id"},
            "title": {"type": "string", "description": "Title of the property listing"},
            "description": {"type": "string", "description": "Detailed description of the property"},
            "base_price": {"type": "float", "description": "Base price per night for booking"},
            "property_type": {"type": "string", "description": "Type of property (house, apartment, villa)"},
            "max_guests": {"type": "int", "description": "Maximum number of guests allowed"},
            "bedrooms": {"type": "int", "description": "Number of bedrooms"},
            "bathrooms": {"type": "int", "description": "Number of bathrooms"},
            "created_at": {"type": "date", "description": "Date the property was listed"},
        }
    },
    "cities": {
        "table_name": "wanderbricks.cities",
        "description": "Reference table for city/location information",
        "columns": {
            "city_id": {"type": "bigint", "description": "Unique identifier for the city", "is_pk": True},
            "city": {"type": "string", "description": "Name of the city"},
            "country": {"type": "string", "description": "Name of the country"},
            "description": {"type": "string", "description": "Detailed description of the city"},
        }
    },
    "hosts": {
        "table_name": "wanderbricks.hosts",
        "description": "Information about property owners/hosts",
        "columns": {
            "host_id": {"type": "bigint", "description": "Unique identifier for the host", "is_pk": True},
            "name": {"type": "string", "description": "Host's full name"},
            "email": {"type": "string", "description": "Host's email address"},
            "phone": {"type": "string", "description": "Host's phone number"},
            "is_verified": {"type": "boolean", "description": "Whether the host is verified"},
            "is_active": {"type": "boolean", "description": "Whether the host account is active"},
            "rating": {"type": "float", "description": "Host rating (3.0-5.0 scale)"},
            "country": {"type": "string", "description": "Host's country of residence"},
            "joined_at": {"type": "date", "description": "Date the host joined"},
        }
    },
    "users": {
        "table_name": "wanderbricks.users",
        "description": "Customer/guest information",
        "columns": {
            "user_id": {"type": "bigint", "description": "Unique identifier for the user", "is_pk": True},
            "email": {"type": "string", "description": "User's email address"},
            "name": {"type": "string", "description": "User's full name"},
            "country": {"type": "string", "description": "Country of residence"},
            "user_type": {"type": "string", "description": "Type of user (individual, business)"},
            "created_at": {"type": "timestamp", "description": "Account creation timestamp"},
            "is_business": {"type": "boolean", "description": "Whether the user is a business account"},
            "company_name": {"type": "string", "description": "Company name if business user"},
        }
    },
    "bookings": {
        "table_name": "wanderbricks.bookings",
        "description": "Reservation/booking records",
        "columns": {
            "booking_id": {"type": "bigint", "description": "Unique identifier for the booking", "is_pk": True},
            "user_id": {"type": "bigint", "description": "Reference to the guest", "fk": "users.user_id"},
            "property_id": {"type": "bigint", "description": "Reference to the booked property", "fk": "properties.property_id"},
            "check_in": {"type": "date", "description": "Check-in date"},
            "check_out": {"type": "date", "description": "Check-out date"},
            "guests_count": {"type": "int", "description": "Number of guests for the booking"},
            "total_amount": {"type": "float", "description": "Total amount paid"},
            "status": {"type": "string", "description": "Booking status (pending, confirmed, cancelled)"},
            "created_at": {"type": "date", "description": "Date the booking was created"},
            "updated_at": {"type": "date", "description": "Date the booking was last updated"},
        }
    },
    "amenities": {
        "table_name": "wanderbricks.amenities",
        "description": "Master list of available amenities",
        "columns": {
            "amenity_id": {"type": "bigint", "description": "Unique identifier for the amenity", "is_pk": True},
            "name": {"type": "string", "description": "Name of the amenity (Wi-Fi, Pool)"},
            "category": {"type": "string", "description": "Category of the amenity (Basic, Luxury)"},
            "icon": {"type": "string", "description": "Icon representation"},
        }
    },
    "property_amenities": {
        "table_name": "wanderbricks.property_amenities",
        "description": "Junction table linking properties to their amenities",
        "columns": {
            "property_id": {"type": "bigint", "description": "Reference to the property", "fk": "properties.property_id"},
            "amenity_id": {"type": "bigint", "description": "Reference to the amenity", "fk": "amenities.amenity_id"},
        }
    },
    "property_images": {
        "table_name": "wanderbricks.property_images",
        "description": "Images associated with properties",
        "columns": {
            "image_id": {"type": "bigint", "description": "Unique identifier for the image", "is_pk": True},
            "property_id": {"type": "bigint", "description": "Reference to the property", "fk": "properties.property_id"},
            "url": {"type": "string", "description": "URL of the image"},
            "sequence": {"type": "int", "description": "Order to display images"},
            "is_primary": {"type": "boolean", "description": "Whether this is the primary image"},
            "uploaded_at": {"type": "date", "description": "Date the image was uploaded"},
        }
    },
    "employees": {
        "table_name": "wanderbricks.employees",
        "description": "Staff associated with hosts",
        "columns": {
            "employee_id": {"type": "bigint", "description": "Unique identifier for the employee", "is_pk": True},
            "host_id": {"type": "bigint", "description": "Reference to the employer host", "fk": "hosts.host_id"},
            "name": {"type": "string", "description": "Employee's full name"},
            "role": {"type": "string", "description": "Employee's role (cleaner, chef)"},
            "email": {"type": "string", "description": "Employee's email address"},
            "phone": {"type": "string", "description": "Employee's phone number"},
            "country": {"type": "string", "description": "Country of employment"},
            "joined_at": {"type": "date", "description": "Date the employee joined"},
            "end_service_date": {"type": "date", "description": "Date the employee left"},
            "is_currently_employed": {"type": "boolean", "description": "Whether currently employed"},
        }
    },
}

# Relationship definitions for JOIN generation
RELATIONSHIPS = [
    {"from": "properties", "to": "hosts", "on": "host_id", "type": "many-to-one"},
    {"from": "properties", "to": "cities", "on": "city_id", "type": "many-to-one"},
    {"from": "bookings", "to": "properties", "on": "property_id", "type": "many-to-one"},
    {"from": "bookings", "to": "users", "on": "user_id", "type": "many-to-one"},
    {"from": "property_amenities", "to": "properties", "on": "property_id", "type": "many-to-one"},
    {"from": "property_amenities", "to": "amenities", "on": "amenity_id", "type": "many-to-one"},
    {"from": "property_images", "to": "properties", "on": "property_id", "type": "many-to-one"},
    {"from": "employees", "to": "hosts", "on": "host_id", "type": "many-to-one"},
]


class QueryType(Enum):
    """Types of SQL queries that can be generated."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    AGGREGATE = "aggregate"


@dataclass
class SQLResult:
    """Result of SQL generation."""
    sql: str
    explanation: str
    tables_used: List[str]
    confidence: float
    query_type: QueryType
    is_valid: bool
    validation_errors: List[str]
    suggested_modifications: Optional[str] = None


@dataclass 
class ReviewState:
    """State for human review workflow."""
    original_query: str
    generated_sql: str
    explanation: str
    status: str  # pending, approved, rejected, modified
    modified_sql: Optional[str] = None
    review_notes: Optional[str] = None
    iteration: int = 0


class SchemaContext:
    """Loads and provides table schema information to the LLM."""
    
    def __init__(self, schema: Dict = None):
        self.schema = schema or WANDERBRICKS_SCHEMA
        self.relationships = RELATIONSHIPS
        
    def get_schema_prompt(self) -> str:
        """Generate a schema description for LLM context."""
        lines = ["# Available Tables and Columns\n"]
        
        for table_name, table_info in self.schema.items():
            lines.append(f"## {table_info['table_name']}")
            lines.append(f"Description: {table_info['description']}")
            lines.append("Columns:")
            
            for col_name, col_info in table_info["columns"].items():
                pk_marker = " (PK)" if col_info.get("is_pk") else ""
                fk_marker = f" (FK -> {col_info['fk']})" if col_info.get("fk") else ""
                lines.append(f"  - {col_name}: {col_info['type']}{pk_marker}{fk_marker} - {col_info['description']}")
            lines.append("")
        
        lines.append("# Table Relationships")
        for rel in self.relationships:
            lines.append(f"  - {rel['from']}.{rel['on']} -> {rel['to']}.{rel['on']}")
        
        return "\n".join(lines)
    
    def get_tables_for_query(self, keywords: List[str]) -> List[str]:
        """Suggest relevant tables based on query keywords."""
        relevant_tables = set()
        
        keyword_to_tables = {
            "property": ["properties", "property_amenities", "property_images"],
            "properties": ["properties", "property_amenities", "property_images"],
            "host": ["hosts", "properties"],
            "hosts": ["hosts", "properties"],
            "city": ["cities", "properties"],
            "location": ["cities", "properties"],
            "book": ["bookings", "users", "properties"],
            "booking": ["bookings", "users", "properties"],
            "reservation": ["bookings", "users", "properties"],
            "user": ["users", "bookings"],
            "guest": ["users", "bookings"],
            "amenity": ["amenities", "property_amenities"],
            "amenities": ["amenities", "property_amenities"],
            "pool": ["amenities", "property_amenities"],
            "wifi": ["amenities", "property_amenities"],
            "image": ["property_images"],
            "photo": ["property_images"],
            "employee": ["employees", "hosts"],
            "staff": ["employees", "hosts"],
            "price": ["properties", "bookings"],
            "rating": ["hosts"],
            "revenue": ["bookings", "properties", "hosts"],
        }
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for kw, tables in keyword_to_tables.items():
                if kw in keyword_lower:
                    relevant_tables.update(tables)
        
        return list(relevant_tables)


class SQLValidator:
    """Validates generated SQL for safety and correctness."""
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        (r'\bDROP\b', "DROP statements are not allowed"),
        (r'\bTRUNCATE\b', "TRUNCATE statements are not allowed"),
        (r'\bALTER\b', "ALTER statements are not allowed"),
        (r'\bCREATE\b', "CREATE statements are not allowed"),
        (r'\bGRANT\b', "GRANT statements are not allowed"),
        (r'\bREVOKE\b', "REVOKE statements are not allowed"),
        (r'DELETE\s+FROM\s+\w+\s*(?:;|$)', "DELETE without WHERE clause is not allowed"),
        (r'UPDATE\s+\w+\s+SET\s+[^W]*(?:;|$)', "UPDATE without WHERE clause is not allowed"),
    ]
    
    # Allowed schemas
    ALLOWED_SCHEMAS = ["wanderbricks"]
    
    def validate(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Validate SQL query for safety.
        Returns (is_valid, list_of_errors).
        """
        errors = []
        sql_upper = sql.upper()
        
        # Check for dangerous patterns
        for pattern, message in self.DANGEROUS_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                errors.append(message)
        
        # Check schema access
        table_refs = re.findall(r'(?:FROM|JOIN|INTO|UPDATE)\s+(\w+\.\w+)', sql, re.IGNORECASE)
        for table_ref in table_refs:
            schema = table_ref.split('.')[0].lower()
            if schema not in self.ALLOWED_SCHEMAS:
                errors.append(f"Access to schema '{schema}' is not allowed")
        
        # Ensure SELECT queries have LIMIT (add warning)
        if sql_upper.strip().startswith('SELECT') and 'LIMIT' not in sql_upper:
            # This is a warning, not an error - we'll add LIMIT
            pass
        
        return len(errors) == 0, errors
    
    def sanitize(self, sql: str) -> str:
        """Sanitize SQL by adding safety measures."""
        sql_upper = sql.upper()
        
        # Add LIMIT if missing from SELECT queries
        if sql_upper.strip().startswith('SELECT') and 'LIMIT' not in sql_upper:
            # Find the end of the query
            sql = sql.rstrip().rstrip(';')
            sql = f"{sql} LIMIT 100"
        
        return sql


class SQLGenerator:
    """
    Dynamic SQL generator using LLM with schema context.
    
    This class generates SQL queries from natural language by:
    1. Loading schema context
    2. Crafting a prompt with schema + user query
    3. Using LLM to generate SQL
    4. Validating the generated SQL
    """
    
    def __init__(self, llm=None):
        """
        Initialize the SQL generator.
        
        Args:
            llm: LangChain compatible LLM. If None, uses databricks model.
        """
        self.schema_context = SchemaContext()
        self.validator = SQLValidator()
        self.llm = llm
        
    def _get_llm(self):
        """Lazy load LLM if not provided."""
        if self.llm is None:
            try:
                from langchain_community.chat_models import ChatDatabricks
                self.llm = ChatDatabricks(
                    endpoint="databricks-meta-llama-3-70b-instruct",
                    temperature=0.0
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize LLM: {e}")
        return self.llm
    
    def generate(self, user_query: str, max_retries: int = 2) -> SQLResult:
        """
        Generate SQL from natural language query.
        
        Args:
            user_query: Natural language query from user
            max_retries: Number of retries if generation fails
            
        Returns:
            SQLResult with generated SQL and metadata
        """
        schema_prompt = self.schema_context.get_schema_prompt()
        
        prompt = f"""You are a SQL expert for the Wanderbricks vacation rental platform.

{schema_prompt}

Generate a SQL query for the following user request:
"{user_query}"

IMPORTANT RULES:
1. Only use tables and columns that exist in the schema above
2. Use proper JOINs based on the foreign key relationships
3. Use ILIKE for case-insensitive string matching
4. Always prefix table names with 'wanderbricks.' schema
5. For SELECT queries, limit results to 100 rows unless specified otherwise

Return your response in this exact JSON format:
{{
    "sql": "YOUR SQL QUERY HERE",
    "explanation": "Brief explanation of what this query does",
    "tables_used": ["list", "of", "tables"],
    "confidence": 0.0 to 1.0,
    "query_type": "select" or "insert" or "update" or "aggregate"
}}

Return ONLY the JSON, no other text."""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            
            # Parse response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return self._error_result(user_query, "Failed to parse LLM response as JSON")
            
            result_dict = json.loads(json_match.group())
            
            # Validate the SQL
            is_valid, errors = self.validator.validate(result_dict["sql"])
            
            # Sanitize if valid
            if is_valid:
                result_dict["sql"] = self.validator.sanitize(result_dict["sql"])
            
            return SQLResult(
                sql=result_dict["sql"],
                explanation=result_dict.get("explanation", ""),
                tables_used=result_dict.get("tables_used", []),
                confidence=result_dict.get("confidence", 0.5),
                query_type=QueryType(result_dict.get("query_type", "select")),
                is_valid=is_valid,
                validation_errors=errors
            )
            
        except json.JSONDecodeError as e:
            return self._error_result(user_query, f"JSON parsing error: {e}")
        except Exception as e:
            return self._error_result(user_query, f"Generation error: {e}")
    
    def _error_result(self, query: str, error: str) -> SQLResult:
        """Create an error SQLResult."""
        return SQLResult(
            sql="",
            explanation=f"Error: {error}",
            tables_used=[],
            confidence=0.0,
            query_type=QueryType.SELECT,
            is_valid=False,
            validation_errors=[error]
        )
    
    def refine(self, original_result: SQLResult, feedback: str) -> SQLResult:
        """
        Refine a generated SQL query based on user feedback.
        
        Args:
            original_result: Previous SQLResult to refine
            feedback: User's feedback or modification request
            
        Returns:
            New SQLResult with refined SQL
        """
        prompt = f"""You are a SQL expert for the Wanderbricks vacation rental platform.

The previous query was:
```sql
{original_result.sql}
```

Explanation: {original_result.explanation}

The user wants to modify it:
"{feedback}"

{self.schema_context.get_schema_prompt()}

Generate the refined SQL query. Return in JSON format:
{{
    "sql": "YOUR REFINED SQL QUERY HERE",
    "explanation": "What changed and why",
    "tables_used": ["list", "of", "tables"],
    "confidence": 0.0 to 1.0,
    "query_type": "select" or "insert" or "update" or "aggregate"
}}

Return ONLY the JSON, no other text."""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return self._error_result(feedback, "Failed to parse refinement response")
            
            result_dict = json.loads(json_match.group())
            is_valid, errors = self.validator.validate(result_dict["sql"])
            
            if is_valid:
                result_dict["sql"] = self.validator.sanitize(result_dict["sql"])
            
            return SQLResult(
                sql=result_dict["sql"],
                explanation=result_dict.get("explanation", ""),
                tables_used=result_dict.get("tables_used", []),
                confidence=result_dict.get("confidence", 0.5),
                query_type=QueryType(result_dict.get("query_type", "select")),
                is_valid=is_valid,
                validation_errors=errors,
                suggested_modifications=f"Refined based on: {feedback}"
            )
            
        except Exception as e:
            return self._error_result(feedback, f"Refinement error: {e}")


class IterativeQueryBuilder:
    """
    Supports multi-step query building with clarifying questions.
    
    For complex queries, this builder can:
    1. Identify ambiguities in the user's request
    2. Ask clarifying questions
    3. Build the query incrementally
    """
    
    def __init__(self, llm=None):
        self.generator = SQLGenerator(llm)
        self.schema_context = SchemaContext()
        self.conversation_history: List[Dict[str, str]] = []
        
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze a query and determine if clarification is needed.
        
        Returns:
            Dict with:
            - needs_clarification: bool
            - clarifying_questions: list of questions if needed
            - initial_sql: SQLResult if no clarification needed
        """
        # Keywords that often need clarification
        ambiguous_patterns = [
            (r'\b(best|top|popular)\b', "How should 'best' or 'top' be determined? (e.g., by rating, bookings, price)"),
            (r'\b(recent|latest)\b', "What time period should be considered for 'recent'?"),
            (r'\b(expensive|cheap|affordable)\b', "What price range do you consider for this?"),
            (r'\b(near|close to|around)\b', "What location should we search near?"),
        ]
        
        questions = []
        for pattern, question in ambiguous_patterns:
            if re.search(pattern, user_query, re.IGNORECASE):
                questions.append(question)
        
        if questions:
            return {
                "needs_clarification": True,
                "clarifying_questions": questions[:2],  # Limit to 2 questions
                "initial_sql": None
            }
        
        # No clarification needed, generate directly
        result = self.generator.generate(user_query)
        return {
            "needs_clarification": False,
            "clarifying_questions": [],
            "initial_sql": result
        }
    
    def build_with_context(self, user_query: str, context: Dict[str, str]) -> SQLResult:
        """
        Build a query with additional context from clarifying answers.
        
        Args:
            user_query: Original user query
            context: Dict of clarifying answers
            
        Returns:
            SQLResult with generated SQL
        """
        # Enhance the query with context
        context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
        enhanced_query = f"{user_query}\n\nAdditional context:\n{context_str}"
        
        return self.generator.generate(enhanced_query)


# Convenience functions for tool integration
def generate_sql_from_nl(query: str, llm=None) -> SQLResult:
    """
    Convenience function to generate SQL from natural language.
    
    Args:
        query: Natural language query
        llm: Optional LLM instance
        
    Returns:
        SQLResult with generated SQL
    """
    generator = SQLGenerator(llm)
    return generator.generate(query)


def get_schema_context() -> str:
    """Get the schema context as a string for LLM prompts."""
    return SchemaContext().get_schema_prompt()
