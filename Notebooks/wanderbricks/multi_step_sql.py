"""
Multi-Step NL-to-SQL Workflow

This module implements an iterative query planning approach where the system
runs intermediate SQL queries to gather context (valid values, schema info)
before generating the final query.

Example workflow:
1. User: "looking for apartment in newyork for 2 nights"
2. System runs: SELECT DISTINCT property_type FROM properties (to validate "apartment")
3. System runs: SELECT city_id, city FROM cities WHERE city ILIKE '%new%york%' (to get city_id)
4. System uses results to construct accurate final query
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from .sql_generator import (
    SQLGenerator, 
    SQLResult, 
    SchemaContext, 
    SQLValidator,
    WANDERBRICKS_SCHEMA,
    QueryType
)


class StepType(Enum):
    """Types of intermediate steps."""
    EXPLORE_VALUES = "explore_values"      # Get distinct values for a column
    VALIDATE_ENTITY = "validate_entity"    # Validate entity exists
    GET_FOREIGN_KEY = "get_foreign_key"    # Resolve foreign key
    CHECK_AVAILABILITY = "check_availability"  # Check date availability
    AGGREGATE_CHECK = "aggregate_check"    # Run aggregation for context
    FINAL_QUERY = "final_query"            # The final query


@dataclass
class QueryStep:
    """A single step in the multi-step query workflow."""
    step_id: int
    step_type: StepType
    purpose: str
    sql: str
    result: Optional[Any] = None
    executed: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "purpose": self.purpose,
            "sql": self.sql,
            "result": self.result,
            "executed": self.executed,
            "error": self.error
        }


@dataclass
class QueryPlan:
    """A complete plan with multiple query steps."""
    original_query: str
    steps: List[QueryStep] = field(default_factory=list)
    final_sql: Optional[str] = None
    context_gathered: Dict[str, Any] = field(default_factory=dict)
    status: str = "planning"  # planning, executing, completed, failed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "steps": [s.to_dict() for s in self.steps],
            "final_sql": self.final_sql,
            "context_gathered": self.context_gathered,
            "status": self.status
        }
    
    def get_display_summary(self) -> str:
        """Get a human-readable summary of the plan."""
        lines = [
            "=" * 60,
            "📋 QUERY EXECUTION PLAN",
            "=" * 60,
            f"Original Query: \"{self.original_query}\"",
            "",
            "Steps:",
        ]
        
        for step in self.steps:
            status_icon = "✅" if step.executed else "⏳"
            lines.append(f"  {status_icon} Step {step.step_id}: {step.purpose}")
            lines.append(f"      SQL: {step.sql[:80]}...")
            if step.result:
                result_preview = str(step.result)[:100]
                lines.append(f"      Result: {result_preview}...")
        
        if self.final_sql:
            lines.extend([
                "",
                "Final Query:",
                f"  {self.final_sql}",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)


class MultiStepSQLGenerator:
    """
    Generates SQL through a multi-step process:
    1. Analyze the user query to identify needed context
    2. Generate intermediate queries to gather context
    3. Execute intermediate queries
    4. Use context to generate accurate final query
    """
    
    # Patterns that suggest needing to explore values
    EXPLORATION_PATTERNS = {
        # Property type exploration
        r'\b(apartment|house|villa|cabin|condo|loft|studio)\b': {
            'table': 'properties',
            'column': 'property_type',
            'purpose': 'Validate property type value'
        },
        # City exploration
        r'\b(in|at|near)\s+(\w+(?:\s+\w+)?)\b': {
            'table': 'cities',
            'column': 'city',
            'purpose': 'Find matching city'
        },
        # Amenity exploration
        r'\b(with|has|have)\s+(pool|wifi|kitchen|gym|parking|ac|air.?conditioning)\b': {
            'table': 'amenities',
            'column': 'name',
            'purpose': 'Find matching amenity'
        },
        # Host rating
        r'\b(top|best|highest).?(rated|rating)\b': {
            'table': 'hosts',
            'column': 'rating',
            'purpose': 'Get rating distribution'
        },
        # Price range
        r'\b(under|below|less than|max|maximum)\s*\$?\s*(\d+)\b': {
            'table': 'properties',
            'column': 'base_price',
            'purpose': 'Validate price range'
        },
    }
    
    def __init__(self, 
                 llm=None, 
                 sql_executor: Optional[Callable[[str], Any]] = None):
        """
        Initialize the multi-step generator.
        
        Args:
            llm: LLM for query generation
            sql_executor: Function to execute SQL and return results
        """
        self.base_generator = SQLGenerator(llm)
        self.schema_context = SchemaContext()
        self.validator = SQLValidator()
        self.sql_executor = sql_executor
        
    def analyze_query(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Analyze the user query to identify what context is needed.
        
        Returns list of exploration needs.
        """
        explorations = []
        
        for pattern, config in self.EXPLORATION_PATTERNS.items():
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                explorations.append({
                    'pattern': pattern,
                    'matched_text': match.group(0),
                    'table': config['table'],
                    'column': config['column'],
                    'purpose': config['purpose'],
                    'groups': match.groups() if match.groups() else None
                })
        
        return explorations
    
    def create_exploration_query(self, exploration: Dict[str, Any]) -> QueryStep:
        """Create an exploration query step."""
        table = exploration['table']
        column = exploration['column']
        full_table = f"wanderbricks.{table}"
        
        if exploration.get('groups') and len(exploration['groups']) > 1:
            # We have a captured value to search for
            search_value = exploration['groups'][-1]  # Last captured group
            sql = f"SELECT DISTINCT {column} FROM {full_table} WHERE {column} ILIKE '%{search_value}%' LIMIT 10"
        else:
            # Just get distinct values
            sql = f"SELECT DISTINCT {column} FROM {full_table} LIMIT 20"
        
        return QueryStep(
            step_id=0,  # Will be set by plan builder
            step_type=StepType.EXPLORE_VALUES,
            purpose=exploration['purpose'],
            sql=sql
        )
    
    def create_query_plan(self, user_query: str) -> QueryPlan:
        """
        Create a complete query plan with all steps.
        """
        plan = QueryPlan(original_query=user_query)
        
        # Analyze what context we need
        explorations = self.analyze_query(user_query)
        
        # Create exploration steps
        for i, exp in enumerate(explorations):
            step = self.create_exploration_query(exp)
            step.step_id = i + 1
            plan.steps.append(step)
        
        # Add a placeholder for the final query step
        final_step = QueryStep(
            step_id=len(plan.steps) + 1,
            step_type=StepType.FINAL_QUERY,
            purpose="Execute the final query with gathered context",
            sql=""  # Will be generated after exploration
        )
        plan.steps.append(final_step)
        
        return plan
    
    def execute_step(self, step: QueryStep) -> QueryStep:
        """Execute a single step and update it with results."""
        if self.sql_executor is None:
            step.error = "No SQL executor available"
            return step
        
        try:
            result = self.sql_executor(step.sql)
            step.result = result
            step.executed = True
        except Exception as e:
            step.error = str(e)
            step.executed = True
        
        return step
    
    def execute_plan(self, plan: QueryPlan) -> QueryPlan:
        """
        Execute all steps in the plan.
        """
        plan.status = "executing"
        
        # Execute exploration steps (all except final)
        for step in plan.steps[:-1]:
            self.execute_step(step)
            
            # Store results in context
            if step.result and not step.error:
                key = f"{step.step_type.value}_{step.step_id}"
                plan.context_gathered[key] = step.result
        
        # Generate final query using gathered context
        final_step = plan.steps[-1]
        
        try:
            final_sql = self.generate_final_query(
                plan.original_query, 
                plan.context_gathered
            )
            final_step.sql = final_sql.sql
            plan.final_sql = final_sql.sql
            
            # Execute final query
            if self.sql_executor:
                self.execute_step(final_step)
            
            plan.status = "completed"
            
        except Exception as e:
            final_step.error = str(e)
            plan.status = "failed"
        
        return plan
    
    def generate_final_query(self, 
                            user_query: str, 
                            context: Dict[str, Any]) -> SQLResult:
        """
        Generate the final query using gathered context.
        """
        # Build context prompt
        context_lines = ["Based on the following database exploration:"]
        
        for key, value in context.items():
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # Extract distinct values
                        values = set()
                        for row in parsed:
                            values.update(str(v) for v in row.values())
                        context_lines.append(f"- {key}: Found values: {list(values)[:10]}")
                except:
                    context_lines.append(f"- {key}: {value[:200]}")
            else:
                context_lines.append(f"- {key}: {value}")
        
        context_str = "\n".join(context_lines)
        
        # Use enhanced prompt
        enhanced_query = f"""
{user_query}

IMPORTANT CONTEXT FROM DATABASE EXPLORATION:
{context_str}

Use the EXACT values from the exploration results when constructing WHERE clauses.
For example, if city exploration found "New York" use that exact string.
"""
        
        return self.base_generator.generate(enhanced_query)
    
    def run(self, user_query: str, show_steps: bool = True) -> Dict[str, Any]:
        """
        Run the complete multi-step workflow.
        
        Args:
            user_query: Natural language query
            show_steps: Whether to show intermediate steps
            
        Returns:
            Dict with plan details and results
        """
        # Create plan
        plan = self.create_query_plan(user_query)
        
        if show_steps:
            print("\n" + "=" * 60)
            print("🔍 MULTI-STEP SQL GENERATION")
            print("=" * 60)
            print(f"Query: \"{user_query}\"")
            print(f"\nIdentified {len(plan.steps) - 1} exploration steps needed.")
        
        # Execute plan
        plan = self.execute_plan(plan)
        
        if show_steps:
            print("\n📊 EXECUTION SUMMARY:")
            for step in plan.steps:
                status = "✅" if step.executed and not step.error else "❌"
                print(f"  {status} Step {step.step_id}: {step.purpose}")
                if step.error:
                    print(f"      Error: {step.error}")
            
            print(f"\n📝 FINAL SQL:")
            print(f"  {plan.final_sql}")
            print("=" * 60 + "\n")
        
        return {
            "status": plan.status,
            "plan": plan.to_dict(),
            "final_sql": plan.final_sql,
            "final_result": plan.steps[-1].result if plan.steps else None
        }


class SmartNLToSQL:
    """
    High-level interface for intelligent NL-to-SQL conversion.
    
    This class provides:
    1. Simple queries: Direct generation
    2. Complex queries: Multi-step exploration
    3. Human review integration
    """
    
    def __init__(self, 
                 llm=None,
                 sql_executor: Optional[Callable[[str], Any]] = None,
                 human_review_enabled: bool = True):
        self.simple_generator = SQLGenerator(llm)
        self.multi_step_generator = MultiStepSQLGenerator(llm, sql_executor)
        self.human_review_enabled = human_review_enabled
        self.sql_executor = sql_executor
        
    def is_complex_query(self, query: str) -> bool:
        """Determine if a query needs multi-step processing."""
        # Simple heuristics for complexity
        complex_indicators = [
            r'\b(best|top|highest|lowest|most|least)\b',  # Superlatives
            r'\b(with|has|have|without)\s+\w+',  # Attribute requirements
            r'\b(in|at|near)\s+\w+',  # Location references
            r'\b(and|or)\b.*\b(and|or)\b',  # Multiple conditions
            r'\b(between|from|to)\b.*\b(and|to)\b',  # Ranges
        ]
        
        for pattern in complex_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def process(self, 
               user_query: str, 
               force_multi_step: bool = False) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            user_query: The natural language query
            force_multi_step: Force multi-step even for simple queries
            
        Returns:
            Dict with results and metadata
        """
        is_complex = self.is_complex_query(user_query) or force_multi_step
        
        if is_complex and self.sql_executor:
            # Use multi-step approach
            print("[SmartNLToSQL] Using multi-step query generation...")
            return self.multi_step_generator.run(user_query)
        else:
            # Use simple direct generation
            print("[SmartNLToSQL] Using direct query generation...")
            result = self.simple_generator.generate(user_query)
            
            return {
                "status": "completed" if result.is_valid else "failed",
                "approach": "direct",
                "final_sql": result.sql,
                "explanation": result.explanation,
                "confidence": result.confidence,
                "validation_errors": result.validation_errors
            }


# Convenience functions

def create_query_plan(user_query: str, llm=None) -> QueryPlan:
    """Create a query plan without executing it."""
    generator = MultiStepSQLGenerator(llm)
    return generator.create_query_plan(user_query)


def run_multi_step_query(
    user_query: str,
    sql_executor: Callable[[str], Any],
    llm=None
) -> Dict[str, Any]:
    """
    Convenience function to run a multi-step query.
    
    Args:
        user_query: Natural language query
        sql_executor: Function to execute SQL
        llm: Optional LLM instance
        
    Returns:
        Query results and metadata
    """
    generator = MultiStepSQLGenerator(llm, sql_executor)
    return generator.run(user_query)
