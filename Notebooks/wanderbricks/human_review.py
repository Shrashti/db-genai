"""
Human Review Module for Wanderbricks NL-to-SQL Layer

This module provides a human-in-the-loop workflow for SQL query review.
Users can approve, modify, or reject generated SQL before execution.
"""

import json
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class ReviewStatus(Enum):
    """Status of a review request."""
    PENDING = "pending"
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ReviewAction(Enum):
    """Actions a reviewer can take."""
    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"
    CANCEL = "cancel"
    REGENERATE = "regenerate"


@dataclass
class ReviewRequest:
    """A request for human review of a generated SQL query."""
    request_id: str
    original_query: str
    generated_sql: str
    explanation: str
    tables_used: List[str]
    confidence: float
    status: ReviewStatus = ReviewStatus.PENDING
    modified_sql: Optional[str] = None
    reviewer_notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    iteration: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "original_query": self.original_query,
            "generated_sql": self.generated_sql,
            "explanation": self.explanation,
            "tables_used": self.tables_used,
            "confidence": self.confidence,
            "status": self.status.value,
            "modified_sql": self.modified_sql,
            "reviewer_notes": self.reviewer_notes,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "iteration": self.iteration
        }
    
    def get_display_message(self) -> str:
        """Generate a human-readable message for display."""
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 SQL REVIEW REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Your Query: "{self.original_query}"

📊 Generated SQL:
```sql
{self.generated_sql}
```

💡 Explanation: {self.explanation}

📋 Tables Used: {', '.join(self.tables_used)}
📈 Confidence: {self.confidence:.0%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Choose an action:
  [1] ✅ APPROVE - Execute this query
  [2] ✏️  MODIFY  - Edit the SQL before execution
  [3] 🔄 REGENERATE - Ask for a different query
  [4] ❌ CANCEL  - Cancel this request
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


@dataclass
class ReviewResult:
    """Result of a review action."""
    request: ReviewRequest
    action: ReviewAction
    final_sql: Optional[str]
    should_execute: bool
    feedback: Optional[str] = None


class HumanReviewNode:
    """
    Human-in-the-loop review node for SQL queries.
    
    This class manages the review workflow:
    1. Present generated SQL to user
    2. Wait for user action (approve/modify/reject)
    3. Return appropriate result for execution
    
    Can be used in both synchronous (CLI) and asynchronous (callback) modes.
    """
    
    def __init__(self, 
                 review_callback: Optional[Callable[[ReviewRequest], ReviewResult]] = None,
                 auto_approve_threshold: float = 0.95):
        """
        Initialize the human review node.
        
        Args:
            review_callback: Optional callback for custom review UI.
                           If None, uses CLI-based review.
            auto_approve_threshold: Confidence threshold above which to skip review.
                                   Set to 1.0 to always require review.
        """
        self.review_callback = review_callback
        self.auto_approve_threshold = auto_approve_threshold
        self.pending_reviews: Dict[str, ReviewRequest] = {}
        self.review_history: List[ReviewRequest] = []
        
    def create_review_request(self,
                             original_query: str,
                             generated_sql: str,
                             explanation: str,
                             tables_used: List[str],
                             confidence: float,
                             iteration: int = 0) -> ReviewRequest:
        """Create a new review request."""
        request = ReviewRequest(
            request_id=str(uuid.uuid4())[:8],
            original_query=original_query,
            generated_sql=generated_sql,
            explanation=explanation,
            tables_used=tables_used,
            confidence=confidence,
            iteration=iteration
        )
        self.pending_reviews[request.request_id] = request
        return request
    
    def should_skip_review(self, confidence: float) -> bool:
        """Check if review can be skipped based on confidence."""
        return confidence >= self.auto_approve_threshold
    
    def review(self, request: ReviewRequest) -> ReviewResult:
        """
        Process a review request.
        
        Uses the callback if provided, otherwise falls back to CLI.
        """
        if self.review_callback:
            return self.review_callback(request)
        else:
            return self._cli_review(request)
    
    def _cli_review(self, request: ReviewRequest) -> ReviewResult:
        """
        CLI-based review for interactive environments.
        
        This is used when no callback is provided.
        """
        print(request.get_display_message())
        
        while True:
            try:
                choice = input("Enter choice (1-4): ").strip()
                
                if choice == "1":
                    # Approve
                    request.status = ReviewStatus.APPROVED
                    request.reviewed_at = datetime.now()
                    self._archive_request(request)
                    return ReviewResult(
                        request=request,
                        action=ReviewAction.APPROVE,
                        final_sql=request.generated_sql,
                        should_execute=True
                    )
                    
                elif choice == "2":
                    # Modify
                    print("\nEnter your modified SQL (end with a line containing only 'END'):")
                    lines = []
                    while True:
                        line = input()
                        if line.strip() == "END":
                            break
                        lines.append(line)
                    
                    modified_sql = "\n".join(lines)
                    notes = input("Any notes about the modification? (optional): ").strip()
                    
                    request.status = ReviewStatus.MODIFIED
                    request.modified_sql = modified_sql
                    request.reviewer_notes = notes
                    request.reviewed_at = datetime.now()
                    self._archive_request(request)
                    
                    return ReviewResult(
                        request=request,
                        action=ReviewAction.MODIFY,
                        final_sql=modified_sql,
                        should_execute=True,
                        feedback=notes
                    )
                    
                elif choice == "3":
                    # Regenerate
                    feedback = input("How should the query be different? ").strip()
                    request.status = ReviewStatus.REJECTED
                    request.reviewer_notes = feedback
                    request.reviewed_at = datetime.now()
                    self._archive_request(request)
                    
                    return ReviewResult(
                        request=request,
                        action=ReviewAction.REGENERATE,
                        final_sql=None,
                        should_execute=False,
                        feedback=feedback
                    )
                    
                elif choice == "4":
                    # Cancel
                    request.status = ReviewStatus.CANCELLED
                    request.reviewed_at = datetime.now()
                    self._archive_request(request)
                    
                    return ReviewResult(
                        request=request,
                        action=ReviewAction.CANCEL,
                        final_sql=None,
                        should_execute=False
                    )
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
                    
            except KeyboardInterrupt:
                print("\n\nReview cancelled by user.")
                request.status = ReviewStatus.CANCELLED
                self._archive_request(request)
                return ReviewResult(
                    request=request,
                    action=ReviewAction.CANCEL,
                    final_sql=None,
                    should_execute=False
                )
    
    def _archive_request(self, request: ReviewRequest):
        """Archive a completed review request."""
        if request.request_id in self.pending_reviews:
            del self.pending_reviews[request.request_id]
        self.review_history.append(request)
    
    def get_pending_reviews(self) -> List[ReviewRequest]:
        """Get all pending review requests."""
        return list(self.pending_reviews.values())
    
    def get_review_history(self) -> List[ReviewRequest]:
        """Get review history."""
        return self.review_history
    
    def submit_review(self, 
                     request_id: str, 
                     action: str,
                     modified_sql: Optional[str] = None,
                     notes: Optional[str] = None) -> Optional[ReviewResult]:
        """
        Submit a review for a pending request (for async/callback usage).
        
        Args:
            request_id: ID of the review request
            action: One of 'approve', 'modify', 'reject', 'cancel'
            modified_sql: Modified SQL if action is 'modify'
            notes: Optional reviewer notes
            
        Returns:
            ReviewResult if request found, None otherwise
        """
        if request_id not in self.pending_reviews:
            return None
        
        request = self.pending_reviews[request_id]
        request.reviewed_at = datetime.now()
        
        action_enum = ReviewAction(action.lower())
        
        if action_enum == ReviewAction.APPROVE:
            request.status = ReviewStatus.APPROVED
            final_sql = request.generated_sql
            should_execute = True
            
        elif action_enum == ReviewAction.MODIFY:
            if not modified_sql:
                raise ValueError("modified_sql required for 'modify' action")
            request.status = ReviewStatus.MODIFIED
            request.modified_sql = modified_sql
            request.reviewer_notes = notes
            final_sql = modified_sql
            should_execute = True
            
        elif action_enum == ReviewAction.REJECT or action_enum == ReviewAction.REGENERATE:
            request.status = ReviewStatus.REJECTED
            request.reviewer_notes = notes
            final_sql = None
            should_execute = False
            
        else:  # CANCEL
            request.status = ReviewStatus.CANCELLED
            final_sql = None
            should_execute = False
        
        self._archive_request(request)
        
        return ReviewResult(
            request=request,
            action=action_enum,
            final_sql=final_sql,
            should_execute=should_execute,
            feedback=notes
        )


class ReviewWorkflowState:
    """
    State machine for the human review workflow.
    
    States:
    - INIT: Initial state
    - GENERATING: Generating SQL
    - PENDING_REVIEW: Waiting for human review
    - APPROVED: Query approved, ready for execution
    - EXECUTING: Executing the query
    - COMPLETED: Workflow completed
    - FAILED: Workflow failed
    """
    
    def __init__(self):
        self.state = "INIT"
        self.current_request: Optional[ReviewRequest] = None
        self.sql_result: Optional[Any] = None
        self.execution_result: Optional[Any] = None
        self.error: Optional[str] = None
        self.iterations: int = 0
        self.max_iterations: int = 3
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "state": self.state,
            "current_request": self.current_request.to_dict() if self.current_request else None,
            "iterations": self.iterations,
            "error": self.error
        }
    
    def can_regenerate(self) -> bool:
        """Check if regeneration is allowed."""
        return self.iterations < self.max_iterations


def create_review_callback_for_notebook() -> Callable[[ReviewRequest], ReviewResult]:
    """
    Create a callback suitable for Databricks notebooks.
    
    Uses dbutils widgets for input in notebook environment.
    Falls back to print/input for other environments.
    """
    def notebook_callback(request: ReviewRequest) -> ReviewResult:
        # Display the review request
        print(request.get_display_message())
        
        # In a real notebook, you would use dbutils.widgets
        # For now, we simulate with input
        try:
            from IPython.display import display, HTML
            # Display formatted HTML in notebook
            html_content = f"""
            <div style="border: 2px solid #4CAF50; padding: 20px; margin: 10px; border-radius: 10px;">
                <h3>🔍 SQL Review Required</h3>
                <p><strong>Query:</strong> {request.original_query}</p>
                <pre style="background: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 5px;">
{request.generated_sql}
                </pre>
                <p><strong>Confidence:</strong> {request.confidence:.0%}</p>
                <p><strong>Tables:</strong> {', '.join(request.tables_used)}</p>
            </div>
            """
            display(HTML(html_content))
        except ImportError:
            pass
        
        # For CLI fallback
        choice = input("Enter action (approve/modify/reject/cancel): ").strip().lower()
        
        if choice == "approve":
            request.status = ReviewStatus.APPROVED
            return ReviewResult(
                request=request,
                action=ReviewAction.APPROVE,
                final_sql=request.generated_sql,
                should_execute=True
            )
        elif choice == "modify":
            modified = input("Enter modified SQL: ").strip()
            request.status = ReviewStatus.MODIFIED
            request.modified_sql = modified
            return ReviewResult(
                request=request,
                action=ReviewAction.MODIFY,
                final_sql=modified,
                should_execute=True
            )
        elif choice == "reject":
            feedback = input("Feedback for regeneration: ").strip()
            request.status = ReviewStatus.REJECTED
            return ReviewResult(
                request=request,
                action=ReviewAction.REGENERATE,
                final_sql=None,
                should_execute=False,
                feedback=feedback
            )
        else:
            request.status = ReviewStatus.CANCELLED
            return ReviewResult(
                request=request,
                action=ReviewAction.CANCEL,
                final_sql=None,
                should_execute=False
            )
    
    return notebook_callback


# Convenience function for integration
def require_human_review(sql: str, 
                        query: str, 
                        explanation: str,
                        tables: List[str],
                        confidence: float,
                        callback: Optional[Callable] = None) -> ReviewResult:
    """
    Convenience function to require human review for a SQL query.
    
    Args:
        sql: Generated SQL query
        query: Original natural language query
        explanation: Explanation of what the query does
        tables: List of tables used
        confidence: Confidence score 0-1
        callback: Optional custom review callback
        
    Returns:
        ReviewResult with the review outcome
    """
    node = HumanReviewNode(review_callback=callback)
    request = node.create_review_request(
        original_query=query,
        generated_sql=sql,
        explanation=explanation,
        tables_used=tables,
        confidence=confidence
    )
    return node.review(request)
