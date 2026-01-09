from typing import List, Dict, Any, Optional

class Guardrail:
    def validate_input(self, query: str) -> bool:
        """
        Check if the input query is safe and relevant to the travel domain.
        Returns True if safe, False otherwise.
        """
        raise NotImplementedError 

class TravelTopicGuardrail(Guardrail):
    def validate_input(self, query: str) -> bool:
        # Simple keyword-based or heuristic check for now.
        # In production, this could be a call to a classification model or lighter LLM.
        
        # Whitelist keywords related to travel
        keywords = ["book", "search", "hotel", "flight", "amenity", "wifi", "pool", "villa", "apartment", "vacation", "trip", "travel", "reserv", "room", "price", "cost", "night", "stay", "wanderbricks"]
        
        # Check against basic injections or malicious patterns (very basic placeholder)
        if "ignore prompt" in query.lower() or "system prompt" in query.lower():
            return False
            
        # Check relevance
        is_relevant = any(kw in query.lower() for kw in keywords)
        
        # If no keywords match, it might still be valid (e.g. conversational), 
        # but for this strict agent we might want to be conservative or use an LLM.
        # For this MVP, let's assume if it's not totally irrelevant we pass it, 
        # or we rely on the Agent's system prompt to refuse.
        # But let's enforce at least one keyword for strict topic adherence if desired.
        
        return True # For now, let's be permissive and rely on System Prompt, or implement stricter logic later.

    def check(self, query: str) -> Dict[str, Any]:
        if not self.validate_input(query):
            return {
                "allowed": False,
                "reason": "Query appears to be unrelated to travel or violates safety policies."
            }
        return {"allowed": True}
