"""
Summary Generation Module for RAG Chunks

This module provides AI-powered summary generation for document chunks
to enhance retrieval quality in RAG systems.
"""

from typing import List, Dict
import re


class SummaryGenerator:
    """
    Generate concise summaries for document chunks.
    Supports both extractive and AI-based summarization.
    """
    
    def __init__(self, method: str = 'extractive'):
        """
        Initialize summary generator.
        
        Args:
            method: 'extractive' (rule-based) or 'ai' (LLM-based)
        """
        self.method = method
    
    def _extractive_summary(self, text: str, title: str = None, max_length: int = 150) -> str:
        """
        Generate extractive summary using first sentences and key phrases.
        
        Args:
            text: Chunk text
            title: Document title
            max_length: Maximum summary length
        
        Returns:
            Extractive summary
        """
        # Clean text
        text = text.strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Start with title context if available
        if title:
            summary_parts = [f"From '{title}':"]
        else:
            summary_parts = []
        
        # Add first sentence (usually most important)
        if sentences:
            first_sentence = sentences[0].strip()
            # Truncate if too long
            if len(first_sentence) > max_length - len(' '.join(summary_parts)):
                first_sentence = first_sentence[:max_length - len(' '.join(summary_parts)) - 3] + "..."
            summary_parts.append(first_sentence)
        
        summary = ' '.join(summary_parts)
        
        # Ensure it's not too long
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."
        
        return summary
    
    def _ai_summary_prompt(self, text: str, title: str = None) -> str:
        """
        Generate prompt for AI-based summarization.
        
        Args:
            text: Chunk text
            title: Document title
        
        Returns:
            Prompt for LLM
        """
        prompt = f"""Summarize the following Databricks documentation chunk in one concise sentence (max 150 characters).
Focus on the main action, concept, or instruction.

{f"Document Title: {title}" if title else ""}

Text:
{text}

Summary:"""
        return prompt
    
    def generate_summary(self, text: str, title: str = None, doc_type: str = None) -> str:
        """
        Generate summary for a chunk.
        
        Args:
            text: Chunk text
            title: Document title
            doc_type: Document type (guide, tutorial, api_reference, etc.)
        
        Returns:
            Generated summary
        """
        if self.method == 'extractive':
            return self._extractive_summary(text, title)
        elif self.method == 'ai':
            # For AI method, return the prompt
            # In production, you would call an LLM here
            return self._ai_summary_prompt(text, title)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def generate_summary_with_llm(self, text: str, title: str = None, 
                                  llm_function=None) -> str:
        """
        Generate summary using an LLM.
        
        Args:
            text: Chunk text
            title: Document title
            llm_function: Function that takes a prompt and returns LLM response
        
        Returns:
            AI-generated summary
        """
        if llm_function is None:
            # Fallback to extractive if no LLM provided
            return self._extractive_summary(text, title)
        
        prompt = self._ai_summary_prompt(text, title)
        summary = llm_function(prompt)
        
        # Ensure summary is not too long
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        return summary.strip()


def extract_enhanced_keywords(text: str, title: str = None, 
                              existing_keywords: Dict[str, List[str]] = None) -> List[str]:
    """
    Extract enhanced keywords combining multiple sources.
    
    Args:
        text: Chunk text
        title: Document title
        existing_keywords: Keywords from MetadataExtractor
    
    Returns:
        List of relevant keywords
    """
    keywords = []
    
    # Add title words (if available)
    if title:
        title_words = [w.lower() for w in re.findall(r'\b\w+\b', title) 
                      if len(w) > 3 and w.lower() not in ['with', 'from', 'that', 'this']]
        keywords.extend(title_words[:5])  # Top 5 from title
    
    # Add existing keywords
    if existing_keywords:
        for key_type, key_list in existing_keywords.items():
            keywords.extend([k.lower() for k in key_list[:3]])  # Top 3 from each type
    
    # Extract important nouns and verbs from text
    # Simple heuristic: capitalized words and common Databricks terms
    text_keywords = re.findall(r'\b(Delta|Spark|MLflow|DBFS|Databricks|cluster|notebook|table|schema|catalog|workspace|job|SQL|Python|Scala|API)\b', text, re.IGNORECASE)
    keywords.extend([k.lower() for k in set(text_keywords)[:5]])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return unique_keywords[:10]  # Return top 10 keywords


# Example usage with Databricks
def create_llm_summary_function():
    """
    Create an LLM summary function using Databricks Foundation Models.
    
    Returns:
        Function that generates summaries using LLM
    """
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        
        w = WorkspaceClient()
        
        def llm_summarize(prompt: str) -> str:
            """Call Databricks Foundation Model for summarization."""
            try:
                response = w.serving_endpoints.query(
                    name="databricks-meta-llama-3-1-70b-instruct",  # or your preferred model
                    messages=[
                        ChatMessage(
                            role=ChatMessageRole.SYSTEM,
                            content="You are a helpful assistant that creates concise summaries of technical documentation."
                        ),
                        ChatMessage(
                            role=ChatMessageRole.USER,
                            content=prompt
                        )
                    ],
                    max_tokens=100,
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM call failed: {e}, falling back to extractive summary")
                # Fallback to extractive
                generator = SummaryGenerator(method='extractive')
                text = prompt.split("Text:")[-1].split("Summary:")[0].strip()
                return generator._extractive_summary(text)
        
        return llm_summarize
    
    except ImportError:
        print("Databricks SDK not available, using extractive summaries")
        return None


# Integration example
def add_summaries_to_chunks(chunks: List[Dict], use_llm: bool = False) -> List[Dict]:
    """
    Add summaries to processed chunks.
    
    Args:
        chunks: List of chunk dictionaries
        use_llm: Whether to use LLM for summarization
    
    Returns:
        Chunks with added summaries
    """
    if use_llm:
        llm_function = create_llm_summary_function()
        generator = SummaryGenerator(method='ai')
    else:
        llm_function = None
        generator = SummaryGenerator(method='extractive')
    
    for chunk in chunks:
        # Generate summary
        if use_llm and llm_function:
            chunk['summary'] = generator.generate_summary_with_llm(
                text=chunk['text'],
                title=chunk.get('doc_title'),
                llm_function=llm_function
            )
        else:
            chunk['summary'] = generator.generate_summary(
                text=chunk['text'],
                title=chunk.get('doc_title'),
                doc_type=chunk.get('doc_type')
            )
        
        # Enhance keywords
        chunk['enhanced_keywords'] = extract_enhanced_keywords(
            text=chunk['text'],
            title=chunk.get('doc_title'),
            existing_keywords=chunk.get('keywords')
        )
    
    return chunks
