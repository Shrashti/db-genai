"""
Databricks Documentation RAG Ingestion Pipeline

This module provides high-quality semantic chunking and metadata extraction
for building an accurate RAG system on Databricks documentation.

Features:
- Read data from Databricks tables
- Semantic chunking using sentence embeddings
- Comprehensive metadata extraction
- Batch processing with PySpark
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, pandas_udf
from pyspark.sql.types import (
    ArrayType, StructType, StructField, StringType, 
    IntegerType, MapType, BooleanType
)
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize


# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# ============================================================================
# Data Reading Functions
# ============================================================================

def read_databricks_docs(table_name: str = "databricks_databricks_documentation_dataset.v01.docs"):
    """
    Read documentation data from Databricks table.
    
    Args:
        table_name: Fully qualified table name
    
    Returns:
        DataFrame with columns: id, url, content
    """
    spark = SparkSession.builder.getOrCreate()
    
    # Read the table
    df = spark.table(table_name)
    
    # Validate schema
    expected_columns = {'id', 'url', 'content'}
    actual_columns = set(df.columns)
    
    if not expected_columns.issubset(actual_columns):
        missing = expected_columns - actual_columns
        raise ValueError(f"Missing expected columns: {missing}")
    
    # Select only required columns and filter out nulls
    df = df.select('id', 'url', 'content').filter(
        col('content').isNotNull() & (col('content') != '')
    )
    
    print(f"Loaded {df.count()} documents from {table_name}")
    return df


# ============================================================================
# Semantic Chunking
# ============================================================================

@dataclass
class SemanticChunk:
    """Represents a semantic chunk of text with metadata."""
    text: str
    start_idx: int
    end_idx: int
    sentence_count: int
    char_count: int


class SemanticChunker:
    """
    Semantic chunking using sentence embeddings and similarity-based splitting.
    This approach maintains semantic coherence within chunks.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.5,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1000,
                 overlap_sentences: int = 1):
        """
        Initialize the semantic chunker.
        
        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Threshold for semantic similarity (0-1)
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling code blocks specially."""
        # Preserve code blocks
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, text)
        
        # Replace code blocks with placeholders
        text_with_placeholders = text
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            text_with_placeholders = text_with_placeholders.replace(block, placeholder, 1)
        
        # Split into sentences
        sentences = sent_tokenize(text_with_placeholders)
        
        # Restore code blocks
        restored_sentences = []
        for sent in sentences:
            for i, block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                sent = sent.replace(placeholder, block)
            restored_sentences.append(sent)
        
        return restored_sentences
    
    def _calculate_similarity_breaks(self, sentences: List[str]) -> List[int]:
        """Calculate break points based on semantic similarity."""
        if len(sentences) <= 1:
            return []
        
        # Generate embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        # Calculate similarities between consecutive sentences
        break_points = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            
            # If similarity drops below threshold, mark as break point
            if similarity < self.similarity_threshold:
                break_points.append(i + 1)
        
        return break_points
    
    def chunk_text(self, text: str) -> List[SemanticChunk]:
        """
        Chunk text semantically based on sentence embeddings.
        
        Args:
            text: Input text to chunk
        
        Returns:
            List of SemanticChunk objects
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) == 0:
            return []
        
        # Calculate semantic break points
        break_points = self._calculate_similarity_breaks(sentences)
        
        # Create chunks respecting break points and size constraints
        chunks = []
        current_chunk_sentences = []
        current_chunk_text = ""
        start_idx = 0
        
        for i, sentence in enumerate(sentences):
            # Check if we should break here
            should_break = (
                i in break_points or 
                len(current_chunk_text) + len(sentence) > self.max_chunk_size
            )
            
            if should_break and len(current_chunk_text) >= self.min_chunk_size:
                # Create chunk
                chunk = SemanticChunk(
                    text=current_chunk_text.strip(),
                    start_idx=start_idx,
                    end_idx=start_idx + len(current_chunk_text),
                    sentence_count=len(current_chunk_sentences),
                    char_count=len(current_chunk_text)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk_sentences) - self.overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_start:]
                current_chunk_text = " ".join(current_chunk_sentences) + " "
                start_idx = start_idx + len(current_chunk_text)
            
            current_chunk_sentences.append(sentence)
            current_chunk_text += sentence + " "
        
        # Add final chunk
        if current_chunk_text.strip():
            chunk = SemanticChunk(
                text=current_chunk_text.strip(),
                start_idx=start_idx,
                end_idx=start_idx + len(current_chunk_text),
                sentence_count=len(current_chunk_sentences),
                char_count=len(current_chunk_text)
            )
            chunks.append(chunk)
        
        return chunks


# ============================================================================
# Metadata Extraction
# ============================================================================

class MetadataExtractor:
    """
    Extract rich metadata from Databricks documentation content.
    This enhances RAG retrieval with structured information.
    """
    
    def __init__(self):
        # Patterns for different content types
        self.patterns = {
            'headers': r'^#{1,6}\s+(.+)$',
            'code_blocks': r'```([\w]*)?\n([\s\S]*?)```',
            'inline_code': r'`([^`]+)`',
            'links': r'\[([^\]]+)\]\(([^\)]+)\)',
            'bullet_points': r'^[\s]*[-*+]\s+(.+)$',
            'numbered_lists': r'^[\s]*\d+\.\s+(.+)$',
            'tables': r'\|(.+)\|',
            'api_endpoints': r'(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-{}:]+)',
            'sql_keywords': r'\b(SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b',
            'databricks_objects': r'\b(notebook|cluster|job|workspace|delta|mlflow|dbfs|catalog|schema|table|view|function)s?\b',
        }
    
    def extract_title(self, content: str) -> Optional[str]:
        """Extract the main title (first H1 header)."""
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('# '):
                return line.strip('# ').strip()
        return None
    
    def extract_headers(self, content: str) -> List[Dict[str, any]]:
        """Extract all headers with their levels."""
        headers = []
        lines = content.split('\n')
        for line in lines:
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headers.append({'level': level, 'text': text})
        return headers
    
    def extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks with language information."""
        code_blocks = []
        matches = re.finditer(self.patterns['code_blocks'], content, re.MULTILINE)
        for match in matches:
            language = match.group(1) or 'unknown'
            code = match.group(2).strip()
            code_blocks.append({
                'language': language,
                'code': code,
                'length': len(code)
            })
        return code_blocks
    
    def extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract all links with their text."""
        links = []
        matches = re.finditer(self.patterns['links'], content)
        for match in matches:
            links.append({
                'text': match.group(1),
                'url': match.group(2)
            })
        return links
    
    def extract_keywords(self, content: str) -> Dict[str, List[str]]:
        """Extract domain-specific keywords."""
        keywords = {
            'sql_keywords': list(set(re.findall(self.patterns['sql_keywords'], content, re.IGNORECASE))),
            'databricks_objects': list(set(re.findall(self.patterns['databricks_objects'], content, re.IGNORECASE))),
            'api_endpoints': [f"{m[0]} {m[1]}" for m in re.findall(self.patterns['api_endpoints'], content)]
        }
        return keywords
    
    def detect_document_type(self, content: str, url: str) -> str:
        """Detect the type of documentation."""
        content_lower = content.lower()
        url_lower = url.lower()
        
        # Check URL patterns
        if '/api/' in url_lower or 'api-reference' in url_lower:
            return 'api_reference'
        elif '/tutorial/' in url_lower or 'getting-started' in url_lower:
            return 'tutorial'
        elif '/guide/' in url_lower or 'how-to' in url_lower:
            return 'guide'
        
        # Check content patterns
        if re.search(r'\b(GET|POST|PUT|DELETE)\s+/', content):
            return 'api_reference'
        elif 'step 1' in content_lower or 'step 2' in content_lower:
            return 'tutorial'
        elif re.search(r'```(python|scala|sql|r)', content):
            return 'code_example'
        elif content.count('|') > 10:  # Likely has tables
            return 'reference'
        
        return 'general'
    
    def extract_metadata(self, doc_id: str, url: str, content: str) -> Dict:
        """
        Extract comprehensive metadata from a document.
        
        Args:
            doc_id: Document ID
            url: Document URL
            content: Document content
        
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            'doc_id': doc_id,
            'url': url,
            'title': self.extract_title(content),
            'document_type': self.detect_document_type(content, url),
            'headers': self.extract_headers(content),
            'code_blocks': self.extract_code_blocks(content),
            'links': self.extract_links(content),
            'keywords': self.extract_keywords(content),
            'statistics': {
                'char_count': len(content),
                'word_count': len(content.split()),
                'line_count': len(content.split('\n')),
                'header_count': len(self.extract_headers(content)),
                'code_block_count': len(self.extract_code_blocks(content)),
                'link_count': len(self.extract_links(content))
            }
        }
        
        # Extract URL components
        url_parts = url.rstrip('/').split('/')
        metadata['url_path'] = url_parts[-1] if url_parts else ''
        metadata['url_category'] = url_parts[-2] if len(url_parts) > 1 else ''
        
        return metadata


# ============================================================================
# Integration Functions
# ============================================================================

def process_document(doc_id: str, url: str, content: str, 
                    chunker: SemanticChunker, 
                    extractor: MetadataExtractor) -> List[Dict]:
    """
    Process a single document: extract metadata and create semantic chunks.
    
    Args:
        doc_id: Document ID
        url: Document URL
        content: Document content
        chunker: SemanticChunker instance
        extractor: MetadataExtractor instance
    
    Returns:
        List of dictionaries, each representing a chunk with metadata
    """
    # Extract document-level metadata
    doc_metadata = extractor.extract_metadata(doc_id, url, content)
    
    # Create semantic chunks
    chunks = chunker.chunk_text(content)
    
    # Combine chunks with metadata
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            'chunk_id': f"{doc_id}_chunk_{i}",
            'doc_id': doc_id,
            'url': url,
            'chunk_index': i,
            'total_chunks': len(chunks),
            'text': chunk.text,
            'char_count': chunk.char_count,
            'sentence_count': chunk.sentence_count,
            'start_idx': chunk.start_idx,
            'end_idx': chunk.end_idx,
            # Document-level metadata
            'doc_title': doc_metadata['title'],
            'doc_type': doc_metadata['document_type'],
            'url_category': doc_metadata['url_category'],
            'url_path': doc_metadata['url_path'],
            # Simplified metadata for the chunk
            'has_code': any(cb['code'] in chunk.text for cb in doc_metadata['code_blocks']),
            'keywords': doc_metadata['keywords']
        }
        processed_chunks.append(chunk_data)
    
    return processed_chunks


def create_processing_udf(chunker: SemanticChunker, extractor: MetadataExtractor):
    """
    Create a PySpark UDF for batch processing documents.
    
    Args:
        chunker: SemanticChunker instance
        extractor: MetadataExtractor instance
    
    Returns:
        PySpark UDF for processing documents
    """
    # Define schema for processed chunks
    chunk_schema = ArrayType(StructType([
        StructField('chunk_id', StringType(), False),
        StructField('doc_id', StringType(), False),
        StructField('url', StringType(), False),
        StructField('chunk_index', IntegerType(), False),
        StructField('total_chunks', IntegerType(), False),
        StructField('text', StringType(), False),
        StructField('char_count', IntegerType(), False),
        StructField('sentence_count', IntegerType(), False),
        StructField('start_idx', IntegerType(), False),
        StructField('end_idx', IntegerType(), False),
        StructField('doc_title', StringType(), True),
        StructField('doc_type', StringType(), False),
        StructField('url_category', StringType(), True),
        StructField('url_path', StringType(), True),
        StructField('has_code', StringType(), False),
        StructField('keywords', MapType(StringType(), ArrayType(StringType())), True)
    ]))
    
    def process_document_udf(doc_id: str, url: str, content: str):
        """UDF wrapper for document processing."""
        try:
            chunks = process_document(doc_id, url, content, chunker, extractor)
            # Convert boolean to string for Spark compatibility
            for chunk in chunks:
                chunk['has_code'] = str(chunk['has_code'])
            return chunks
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
            return []
    
    return udf(process_document_udf, chunk_schema)


def process_all_documents(docs_df, output_table: str = None):
    """
    Process all documents with semantic chunking and metadata extraction.
    
    Args:
        docs_df: Input DataFrame with columns: id, url, content
        output_table: Optional table name to save results
    
    Returns:
        DataFrame with processed chunks
    """
    # Initialize chunker and extractor
    chunker = SemanticChunker(
        similarity_threshold=0.5,
        min_chunk_size=200,
        max_chunk_size=1000,
        overlap_sentences=2
    )
    extractor = MetadataExtractor()
    
    # Create UDF
    process_udf = create_processing_udf(chunker, extractor)
    
    # Process all documents
    print("Processing all documents with semantic chunking and metadata extraction...")
    processed_df = docs_df.withColumn(
        'chunks',
        process_udf(col('id'), col('url'), col('content'))
    )
    
    # Explode chunks into separate rows
    chunks_df = processed_df.select(
        explode(col('chunks')).alias('chunk')
    ).select(
        col('chunk.chunk_id'),
        col('chunk.doc_id'),
        col('chunk.url'),
        col('chunk.chunk_index'),
        col('chunk.total_chunks'),
        col('chunk.text'),
        col('chunk.char_count'),
        col('chunk.sentence_count'),
        col('chunk.doc_title'),
        col('chunk.doc_type'),
        col('chunk.url_category'),
        col('chunk.has_code'),
        col('chunk.keywords')
    )
    
    print(f"\nTotal chunks created: {chunks_df.count()}")
    
    # Save to table if specified
    if output_table:
        chunks_df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(output_table)
        print(f"Saved processed chunks to {output_table}")
    
    return chunks_df


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # This section shows example usage when running as a script
    
    # Read data
    docs_df = read_databricks_docs()
    
    # Process documents
    chunks_df = process_all_documents(
        docs_df,
        output_table="databricks_databricks_documentation_dataset.v01.processed_chunks"
    )
    
    # Display statistics
    print("\nProcessing Statistics:")
    chunks_df.groupBy('doc_type').count().show()
    
    print("\nChunk size distribution:")
    chunks_df.select('char_count').describe().show()
    
    # Display sample
    chunks_df.limit(10).show(truncate=False)
