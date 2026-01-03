"""
Databricks Documentation RAG Ingestion Pipeline

This module provides high-quality semantic chunking and metadata extraction
for building an accurate RAG system on Databricks documentation.

Features:
- Read data from Databricks tables
- Semantic chunking using sentence embeddings
- Comprehensive metadata extraction
- Batch processing with PySpark

DATABRICKS SETUP:
-----------------
Before running this notebook in Databricks, execute the following in a cell:

    # Install required packages
    %pip install sentence-transformers scikit-learn nltk
    
    # Download NLTK data
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    # Restart Python to use updated packages
    dbutils.library.restartPython()

Then you can import and use this module.
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


# Initialize NLTK (with fallback for Databricks)
def _initialize_nltk():
    """Initialize NLTK data. Call this explicitly in Databricks if needed."""
    resources = ['punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download NLTK resource '{resource}': {e}")
                print(f"Please run: nltk.download('{resource}') manually in your notebook")

# Try to initialize NLTK automatically
_initialize_nltk()


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
                 overlap_sentences: int = 1,
                 cache_folder: str = None):
        """
        Initialize the semantic chunker.
        
        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Threshold for semantic similarity (0-1)
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            overlap_sentences: Number of sentences to overlap between chunks
            cache_folder: Directory to cache downloaded models (optional, uses default if None)
        """
        import os
        
        # Set cache directory if specified
        if cache_folder:
            print(f"Setting cache directory to: {cache_folder}")
            os.environ['TRANSFORMERS_CACHE'] = cache_folder
            os.environ['HF_HOME'] = cache_folder
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_folder
        
        try:
            print(f"Loading sentence-transformers model: {model_name}")
            if cache_folder:
                self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
            else:
                self.model = SentenceTransformer(model_name)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
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
        # Convert keywords to JSON string for Spark compatibility
        import json
        keywords_json = json.dumps(doc_metadata['keywords'])
        
        chunk_data = {
            'chunk_id': f"{doc_id}_chunk_{i}",
            'doc_id': str(doc_id),
            'url': url,
            'chunk_index': i,
            'total_chunks': len(chunks),
            'text': chunk.text,
            'char_count': chunk.char_count,
            'sentence_count': chunk.sentence_count,
            # Document-level metadata
            'doc_title': doc_metadata['title'] if doc_metadata['title'] else '',
            'doc_type': doc_metadata['document_type'],
            'url_category': doc_metadata['url_category'] if doc_metadata['url_category'] else '',
            'url_path': doc_metadata['url_path'] if doc_metadata['url_path'] else '',
            # Simplified metadata for the chunk
            'has_code': 'true' if any(cb['code'] in chunk.text for cb in doc_metadata['code_blocks']) else 'false',
            'keywords': keywords_json  # Store as JSON string
        }
        processed_chunks.append(chunk_data)
    
    return processed_chunks


def create_processing_udf():
    """
    Create a PySpark pandas UDF for batch processing documents.
    Uses pandas_udf to avoid model serialization issues.
    
    Returns:
        PySpark pandas UDF for processing documents
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType, MapType
    
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
    
    @pandas_udf(chunk_schema)
    def process_documents_batch(doc_ids: pd.Series, urls: pd.Series, contents: pd.Series) -> pd.Series:
        """Pandas UDF wrapper for batch document processing."""
        import sys
        
        print(f"[UDF] Starting batch processing of {len(doc_ids)} documents", file=sys.stderr)
        
        # Set cache directory to writable location for Databricks serverless
        import os
        cache_dir = '/tmp/sentence_transformers'
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        
        print(f"[UDF] Cache directory set to: {cache_dir}", file=sys.stderr)
        
        # Initialize chunker and extractor INSIDE the UDF
        # This avoids serialization issues with the sentence-transformers model
        try:
            print("[UDF] Initializing SemanticChunker...", file=sys.stderr)
            chunker = SemanticChunker(
                similarity_threshold=0.5,
                min_chunk_size=200,
                max_chunk_size=1000,
                overlap_sentences=2,
                cache_folder=cache_dir
            )
            print("[UDF] SemanticChunker initialized successfully", file=sys.stderr)
        except Exception as e:
            print(f"[UDF] ERROR initializing SemanticChunker: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Return empty results if model fails to load
            return pd.Series([[] for _ in range(len(doc_ids))])
        
        try:
            print("[UDF] Initializing MetadataExtractor...", file=sys.stderr)
            extractor = MetadataExtractor()
            print("[UDF] MetadataExtractor initialized successfully", file=sys.stderr)
        except Exception as e:
            print(f"[UDF] ERROR initializing MetadataExtractor: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return pd.Series([[] for _ in range(len(doc_ids))])
        
        results = []
        for idx, (doc_id, url, content) in enumerate(zip(doc_ids, urls, contents)):
            try:
                if content is None or len(str(content).strip()) == 0:
                    print(f"[UDF] Doc {doc_id}: Empty content, skipping", file=sys.stderr)
                    results.append([])
                    continue
                
                print(f"[UDF] Doc {doc_id}: Processing {len(str(content))} chars", file=sys.stderr)
                chunks = process_document(str(doc_id), str(url), str(content), chunker, extractor)
                print(f"[UDF] Doc {doc_id}: Generated {len(chunks)} chunks", file=sys.stderr)
                
                # Convert boolean to string for Spark compatibility
                for chunk in chunks:
                    chunk['has_code'] = str(chunk['has_code'])
                results.append(chunks)
            except Exception as e:
                print(f"[UDF] ERROR processing document {doc_id}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                results.append([])
        
        return pd.Series(results)
    
    return process_documents_batch

# ============================================================================
# Model Pre-download Helper
# ============================================================================

def download_model(model_name: str = 'all-MiniLM-L6-v2', cache_folder: str = None):
    """
    Pre-download the sentence-transformers model.
    Run this once before processing to ensure the model is cached.
    
    Args:
        model_name: Name of the sentence-transformers model
        cache_folder: Optional cache directory (uses default if None)
    
    Returns:
        True if successful, False otherwise
    """
    import os
    
    if cache_folder:
        print(f"Setting cache directory to: {cache_folder}")
        os.environ['TRANSFORMERS_CACHE'] = cache_folder
        os.environ['HF_HOME'] = cache_folder
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_folder
    
    try:
        print(f"Downloading model: {model_name}")
        print("This may take a minute on first run (~90MB download)...")
        
        if cache_folder:
            model = SentenceTransformer(model_name, cache_folder=cache_folder)
        else:
            model = SentenceTransformer(model_name)
        
        print(f"✓ Model '{model_name}' downloaded and cached successfully!")
        print(f"  Max sequence length: {model.max_seq_length}")
        print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Document Processing Functions
# ============================================================================

def process_all_documents(docs_df, output_table: str = None, batch_size: int = 100):
    """
    Process all documents with semantic chunking and metadata extraction.
    Uses driver-based processing for reliability on Databricks serverless.
    
    Args:
        docs_df: Input DataFrame with columns: id, url, content
        output_table: Optional table name to save results
        batch_size: Number of documents to collect and process at once (default: 100)
    
    Returns:
        DataFrame with processed chunks
    """
    total_docs = docs_df.count()
    print(f"Input: {total_docs} documents")
    print("Initializing models in driver...")
    
    # Initialize models once in driver (using default cache location)
    chunker = SemanticChunker(
        similarity_threshold=0.5,
        min_chunk_size=200,
        max_chunk_size=1000,
        overlap_sentences=2
        # cache_folder not specified - uses default location
    )
    extractor = MetadataExtractor()
    print("✓ Models initialized successfully")
    
    # Process documents in batches
    all_chunks = []
    print(f"\nProcessing {total_docs} documents in batches of {batch_size}...")
    
    for batch_num in range(0, total_docs, batch_size):
        batch_docs = docs_df.offset(batch_num).limit(batch_size).collect()
        batch_id = batch_num // batch_size + 1
        
        print(f"\nBatch {batch_id}/{(total_docs + batch_size - 1) // batch_size}: Processing {len(batch_docs)} documents...")
        batch_chunks_count = 0
        
        for doc in batch_docs:
            try:
                doc_id = str(doc['id'])
                url = str(doc['url'])
                content = str(doc['content'])
                
                # Skip empty documents
                if not content or len(content.strip()) < 100:
                    print(f"  Doc {doc_id}: Skipped (content too short: {len(content) if content else 0} chars)")
                    continue
                
                # Process document
                print(f"  Doc {doc_id}: Processing {len(content)} chars...")
                chunks = process_document(doc_id, url, content, chunker, extractor)
                print(f"  Doc {doc_id}: Generated {len(chunks)} chunks")
                
                if len(chunks) == 0:
                    print(f"  ⚠️ Doc {doc_id}: WARNING - 0 chunks generated!")
                
                # Convert to dict format for DataFrame
                for chunk in chunks:
                    chunk['has_code'] = str(chunk['has_code'])
                    all_chunks.append(chunk)
                    batch_chunks_count += 1
                
                # Progress update
                if len(all_chunks) % 500 == 0 and len(all_chunks) > 0:
                    print(f"  ✓ Total chunks so far: {len(all_chunks)}")
                    
            except Exception as e:
                print(f"  ✗ Error processing doc {doc['id']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Batch {batch_id} complete: Generated {batch_chunks_count} chunks from {len(batch_docs)} documents")
    
    print(f"\n✓ Generated {len(all_chunks)} total chunks from {total_docs} documents")
    
    if len(all_chunks) == 0:
        print("\n⚠️  WARNING: No chunks were created!")
        print("Check if documents have valid content")
        # Return empty DataFrame with proper schema
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        return spark.createDataFrame([], schema=StructType([]))
    
    # Convert to Spark DataFrame
    print("Converting to Spark DataFrame...")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    chunks_df = spark.createDataFrame(all_chunks)
    
    # Save to table if specified
    if output_table:
        print(f"\nSaving {len(all_chunks)} chunks to {output_table}...")
        chunks_df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .option("delta.enableChangeDataFeed", "true") \
            .saveAsTable(output_table)
        print(f"✓ Saved processed chunks to {output_table}")
        print(f"✓ Change Data Feed enabled for vector index creation")
    
    return chunks_df


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # This section shows example usage when running as a script
    
    # STEP 1: Pre-download the model (run this once)
    print("=" * 70)
    print("STEP 1: Pre-downloading sentence-transformers model")
    print("=" * 70)
    download_model()  # Uses default cache location
    # Or specify a cache folder: download_model(cache_folder='/dbfs/tmp/models')
    
    # STEP 2: Read data
    print("\n" + "=" * 70)
    print("STEP 2: Reading documentation data")
    print("=" * 70)
    docs_df = read_databricks_docs()
    
    # STEP 3: Process documents
    print("\n" + "=" * 70)
    print("STEP 3: Processing documents with semantic chunking")
    print("=" * 70)
    chunks_df = process_all_documents(
        docs_df,
        output_table="databricks_databricks_documentation_dataset.v01.processed_chunks",
        batch_size=100
    )
    
    # STEP 4: Display statistics
    print("\n" + "=" * 70)
    print("STEP 4: Results")
    print("=" * 70)
    print("\nProcessing Statistics:")
    chunks_df.groupBy('doc_type').count().show()
    
    print("\nChunk size distribution:")
    chunks_df.select('char_count').describe().show()
    
    # Display sample
    print("\nSample chunks:")
    chunks_df.limit(10).show(truncate=50)

