# RAG Ingestion Pipeline for Databricks Documentation

This implementation provides a production-ready RAG (Retrieval-Augmented Generation) ingestion pipeline specifically designed for Databricks documentation.

## Features

### 1. **Semantic Chunking**
- Uses sentence embeddings (SentenceTransformers) to create semantically coherent chunks
- Maintains context by splitting at semantic boundaries rather than arbitrary character limits
- Configurable similarity threshold for fine-tuning chunk boundaries
- Handles code blocks specially to preserve their integrity
- Supports overlapping chunks for better context preservation

### 2. **Comprehensive Metadata Extraction**
- **Document Classification**: Automatically detects document type (API reference, tutorial, guide, code example, etc.)
- **Structure Extraction**: Headers, code blocks, links, tables
- **Keyword Extraction**: SQL keywords, Databricks objects, API endpoints
- **URL Analysis**: Extracts category and path information from URLs
- **Statistics**: Character count, word count, line count, etc.

### 3. **Batch Processing**
- PySpark UDF for distributed processing
- Handles large datasets efficiently
- Saves results to Delta tables for easy querying

## Files

- **`ingestion_pipeline.py`**: Core module with all functions
- **`ingestion_notebook_code.py`**: Databricks notebook code (copy into your notebook)
- **`Ingestion.ipynb`**: Original notebook file

## Installation

### Required Libraries

```python
%pip install sentence-transformers scikit-learn nltk
```

After installation, restart the Python environment:
```python
dbutils.library.restartPython()
```

## Usage

### Option 1: Import the Module (Recommended)

1. Upload `ingestion_pipeline.py` to your Databricks workspace
2. Import and use the functions:

```python
from ingestion_pipeline import (
    read_databricks_docs,
    SemanticChunker,
    MetadataExtractor,
    process_all_documents
)

# Read data
docs_df = read_databricks_docs("databricks_databricks_documentation_dataset.v01.docs")

# Process all documents
chunks_df = process_all_documents(
    docs_df=docs_df,
    output_table="your_catalog.your_schema.processed_chunks"
)
```

### Option 2: Copy Code into Notebook

Copy the contents of `ingestion_notebook_code.py` into your Databricks notebook cells.

## Function Reference

### `read_databricks_docs(table_name)`
Reads documentation from a Databricks table.

**Parameters:**
- `table_name` (str): Fully qualified table name

**Returns:** PySpark DataFrame with columns: id, url, content

### `SemanticChunker`
Creates semantically coherent chunks using sentence embeddings.

**Parameters:**
- `model_name` (str): SentenceTransformer model name (default: 'all-MiniLM-L6-v2')
- `similarity_threshold` (float): Semantic similarity threshold 0-1 (default: 0.5)
- `min_chunk_size` (int): Minimum characters per chunk (default: 100)
- `max_chunk_size` (int): Maximum characters per chunk (default: 1000)
- `overlap_sentences` (int): Sentences to overlap between chunks (default: 1)

**Methods:**
- `chunk_text(text: str) -> List[SemanticChunk]`: Chunk text semantically

### `MetadataExtractor`
Extracts rich metadata from documentation content.

**Methods:**
- `extract_title(content: str) -> Optional[str]`: Extract main title
- `extract_headers(content: str) -> List[Dict]`: Extract all headers
- `extract_code_blocks(content: str) -> List[Dict]`: Extract code blocks
- `extract_links(content: str) -> List[Dict]`: Extract links
- `extract_keywords(content: str) -> Dict[str, List[str]]`: Extract keywords
- `detect_document_type(content: str, url: str) -> str`: Detect document type
- `extract_metadata(doc_id, url, content) -> Dict`: Extract all metadata

### `process_document(doc_id, url, content, chunker, extractor)`
Process a single document with chunking and metadata extraction.

**Returns:** List of dictionaries with chunk data and metadata

### `process_all_documents(docs_df, output_table)`
Process all documents in batch using PySpark.

**Parameters:**
- `docs_df`: Input DataFrame
- `output_table` (optional): Table name to save results

**Returns:** DataFrame with processed chunks

## Output Schema

The processed chunks DataFrame contains:

| Column | Type | Description |
|--------|------|-------------|
| chunk_id | string | Unique chunk identifier |
| doc_id | string | Source document ID |
| url | string | Source document URL |
| chunk_index | int | Index of chunk within document |
| total_chunks | int | Total chunks in document |
| text | string | Chunk text content |
| char_count | int | Character count |
| sentence_count | int | Sentence count |
| doc_title | string | Document title |
| doc_type | string | Document type classification |
| url_category | string | URL category |
| has_code | string | Whether chunk contains code |
| keywords | map | Extracted keywords |

## Configuration Recommendations

### For High Accuracy RAG:

```python
chunker = SemanticChunker(
    similarity_threshold=0.5,  # Higher = more splits
    min_chunk_size=200,        # Ensure meaningful chunks
    max_chunk_size=1000,       # Prevent too large chunks
    overlap_sentences=2        # Better context preservation
)
```

### For Different Document Types:

- **API Documentation**: Lower similarity threshold (0.4) for more granular chunks
- **Tutorials**: Higher max_chunk_size (1500) to keep steps together
- **Code Examples**: Ensure code blocks stay intact (handled automatically)

## Next Steps for RAG System

1. **Generate Embeddings**: Use the chunk text to generate embeddings
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   embeddings = model.encode(chunks_df.select('text').collect())
   ```

2. **Store in Vector Database**: Index chunks with embeddings in a vector database (e.g., FAISS, Pinecone, Weaviate)

3. **Build Retrieval System**: Use metadata filters for hybrid search:
   - Filter by `doc_type` for specific documentation types
   - Use `has_code` to prioritize code examples
   - Leverage `keywords` for keyword-based filtering

4. **Implement RAG Pipeline**: Combine retrieval with LLM generation

## Performance Considerations

- **Batch Size**: Process documents in batches for large datasets
- **Model Loading**: Load SentenceTransformer model once and reuse
- **Caching**: Cache processed chunks to avoid reprocessing
- **Partitioning**: Partition output table by `doc_type` for faster queries

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size or use smaller embedding model

### Issue: Chunks too small/large
**Solution**: Adjust `min_chunk_size` and `max_chunk_size` parameters

### Issue: Poor semantic boundaries
**Solution**: Adjust `similarity_threshold` (lower = fewer splits, higher = more splits)

## Example Queries

```python
# Find all API reference chunks
api_chunks = chunks_df.filter(col('doc_type') == 'api_reference')

# Find chunks with code examples
code_chunks = chunks_df.filter(col('has_code') == 'True')

# Find chunks about specific topics
delta_chunks = chunks_df.filter(
    array_contains(col('keywords.databricks_objects'), 'delta')
)
```

## License

This code is provided as-is for use with Databricks documentation processing.
