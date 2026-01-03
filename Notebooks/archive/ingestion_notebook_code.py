# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Documentation RAG Ingestion Pipeline
# MAGIC This notebook implements a high-quality RAG ingestion pipeline with:
# MAGIC - Data reading from Databricks tables
# MAGIC - Semantic chunking for context-aware splitting
# MAGIC - Metadata extraction for enhanced retrieval

# COMMAND ----------

# MAGIC %pip install sentence-transformers scikit-learn nltk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Import the ingestion pipeline module
from ingestion_pipeline import (
    read_databricks_docs,
    SemanticChunker,
    MetadataExtractor,
    process_document,
    process_all_documents
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Read Data from Databricks Table

# COMMAND ----------

# Read documentation data
docs_df = read_databricks_docs("databricks_databricks_documentation_dataset.v01.docs")
display(docs_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Test Semantic Chunking on a Sample Document

# COMMAND ----------

# Initialize semantic chunker
chunker = SemanticChunker(
    model_name='all-MiniLM-L6-v2',
    similarity_threshold=0.5,
    min_chunk_size=200,
    max_chunk_size=1000,
    overlap_sentences=2
)

# Test on a sample document
sample_doc = docs_df.first()
chunks = chunker.chunk_text(sample_doc['content'])

print(f"Created {len(chunks)} semantic chunks from sample document")
print("\nChunk Statistics:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}:")
    print(f"  - Characters: {chunk.char_count}")
    print(f"  - Sentences: {chunk.sentence_count}")
    print(f"  - Preview: {chunk.text[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Test Metadata Extraction

# COMMAND ----------

# Initialize metadata extractor
extractor = MetadataExtractor()

# Extract metadata from sample document
metadata = extractor.extract_metadata(
    doc_id=sample_doc['id'],
    url=sample_doc['url'],
    content=sample_doc['content']
)

print("Extracted Metadata:")
print(f"  Title: {metadata['title']}")
print(f"  Document Type: {metadata['document_type']}")
print(f"  URL Category: {metadata['url_category']}")
print(f"  URL Path: {metadata['url_path']}")
print(f"\nHeaders ({len(metadata['headers'])}):")
for header in metadata['headers'][:5]:
    print(f"  {'#' * header['level']} {header['text']}")

print(f"\nCode Blocks: {len(metadata['code_blocks'])}")
for i, cb in enumerate(metadata['code_blocks'][:3]):
    print(f"  Block {i+1}: {cb['language']} ({cb['length']} chars)")

print(f"\nLinks: {len(metadata['links'])}")
for link in metadata['links'][:5]:
    print(f"  - {link['text']}: {link['url']}")

print(f"\nKeywords:")
for key, values in metadata['keywords'].items():
    print(f"  {key}: {values[:5]}")

print(f"\nStatistics:")
for key, value in metadata['statistics'].items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Process Single Document with Both Functions

# COMMAND ----------

# Process a single document
processed_chunks = process_document(
    doc_id=sample_doc['id'],
    url=sample_doc['url'],
    content=sample_doc['content'],
    chunker=chunker,
    extractor=extractor
)

print(f"Processed document into {len(processed_chunks)} chunks with metadata")
print(f"\nFirst chunk details:")
chunk = processed_chunks[0]
for key, value in chunk.items():
    if key != 'text':  # Skip full text for brevity
        print(f"  {key}: {value}")
print(f"\nText preview: {chunk['text'][:300]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Process All Documents (Batch Processing)

# COMMAND ----------

# Process all documents and save to Delta table
output_table = "databricks_databricks_documentation_dataset.v01.processed_chunks"

chunks_df = process_all_documents(
    docs_df=docs_df,
    output_table=output_table
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Analyze Results

# COMMAND ----------

# Display processing statistics
print("Document Type Distribution:")
chunks_df.groupBy('doc_type').count().orderBy('count', ascending=False).show()

# COMMAND ----------

print("Chunk Size Statistics:")
chunks_df.select('char_count', 'sentence_count').describe().show()

# COMMAND ----------

# Display sample chunks
print("Sample Processed Chunks:")
display(chunks_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Query Processed Chunks

# COMMAND ----------

# Example: Find chunks with code
code_chunks = chunks_df.filter(col('has_code') == 'True')
print(f"Chunks containing code: {code_chunks.count()}")
display(code_chunks.limit(5))

# COMMAND ----------

# Example: Find chunks by document type
api_chunks = chunks_df.filter(col('doc_type') == 'api_reference')
print(f"API reference chunks: {api_chunks.count()}")
display(api_chunks.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Export for Vector Database (Optional)

# COMMAND ----------

# Option 1: Export to JSON for vector database ingestion
# chunks_df.write.format("json").mode("overwrite").save("/dbfs/path/to/output/chunks.json")

# Option 2: Export to Parquet
# chunks_df.write.format("parquet").mode("overwrite").save("/dbfs/path/to/output/chunks.parquet")

# Option 3: Already saved to Delta table (recommended)
print(f"Processed chunks are available in table: {output_table}")
print("You can now use these chunks for embedding generation and vector database ingestion.")
