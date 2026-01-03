# Databricks notebook source
# MAGIC %md
# MAGIC # Complete RAG Ingestion with Title, Keywords, and Summary
# MAGIC 
# MAGIC This notebook demonstrates the complete pipeline with:
# MAGIC - Semantic chunking
# MAGIC - Metadata extraction (title, keywords)
# MAGIC - AI-powered summary generation

# COMMAND ----------

# MAGIC %pip install sentence-transformers scikit-learn nltk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Import modules
from ingestion_pipeline import (
    read_databricks_docs,
    SemanticChunker,
    MetadataExtractor,
    process_document
)
from summary_generator import (
    SummaryGenerator,
    extract_enhanced_keywords,
    add_summaries_to_chunks
)
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType, MapType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Read Data

# COMMAND ----------

# Read documentation
docs_df = read_databricks_docs("databricks_databricks_documentation_dataset.v01.docs")
display(docs_df.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Process Single Document (Example)

# COMMAND ----------

# Initialize components
chunker = SemanticChunker(
    similarity_threshold=0.5,
    min_chunk_size=200,
    max_chunk_size=1000,
    overlap_sentences=2
)
extractor = MetadataExtractor()
summary_gen = SummaryGenerator(method='extractive')

# Process one document
sample_doc = docs_df.first()
chunks = process_document(
    doc_id=sample_doc['id'],
    url=sample_doc['url'],
    content=sample_doc['content'],
    chunker=chunker,
    extractor=extractor
)

# Add summaries
chunks_with_summary = add_summaries_to_chunks(chunks, use_llm=False)

# Display results
print(f"Document chunked into {len(chunks_with_summary)} pieces\n")
for i, chunk in enumerate(chunks_with_summary[:2]):
    print(f"=== Chunk {i+1} ===")
    print(f"Title: {chunk['doc_title']}")
    print(f"Keywords: {chunk['enhanced_keywords'][:5]}")
    print(f"Summary: {chunk['summary']}")
    print(f"Text preview: {chunk['text'][:200]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Enhanced Processing UDF

# COMMAND ----------

# Define enhanced schema with summary
enhanced_chunk_schema = ArrayType(StructType([
    StructField('chunk_id', StringType(), False),
    StructField('doc_id', StringType(), False),
    StructField('url', StringType(), False),
    StructField('chunk_index', IntegerType(), False),
    StructField('total_chunks', IntegerType(), False),
    StructField('text', StringType(), False),
    StructField('char_count', IntegerType(), False),
    StructField('sentence_count', IntegerType(), False),
    StructField('doc_title', StringType(), True),
    StructField('summary', StringType(), True),  # NEW: Summary field
    StructField('keywords', ArrayType(StringType()), True),  # NEW: Enhanced keywords
    StructField('doc_type', StringType(), False),
    StructField('url_category', StringType(), True),
    StructField('has_code', StringType(), False)
]))

def process_with_summary_udf(doc_id: str, url: str, content: str):
    """Enhanced UDF with summary generation."""
    try:
        # Process document
        chunks = process_document(doc_id, url, content, chunker, extractor)
        
        # Add summaries
        chunks = add_summaries_to_chunks(chunks, use_llm=False)
        
        # Format for Spark
        formatted_chunks = []
        for chunk in chunks:
            formatted_chunks.append({
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'url': chunk['url'],
                'chunk_index': chunk['chunk_index'],
                'total_chunks': chunk['total_chunks'],
                'text': chunk['text'],
                'char_count': chunk['char_count'],
                'sentence_count': chunk['sentence_count'],
                'doc_title': chunk['doc_title'],
                'summary': chunk['summary'],
                'keywords': chunk['enhanced_keywords'],
                'doc_type': chunk['doc_type'],
                'url_category': chunk['url_category'],
                'has_code': str(chunk['has_code'])
            })
        
        return formatted_chunks
    except Exception as e:
        print(f"Error processing {doc_id}: {str(e)}")
        return []

# Register UDF
process_enhanced_udf = udf(process_with_summary_udf, enhanced_chunk_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Process All Documents

# COMMAND ----------

# Process all documents
print("Processing all documents with chunking, metadata, and summaries...")
processed_df = docs_df.withColumn(
    'chunks',
    process_enhanced_udf(col('id'), col('url'), col('content'))
)

# Explode into individual chunks
final_chunks_df = processed_df.select(
    explode(col('chunks')).alias('chunk')
).select(
    col('chunk.chunk_id'),
    col('chunk.doc_id'),
    col('chunk.url'),
    col('chunk.chunk_index'),
    col('chunk.total_chunks'),
    col('chunk.text'),
    col('chunk.doc_title').alias('title'),  # Renamed for clarity
    col('chunk.summary'),
    col('chunk.keywords'),
    col('chunk.doc_type'),
    col('chunk.char_count'),
    col('chunk.has_code')
)

print(f"\nTotal chunks created: {final_chunks_df.count()}")
display(final_chunks_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Save to Delta Table

# COMMAND ----------

# Save processed chunks
output_table = "databricks_databricks_documentation_dataset.v01.processed_chunks_with_metadata"

final_chunks_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(output_table)

print(f"✅ Saved {final_chunks_df.count()} chunks to {output_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Analyze Results

# COMMAND ----------

# Show sample with all metadata
print("Sample chunks with complete metadata:")
display(
    final_chunks_df
    .select('chunk_id', 'title', 'summary', 'keywords', 'doc_type', 'text')
    .limit(5)
)

# COMMAND ----------

# Document type distribution
print("Document Type Distribution:")
final_chunks_df.groupBy('doc_type').count().orderBy('count', ascending=False).show()

# COMMAND ----------

# Keyword analysis
from pyspark.sql.functions import explode as spark_explode, count, desc

print("Top Keywords Across All Chunks:")
final_chunks_df.select(spark_explode('keywords').alias('keyword')) \
    .groupBy('keyword') \
    .count() \
    .orderBy(desc('count')) \
    .limit(20) \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Example Retrieval Queries

# COMMAND ----------

# Example 1: Find chunks about "Delta"
delta_chunks = final_chunks_df.filter(
    col('text').contains('Delta') | 
    col('keywords').contains('delta')
)
print(f"Found {delta_chunks.count()} chunks about Delta")
display(delta_chunks.select('title', 'summary', 'keywords').limit(5))

# COMMAND ----------

# Example 2: Find tutorial-type documents
tutorial_chunks = final_chunks_df.filter(col('doc_type') == 'tutorial')
print(f"Found {tutorial_chunks.count()} tutorial chunks")
display(tutorial_chunks.select('title', 'summary').limit(5))

# COMMAND ----------

# Example 3: Find chunks with code examples
code_chunks = final_chunks_df.filter(col('has_code') == 'True')
print(f"Found {code_chunks.count()} chunks with code")
display(code_chunks.select('title', 'summary', 'text').limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC ✅ **Data processed successfully!**
# MAGIC 
# MAGIC Your chunks now include:
# MAGIC - **Title**: Document title for context
# MAGIC - **Summary**: Concise description of chunk content
# MAGIC - **Keywords**: Enhanced keywords for retrieval
# MAGIC - **Text**: Semantically coherent chunk content
# MAGIC - **Metadata**: Document type, URL, code presence, etc.
# MAGIC 
# MAGIC **Next Steps for RAG:**
# MAGIC 1. Generate embeddings for the `text` field
# MAGIC 2. Store in vector database with metadata
# MAGIC 3. Implement hybrid search (vector + keyword + metadata filters)
# MAGIC 4. Build chatbot with retrieval + LLM generation
