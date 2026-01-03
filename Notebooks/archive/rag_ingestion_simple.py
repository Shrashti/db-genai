# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Ingestion Pipeline - Simple Version
# MAGIC 
# MAGIC This notebook demonstrates:
# MAGIC 1. Reading data from Databricks table
# MAGIC 2. Semantic chunking
# MAGIC 3. Metadata extraction (title, keywords, summary)

# COMMAND ----------

# MAGIC %pip install sentence-transformers scikit-learn nltk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and Setup

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import re

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Data from Databricks Table

# COMMAND ----------

def read_data(table_name="databricks_databricks_documentation_dataset.v01.docs"):
    """Read documentation from Databricks table."""
    spark = SparkSession.builder.getOrCreate()
    df = spark.table(table_name)
    
    # Filter out empty content
    df = df.select('id', 'url', 'content').filter(
        col('content').isNotNull() & (col('content') != '')
    )
    
    print(f"Loaded {df.count()} documents")
    return df

# Read data
docs_df = read_data()
display(docs_df.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Semantic Chunking Function

# COMMAND ----------

class SemanticChunker:
    """Chunk text based on semantic similarity."""
    
    def __init__(self, max_chunk_size=1000, similarity_threshold=0.5):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
    
    def chunk_text(self, text):
        """Split text into semantic chunks."""
        # Split into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        
        # Get embeddings
        embeddings = self.model.encode(sentences)
        
        # Find break points based on similarity
        chunks = []
        current_chunk = sentences[0]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            # Check if we should start a new chunk
            if similarity < self.similarity_threshold or len(current_chunk) + len(sentences[i]) > self.max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentences[i]
            else:
                current_chunk += " " + sentences[i]
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# Test chunking
chunker = SemanticChunker(max_chunk_size=1000, similarity_threshold=0.5)
sample_text = docs_df.first()['content']
test_chunks = chunker.chunk_text(sample_text)
print(f"Created {len(test_chunks)} chunks")
print(f"First chunk: {test_chunks[0][:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Metadata Extraction Functions

# COMMAND ----------

def extract_title(content):
    """Extract title from first H1 header."""
    lines = content.split('\n')
    for line in lines:
        if line.strip().startswith('# '):
            return line.strip('# ').strip()
    return None

def extract_keywords(text):
    """Extract important keywords."""
    keywords = []
    
    # Databricks-specific terms
    databricks_terms = re.findall(
        r'\b(Delta|Spark|MLflow|DBFS|cluster|notebook|table|workspace|catalog|schema)\b',
        text,
        re.IGNORECASE
    )
    keywords.extend([k.lower() for k in set(databricks_terms)])
    
    # SQL keywords
    sql_keywords = re.findall(
        r'\b(SELECT|FROM|WHERE|CREATE|INSERT|UPDATE|DELETE)\b',
        text,
        re.IGNORECASE
    )
    keywords.extend([k.lower() for k in set(sql_keywords)])
    
    return list(set(keywords))[:10]  # Top 10 unique keywords

def generate_summary(text, title=None, max_length=150):
    """Generate a simple extractive summary."""
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    
    # Use first sentence as summary
    summary = sentences[0]
    
    # Add title context if available
    if title:
        summary = f"From '{title}': {summary}"
    
    # Truncate if too long
    if len(summary) > max_length:
        summary = summary[:max_length-3] + "..."
    
    return summary

# Test metadata extraction
sample_doc = docs_df.first()
print(f"Title: {extract_title(sample_doc['content'])}")
print(f"Keywords: {extract_keywords(sample_doc['content'])}")
print(f"Summary: {generate_summary(sample_doc['content'][:500])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Process Documents

# COMMAND ----------

def process_document(doc_id, url, content):
    """Process a single document: chunk + extract metadata."""
    # Extract document-level metadata
    title = extract_title(content)
    
    # Create chunks
    chunks = chunker.chunk_text(content)
    
    # Add metadata to each chunk
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            'chunk_id': f"{doc_id}_chunk_{i}",
            'doc_id': doc_id,
            'url': url,
            'chunk_index': i,
            'text': chunk_text,
            'title': title,
            'keywords': extract_keywords(chunk_text),
            'summary': generate_summary(chunk_text, title)
        })
    
    return result

# Test on one document
sample = docs_df.first()
processed = process_document(sample['id'], sample['url'], sample['content'])
print(f"Processed into {len(processed)} chunks\n")
print("First chunk:")
print(f"  Title: {processed[0]['title']}")
print(f"  Keywords: {processed[0]['keywords']}")
print(f"  Summary: {processed[0]['summary']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Batch Process All Documents

# COMMAND ----------

# Define schema for output
chunk_schema = ArrayType(StructType([
    StructField('chunk_id', StringType()),
    StructField('doc_id', StringType()),
    StructField('url', StringType()),
    StructField('chunk_index', IntegerType()),
    StructField('text', StringType()),
    StructField('title', StringType()),
    StructField('keywords', ArrayType(StringType())),
    StructField('summary', StringType())
]))

# Create UDF
def process_udf(doc_id, url, content):
    try:
        return process_document(doc_id, url, content)
    except Exception as e:
        print(f"Error processing {doc_id}: {e}")
        return []

process_docs_udf = udf(process_udf, chunk_schema)

# Process all documents
processed_df = docs_df.withColumn(
    'chunks',
    process_docs_udf(col('id'), col('url'), col('content'))
)

# Explode into individual chunks
final_df = processed_df.select(
    explode(col('chunks')).alias('chunk')
).select(
    col('chunk.chunk_id'),
    col('chunk.doc_id'),
    col('chunk.url'),
    col('chunk.text'),
    col('chunk.title'),
    col('chunk.keywords'),
    col('chunk.summary')
)

print(f"Total chunks: {final_df.count()}")
display(final_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Results

# COMMAND ----------

# Save to Delta table
output_table = "processed_chunks"

final_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable(output_table)

print(f"✅ Saved to table: {output_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Example Queries

# COMMAND ----------

# Query 1: Find chunks about Delta
delta_chunks = final_df.filter(
    col('text').contains('Delta') | 
    col('keywords').contains('delta')
)
print(f"Chunks about Delta: {delta_chunks.count()}")
display(delta_chunks.select('title', 'summary').limit(5))

# COMMAND ----------

# Query 2: Search by keyword
spark_chunks = final_df.filter(col('keywords').contains('spark'))
print(f"Chunks about Spark: {spark_chunks.count()}")
display(spark_chunks.select('title', 'summary').limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC ✅ **Pipeline Complete!**
# MAGIC 
# MAGIC - Read data from Databricks table
# MAGIC - Semantically chunked documents
# MAGIC - Extracted title, keywords, and summary
# MAGIC - Saved to `processed_chunks` table
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC 1. Generate embeddings for `text` field
# MAGIC 2. Store in vector database
# MAGIC 3. Build RAG chatbot
