"""
Test script to debug chunking issues outside of UDF context.
Run this in a Databricks notebook cell to test if the chunking logic works.
"""

# Setup
import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/sentence_transformers'
os.environ['HF_HOME'] = '/tmp/sentence_transformers'

# Test 1: Check if we can load a single document
print("=" * 60)
print("TEST 1: Loading sample document from table")
print("=" * 60)

sample_doc = spark.table("databricks_databricks_documentation_dataset.v01.docs").limit(1).collect()[0]
print(f"Document ID: {sample_doc['id']}")
print(f"URL: {sample_doc['url']}")
print(f"Content length: {len(sample_doc['content'])} chars")
print(f"Content preview: {sample_doc['content'][:200]}...")

# Test 2: Try to initialize the chunker
print("\n" + "=" * 60)
print("TEST 2: Initializing SemanticChunker")
print("=" * 60)

try:
    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/tmp/sentence_transformers')
    print("✓ Model loaded successfully!")
    print(f"Model max sequence length: {model.max_seq_length}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try semantic chunking
print("\n" + "=" * 60)
print("TEST 3: Testing semantic chunking on sample document")
print("=" * 60)

try:
    # Import from the ingestion pipeline
    import sys
    sys.path.append('/Workspace/Repos/shrashti.90@gmail.com/db-genai/RAG')
    from ingestion_pipeline import SemanticChunker, MetadataExtractor, process_document
    
    print("Creating chunker...")
    chunker = SemanticChunker(
        similarity_threshold=0.5,
        min_chunk_size=200,
        max_chunk_size=1000,
        overlap_sentences=2,
        cache_folder='/tmp/sentence_transformers'
    )
    print("✓ Chunker created")
    
    print("Creating metadata extractor...")
    extractor = MetadataExtractor()
    print("✓ Extractor created")
    
    print(f"\nProcessing document {sample_doc['id']}...")
    chunks = process_document(
        str(sample_doc['id']),
        str(sample_doc['url']),
        str(sample_doc['content']),
        chunker,
        extractor
    )
    
    print(f"✓ Generated {len(chunks)} chunks!")
    
    if len(chunks) > 0:
        print(f"\nFirst chunk preview:")
        print(f"  - Chunk ID: {chunks[0]['chunk_id']}")
        print(f"  - Text length: {chunks[0]['char_count']} chars")
        print(f"  - Sentences: {chunks[0]['sentence_count']}")
        print(f"  - Text preview: {chunks[0]['text'][:150]}...")
    else:
        print("⚠ No chunks generated - investigating why...")
        
        # Try direct chunking
        print("\nTrying direct chunk_text method...")
        semantic_chunks = chunker.chunk_text(sample_doc['content'])
        print(f"Direct chunking result: {len(semantic_chunks)} chunks")
        
except Exception as e:
    print(f"✗ Error during chunking: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
