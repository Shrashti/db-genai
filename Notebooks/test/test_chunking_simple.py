"""
Simple test to verify chunking works without all the complexity.
Run this in your Databricks notebook to test if the basic chunking logic works.
"""

# Test 1: Can we create chunks with simple text splitting?
print("=" * 70)
print("TEST 1: Simple text splitting (no ML model)")
print("=" * 70)

def simple_chunk(text, chunk_size=500):
    """Simple chunking - split every N characters."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append({
            'chunk_id': f'chunk_{i}',
            'text': text[i:i+chunk_size],
            'char_count': len(text[i:i+chunk_size])
        })
    return chunks

# Get a sample document
sample = spark.table("databricks_databricks_documentation_dataset.v01.docs").limit(1).collect()[0]
print(f"Sample doc ID: {sample['id']}")
print(f"Content length: {len(sample['content'])} chars")

simple_chunks = simple_chunk(sample['content'])
print(f"✓ Simple chunking created {len(simple_chunks)} chunks")

# Test 2: Can we use the SemanticChunker?
print("\n" + "=" * 70)
print("TEST 2: Semantic chunking with sentence-transformers")
print("=" * 70)

try:
    from ingestion_pipeline import SemanticChunker
    
    print("Creating SemanticChunker...")
    chunker = SemanticChunker(
        similarity_threshold=0.5,
        min_chunk_size=200,
        max_chunk_size=1000,
        overlap_sentences=2
    )
    print("✓ SemanticChunker created")
    
    print(f"Chunking sample document...")
    semantic_chunks = chunker.chunk_text(sample['content'])
    print(f"✓ Semantic chunking created {len(semantic_chunks)} chunks")
    
    if len(semantic_chunks) > 0:
        print(f"\nFirst chunk preview:")
        print(f"  Text length: {semantic_chunks[0].char_count} chars")
        print(f"  Sentences: {semantic_chunks[0].sentence_count}")
        print(f"  Preview: {semantic_chunks[0].text[:150]}...")
    else:
        print("⚠️ WARNING: Semantic chunking returned 0 chunks!")
        print("This is the problem - let's debug why...")
        
        # Debug: Check sentence splitting
        sentences = chunker._split_into_sentences(sample['content'])
        print(f"\nDebug info:")
        print(f"  Total sentences: {len(sentences)}")
        if len(sentences) > 0:
            print(f"  First sentence: {sentences[0][:100]}...")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
