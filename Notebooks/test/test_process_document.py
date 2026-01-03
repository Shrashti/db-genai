"""
Test the full process_document function to see if it works end-to-end.
"""

print("=" * 70)
print("TEST: Full process_document function")
print("=" * 70)

from ingestion_pipeline import SemanticChunker, MetadataExtractor, process_document

# Get sample document
sample = spark.table("databricks_databricks_documentation_dataset.v01.docs").limit(1).collect()[0]

print(f"Sample doc ID: {sample['id']}")
print(f"Content length: {len(sample['content'])} chars")

# Initialize chunker and extractor
print("\nInitializing models...")
chunker = SemanticChunker(
    similarity_threshold=0.5,
    min_chunk_size=200,
    max_chunk_size=1000,
    overlap_sentences=2
)
extractor = MetadataExtractor()
print("✓ Models initialized")

# Process the document
print("\nProcessing document...")
chunks = process_document(
    str(sample['id']),
    str(sample['url']),
    str(sample['content']),
    chunker,
    extractor
)

print(f"✓ Generated {len(chunks)} chunks")

if len(chunks) > 0:
    print(f"\nFirst chunk structure:")
    first_chunk = chunks[0]
    for key, value in first_chunk.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}... ({len(value)} chars)")
        else:
            print(f"  {key}: {value}")
    
    # Try to create a DataFrame from chunks
    print("\nTrying to create Spark DataFrame...")
    try:
        chunks_df = spark.createDataFrame(chunks)
        print(f"✓ DataFrame created with {chunks_df.count()} rows")
        print("\nDataFrame schema:")
        chunks_df.printSchema()
        print("\nSample row:")
        chunks_df.show(1, truncate=50, vertical=True)
    except Exception as e:
        print(f"✗ Error creating DataFrame: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️ No chunks generated!")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
