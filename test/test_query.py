# test_query.py

import sys
import os
import numpy as np

# Ensure the parent directory is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import connections, Collection
from utils.embedding import EmbeddingGenerator  # Adjusted import path based on your structure
import config

def query_vector(query_text):
    # Step 1: Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Step 2: Access the Collection
    collection = Collection(config.COLLECTION_NAME)

    # Step 3: Generate Embedding for the Query Text
    embedding_generator = EmbeddingGenerator()
    query_embedding = embedding_generator.generate_embedding(query_text)

    # Ensure the embedding is a list of floats
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()  # Convert numpy array to list
    elif not isinstance(query_embedding, list):
        raise ValueError("Embedding generator returned an unexpected format. Expected list of floats.")

    print(f"Generated Embedding: {query_embedding}")
    print(f"Type: {type(query_embedding)}, Length: {len(query_embedding)}")

    # Milvus expects the embeddings to be a list of floats
    if len(query_embedding) == 1 and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]  # Unwrap the embedding if it's wrapped in an extra list

    if not all(isinstance(x, float) for x in query_embedding):
        raise ValueError("Each item in the embedding must be a float.")

    # Step 4: Load the Collection to ensure it's ready for querying
    collection.load()

    # Step 5: Search for the Most Similar Embedding in Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "embedding", param=search_params, limit=1)

    # Step 6: Retrieve and Display the Most Similar Result
    if results:
        for result in results[0]:
            similar_embedding = result.entity.get("embedding")
            print(f"Most similar embedding found: {similar_embedding}")
            print(f"Equivalent .NET Code: string")  # Replace with actual logic to convert the embedding to .NET code if necessary
    else:
        print("No results found.")

if __name__ == "__main__":
    # Example query
    query_vector("cobol code PIC X")
