# view_vectors.py

from pymilvus import connections, Collection

def view_vectors():
    # Step 1: Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Step 2: Access the Collection
    collection = Collection("knowledge_base_embeddings")  # Replace "knowledge_base" with your actual collection name if different

    # Step 3: Load the Collection to ensure it's ready for querying
    collection.load()

    # Step 4: Perform a Simple Query to get all vectors
    results = collection.query(expr="", output_fields=["embedding"], offset=0, limit=100)

    # Step 5: View the Embeddings
    for result in results:
        print(f"Embedding: {result['embedding']}")

if __name__ == "__main__":
    view_vectors()
