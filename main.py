# main.py

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataTypeM
from utils.chunking import Chunker
from utils.embedding import EmbeddingGenerator
import csv
import config

# Establish a connection to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the schema for the collection
def create_collection():
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
    
    collection_schema = CollectionSchema(fields=[id_field, embedding_field], description="Knowledge Base Embeddings")
    
    collection = Collection(name=config.COLLECTION_NAME, schema=collection_schema)
    collection.create_index(field_name="embedding", index_params={"metric_type": "L2"})
    collection.load()
    return collection

def insert_data(collection, embeddings):
    # Insert only the embeddings as expected by the schema
    entities = [
        embeddings  # Expecting a list of embedding vectors
    ]
    collection.insert(entities)

def main():
    # Initialize the chunking mechanism
    chunker = Chunker()

    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator()

    # Create Milvus collection
    collection = create_collection()

    # Process knowledge base CSV files
    knowledge_base_files = config.KNOWLEDGE_BASE_FILES

    for csv_file in knowledge_base_files:
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Select the text to chunk based on the relevant column in your CSV
                text_to_chunk = row["Description"]  # Adjust this column name as needed
                chunks = chunker.create_chunks(text_to_chunk)

                for chunk in chunks:
                    # Generate embedding for each chunk
                    embedding = embedding_generator.generate_embedding(chunk)

                    # Store the embedding into Milvus
                    insert_data(collection, embedding)

    print("Knowledge base embeddings have been successfully stored in Milvus.")

if __name__ == "__main__":
    main()
