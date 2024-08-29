# Configuration file

# Knowledge base files
KNOWLEDGE_BASE_FILES = [
    "knowledge_base/data_types.csv",
    "knowledge_base/file_handling.csv"
]

# Embedding model
EMBEDDING_MODEL = "microsoft/codebert-base"

# Milvus collection configuration
COLLECTION_NAME = "knowledge_base_embeddings"
EMBEDDING_DIM = 768  # Typically 768 for BERT-like models
