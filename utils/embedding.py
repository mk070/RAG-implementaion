# embedding.py
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
import config

class EmbeddingGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)
        self.collection = self.create_collection()

    def create_collection(self):
        collection = Collection(
            config.COLLECTION_NAME,
            schema=CollectionSchema(
                fields=[
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM),
                ],
                description="Knowledge Base Embeddings",
            ),
        )
        collection.create_index("embedding", index_params={"metric_type": "L2"})
        return collection

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1).numpy().tolist()
        return embeddings

    def store_embedding(self, embedding, metadata):
        entities = [
            {"name": "embedding", "values": embedding, "type": DataType.FLOAT_VECTOR},
        ]
        self.collection.insert(entities)
        self.collection.load()
