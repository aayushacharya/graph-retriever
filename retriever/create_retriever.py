
from retriever import Retriever
from retriever.cosine_retriever import CosineRetriever

def create_retriever(retriever_type: str) -> Retriever:
    """Factory function to create retriever instances"""
    retrievers = {
        'cosine': CosineRetriever
    }
    
    if retriever_type not in retrievers:
        raise ValueError(f"Unknown retriever type: {retriever_type}. Available: {list(retrievers.keys())}")
    
    return retrievers[retriever_type]()