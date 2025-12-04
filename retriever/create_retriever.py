
from retriever import Retriever
from retriever.gemini_baseline_retriever import GeminiBaselineRetriever

def create_retriever(retriever_type: str) -> Retriever:
    """Factory function to create retriever instances"""
    retrievers = {
        'gemini_baseline_retriever': GeminiBaselineRetriever
    }
    
    if retriever_type not in retrievers:
        raise ValueError(f"Unknown retriever type: {retriever_type}. Available: {list(retrievers.keys())}")
    
    return retrievers[retriever_type]()