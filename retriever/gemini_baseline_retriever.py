from retriever import Retriever
from typing import List, Dict, Any, Tuple
from graph import Graph
import numpy as np
from utils.gemini.helpers import load_gemini_client, generate_text


class GeminiBaselineRetriever(Retriever):
    """Gemini Baseline Retriever Implementation"""

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        self.name="GeminiBaselineRetriever"
        self.client=load_gemini_client()
    
    def retrieve(self, query: str,graph: Graph) -> List[Any]:
        """Retrieve content based on the query using Gemini model
        
        Returns:
            List of retrieved contents
        """
        prompt = f"Retrieve SPARQL query for the following Freebase: {query}"
        sparql_query = generate_text(self.client, prompt)
        results=graph.execute_query(sparql_query)
        return results
        

    def __str__(self) -> str:
        """Return the name of the retriever model"""
        return self.name
    