"""
GrailQA Graph Implementation - Knowledge Graph for GrailQA dataset
Built on Freebase knowledge graph
"""

from typing import List, Any, Dict, cast
import numpy as np
from collections import defaultdict
from graph import Graph
from SPARQLWrapper import SPARQLWrapper, JSON


class Freebase(Graph):
    """Freebase-based Graph for GrailQA Dataset
    """
    
    def __init__(self, sparql_endpoint: str = "http://localhost:3001/sparql"):
        self.name="Freebase"
        self.sparql=SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        
    def build(self, data: Any) -> None:
        """
        Freebase graph is too large to load, so we query it on demand.
        """
        pass

    def __str__(self) -> str:
        """Return the name of the graph"""
        return self.name

    def execute_query(self, query: str) -> List[Any]:
        """
        query: str - SPARQL query string
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
            results = cast(Dict[str, Any], results)
            output = []
            for result in results["results"]["bindings"]:
                for var in result:
                    entity_id=result[var]['value'].split('/')[-1]
                    output.append(entity_id)
            return output
        except Exception as e:
            print(f"Error executing SPARQL query: {e}")
            return []
    
    