"""
GrailQA Graph Implementation - Knowledge Graph for GrailQA dataset
Built on Freebase knowledge graph
"""

from typing import List, Any, Dict, Optional, cast
import numpy as np
from collections import defaultdict
from graph import Graph, Node, Neighbor
from SPARQLWrapper import SPARQLWrapper, JSON
import os


class Freebase(Graph):
    """Freebase-based Graph for GrailQA Dataset
    """
    
    def __init__(self, sparql_endpoint: str = "http://localhost:3001/sparql", ontology_path: str = "ontology/"):
        self.name="Freebase"
        self.ontology_path = ontology_path
        self.fb_roles = self._load_fb_roles()
        self.fb_types = self._load_fb_types()
        self.reverse_roles = {v: k for k, v in self.fb_roles.items()}
        self.reverse_types = {v: k for k, v in self.fb_types.items()}
        self.sparql=SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        # SPARQL prefixes for Freebase
        self.sparql_prefixes = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ns: <http://rdf.freebase.com/ns/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

    def _load_fb_roles(self):
        """Load Freebase relations (MID -> readable name)"""
        roles_file = os.path.join(self.ontology_path, "fb_roles")
        roles = {}
        if os.path.exists(roles_file):
            with open(roles_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        roles[parts[0]] = parts[1]
        return roles
    
    def _load_fb_types(self):
        """Load Freebase types (MID -> readable name)"""
        types_file = os.path.join(self.ontology_path, "fb_types")
        types = {}
        if os.path.exists(types_file):
            with open(types_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        types[parts[0]] = parts[1]
        return types
        
    def build(self, data: Any) -> None:
        """
        Freebase graph is too large to load, so we query it on demand.
        """
        pass

    def search_nodes(self, criteria: Dict[str, Any]) -> List[Any]:
        return []
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by its identifier"""
        
        return None
    
    def get_type_mid(self, type_name: str) -> Optional[str]:
        """Convert human-readable type to Freebase MID"""
        if type_name in self.reverse_types:
            return self.reverse_types[type_name]
        
        # Fuzzy match
        type_lower = type_name.lower()
        for mid, name in self.fb_types.items():
            if type_lower in name.lower():
                return mid
        
        return None
    
    def get_relation_mid(self, relation_name: str) -> Optional[str]:
        """Convert human-readable relation to Freebase MID"""
        # Direct lookup
        if relation_name in self.reverse_roles:
            return self.reverse_roles[relation_name]
        
        # Fuzzy match
        relation_lower = relation_name.lower()
        for mid, name in self.fb_roles.items():
            if relation_lower in name.lower():
                return mid
        
        return None

    def get_neighbors(self, node_id: str, hops: int=1, relations: Optional[List[str]] = None) -> List[Neighbor]:
        """Get neighbors of an entity within specified hops"""
        entity=node_id
         # Ensure entity has proper ns: prefix
        if not entity.startswith('ns:') and not entity.startswith('http'):
            entity = f'ns:{entity}'
        
        # Convert relation names to MIDs if needed
        relation_mids = []
        if relations and relations != ['all']:
            for rel in relations:
                if rel.startswith('ns:') or rel.startswith('http'):
                    relation_mids.append(rel)
                else:
                    mid = self.get_relation_mid(rel)
                    if mid:
                        relation_mids.append(f'ns:{mid}')
        
        # Build SPARQL query for n-hop neighbors
        if hops == 1:
            # Single hop query
            if relation_mids:
                # Filter by specific relations
                relation_values = ' '.join([f'<{r}>' if r.startswith('http') else r for r in relation_mids])
                query = f"""
{self.sparql_prefixes}
SELECT DISTINCT ?rel ?obj WHERE {{
    {entity} ?rel ?obj .
    FILTER (?rel IN ({relation_values}))
}}
LIMIT 1000
"""
            else:
                # All relations
                query = f"""
{self.sparql_prefixes}
SELECT DISTINCT ?rel ?obj WHERE {{
    {entity} ?rel ?obj .
}}
LIMIT 1000
"""
        else:
            # Multi-hop query using property paths
            if relation_mids:
                # For specific relations, build alternation path
                rel_path = '|'.join([f'{r}' for r in relation_mids])
                query = f"""
{self.sparql_prefixes}
SELECT DISTINCT ?rel ?obj WHERE {{
    {entity} ({rel_path}){{1,{hops}}} ?obj .
    OPTIONAL {{ {entity} ?rel ?obj }}
}}
LIMIT 2000
"""
            else:
                # All relations up to n hops
                query = f"""
{self.sparql_prefixes}
SELECT DISTINCT ?rel ?obj WHERE {{
    {entity} ?rel{{1,{hops}}} ?obj .
}}
LIMIT 2000
"""
        
        # Execute query
        # print(query)
        results = self.execute_query(query)
        
        output: List[Neighbor]=[]

        for result in results:
            rel = result.get('rel', {}).get('value', '')
            obj = result.get('obj', {}).get('value', '')
            node:Node =Node(node_id=obj,node_type=None, attributes={})
            
            neighbor:Neighbor=Neighbor(node=node, relation=rel)
            output.append(neighbor)

        return output


    def __str__(self) -> str:
        """Return the name of the graph"""
        return self.name

    def execute_query(self, query: str) -> List[Any]:
        """
        query: str - SPARQL query string
        """
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
            results = cast(Dict[str, Any], results)
            return results["results"]["bindings"]
        except Exception as e:
            print(f"Error executing SPARQL query: {e}")
            return []
    
    