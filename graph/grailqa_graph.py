"""
GrailQA Graph Implementation - Knowledge Graph for GrailQA dataset
Built on Freebase knowledge graph
"""

from typing import List, Any, Dict, Set, Tuple, Optional
import numpy as np
from collections import defaultdict
from graph import Graph


class GrailQAGraph(Graph):
    """Knowledge graph implementation for GrailQA dataset
    
    GrailQA is built on Freebase with:
    - Entities with MIDs (Machine IDs)
    - Relations with domain/type structure
    - Support for CVT (Compound Value Type) nodes
    - Schema-level information (domains, types, properties)
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.nodes = {}  # node_id -> node_info
        self.node_features = {}  # node_id -> embedding
        self.edges = {}  # source -> [(target, relation, weight)]
        self.reverse_edges = {}  # target -> [(source, relation, weight)]
        
        # Freebase specific mappings
        self.mid_to_id = {}  # Freebase MID -> node_id
        self.id_to_mid = {}  # node_id -> Freebase MID
        self.mid_to_name = {}  # MID -> human-readable name
        self.name_to_mid = {}  # name -> MID
        
        # Schema information
        self.relation_types = set()  # All relation types
        self.entity_types = {}  # entity_id -> [types]
        self.domains = set()  # Freebase domains
        
        # CVT nodes (Compound Value Type - intermediate nodes)
        self.cvt_nodes = set()
        
        # Relation hierarchy and properties
        self.relation_domain = {}  # relation -> domain
        self.relation_range = {}  # relation -> range type
        
    def build(self, data: Any) -> None:
        """Build knowledge graph from GrailQA data
        
        Expected format:
        {
            'entities': {
                'mid': {
                    'name': str,
                    'types': [type_ids],
                    'embedding': np.ndarray
                }
            },
            'triples': [(head_mid, relation, tail_mid), ...],
            'schema': {
                'relations': {relation_id: {'domain': str, 'range': str}},
                'domains': [domain_ids]
            }
        }
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        entities = data.get('entities', {})
        triples = data.get('triples', [])
        schema = data.get('schema', {})
        
        # Load schema information
        self._load_schema(schema)
        
        # Add entities
        for mid, entity_info in entities.items():
            node_id = self._get_or_create_node_id(mid)
            
            # Store entity metadata
            name = entity_info.get('name', mid)
            self.mid_to_name[mid] = name
            self.name_to_mid[name] = mid
            
            # Store types
            types = entity_info.get('types', [])
            self.entity_types[node_id] = types
            
            # Add node with embedding
            embedding = entity_info.get('embedding')
            if embedding is None:
                embedding = self._generate_embedding(mid, types)
            
            self.add_node(node_id, embedding)
            
            # Mark CVT nodes
            if entity_info.get('is_cvt', False):
                self.cvt_nodes.add(node_id)
        
        # Add triples
        for head_mid, relation, tail_mid in triples:
            head_id = self._get_or_create_node_id(head_mid)
            tail_id = self._get_or_create_node_id(tail_mid)
            
            # If nodes don't exist, create them
            if head_id not in self.nodes:
                self.add_node(head_id, self._generate_embedding(head_mid, []))
            if tail_id not in self.nodes:
                self.add_node(tail_id, self._generate_embedding(tail_mid, []))
            
            self.add_edge(head_id, tail_id, relation=relation)
    
    def _load_schema(self, schema: Dict) -> None:
        """Load Freebase schema information"""
        relations = schema.get('relations', {})
        for relation_id, rel_info in relations.items():
            self.relation_types.add(relation_id)
            self.relation_domain[relation_id] = rel_info.get('domain', '')
            self.relation_range[relation_id] = rel_info.get('range', '')
        
        self.domains = set(schema.get('domains', []))
    
    def _get_or_create_node_id(self, mid: str) -> str:
        """Get existing node ID or create new one for a MID"""
        if mid not in self.mid_to_id:
            node_id = f"entity_{len(self.mid_to_id)}"
            self.mid_to_id[mid] = node_id
            self.id_to_mid[node_id] = mid
        return self.mid_to_id[mid]
    
    def _generate_embedding(self, mid: str, types: List[str]) -> np.ndarray:
        """Generate embedding for an entity"""
        # Use hash-based generation for consistency
        hash_val = hash(mid)
        np.random.seed(hash_val % (2**32))
        
        embedding = np.random.randn(self.embedding_dim)
        
        # Add type information to embedding
        for type_id in types:
            type_hash = hash(type_id)
            np.random.seed(type_hash % (2**32))
            type_vec = np.random.randn(self.embedding_dim) * 0.1
            embedding += type_vec
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def add_node(self, node_id: str, features: np.ndarray) -> None:
        """Add an entity node with its embedding"""
        self.nodes[node_id] = True
        
        # Ensure features are the right dimension
        if len(features) != self.embedding_dim:
            if len(features) < self.embedding_dim:
                padded = np.zeros(self.embedding_dim, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            else:
                features = features[:self.embedding_dim]
        
        self.node_features[node_id] = np.array(features, dtype=np.float32)
        
        if node_id not in self.edges:
            self.edges[node_id] = []
        if node_id not in self.reverse_edges:
            self.reverse_edges[node_id] = []
    
    def add_edge(self, source: str, target: str, weight: float = 1.0, 
                 relation: str = "related_to") -> None:
        """Add a typed edge (relation) between entities"""
        if source not in self.edges:
            self.edges[source] = []
        if target not in self.reverse_edges:
            self.reverse_edges[target] = []
        
        self.edges[source].append((target, relation, weight))
        self.reverse_edges[target].append((source, relation, weight))
        self.relation_types.add(relation)
    
    def get_neighbors(self, node_id: str, relation: str = None, 
                     reverse: bool = False) -> List[str]:
        """Get neighbors of a node, optionally filtered by relation
        
        Args:
            node_id: Source node
            relation: Relation type to filter by
            reverse: If True, get incoming edges instead of outgoing
        """
        edge_dict = self.reverse_edges if reverse else self.edges
        
        if node_id not in edge_dict:
            return []
        
        if relation is None:
            return [target for target, _, _ in edge_dict[node_id]]
        else:
            return [target for target, rel, _ in edge_dict[node_id] if rel == relation]
    
    def get_node_features(self, node_id: str) -> np.ndarray:
        """Get embedding features of an entity node"""
        return self.node_features.get(node_id, np.zeros(self.embedding_dim, dtype=np.float32))
    
    def get_entity_name(self, node_id: str) -> str:
        """Get human-readable name for an entity"""
        mid = self.id_to_mid.get(node_id, node_id)
        return self.mid_to_name.get(mid, mid)
    
    def get_entity_mid(self, node_id: str) -> str:
        """Get Freebase MID for a node"""
        return self.id_to_mid.get(node_id, node_id)
    
    def get_node_id_from_mid(self, mid: str) -> Optional[str]:
        """Get node ID from Freebase MID"""
        return self.mid_to_id.get(mid)
    
    def get_node_id_from_name(self, name: str) -> Optional[str]:
        """Get node ID from entity name"""
        mid = self.name_to_mid.get(name)
        if mid:
            return self.mid_to_id.get(mid)
        return None
    
    def get_entity_types(self, node_id: str) -> List[str]:
        """Get types of an entity"""
        return self.entity_types.get(node_id, [])
    
    def is_cvt_node(self, node_id: str) -> bool:
        """Check if a node is a CVT (Compound Value Type) node"""
        return node_id in self.cvt_nodes
    
    def get_relations(self, source: str, target: str) -> List[str]:
        """Get all relations between two entities"""
        if source not in self.edges:
            return []
        
        relations = []
        for tgt, rel, _ in self.edges[source]:
            if tgt == target:
                relations.append(rel)
        return relations
    
    def get_relation_domain(self, relation: str) -> str:
        """Get the domain of a relation"""
        return self.relation_domain.get(relation, '')
    
    def get_relation_range(self, relation: str) -> str:
        """Get the range type of a relation"""
        return self.relation_range.get(relation, '')
    
    def sparql_query(self, subject: str = None, predicate: str = None, 
                    object: str = None) -> List[Tuple[str, str, str]]:
        """Simple SPARQL-like triple pattern matching
        
        Args:
            subject: Subject node ID or None for wildcard
            predicate: Relation or None for wildcard
            object: Object node ID or None for wildcard
        
        Returns:
            List of matching (subject, predicate, object) triples
        """
        results = []
        
        if subject is not None:
            # Query from specific subject
            if subject in self.edges:
                for obj, rel, _ in self.edges[subject]:
                    if (predicate is None or predicate == rel) and \
                       (object is None or object == obj):
                        results.append((subject, rel, obj))
        else:
            # Query all subjects
            for subj in self.edges:
                for obj, rel, _ in self.edges[subj]:
                    if (predicate is None or predicate == rel) and \
                       (object is None or object == obj):
                        results.append((subj, rel, obj))
        
        return results
    
    def constrained_search(self, start_node: str, constraints: Dict[str, Any],
                          max_depth: int = 3) -> List[str]:
        """Search for entities matching constraints from a start node
        
        Args:
            start_node: Starting entity
            constraints: Dictionary of constraints like:
                {
                    'type': [required_types],
                    'relations': [(relation, direction), ...],
                    'exclude_cvt': bool
                }
            max_depth: Maximum search depth
        
        Returns:
            List of matching node IDs
        """
        results = []
        visited = set()
        queue = [(start_node, 0)]
        
        required_types = set(constraints.get('type', []))
        required_relations = constraints.get('relations', [])
        exclude_cvt = constraints.get('exclude_cvt', True)
        
        while queue:
            current, depth = queue.pop(0)
            
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current)
            
            # Check if current node matches constraints
            matches = True
            
            if exclude_cvt and self.is_cvt_node(current):
                matches = False
            
            if required_types:
                node_types = set(self.get_entity_types(current))
                if not required_types.intersection(node_types):
                    matches = False
            
            if matches and current != start_node:
                results.append(current)
            
            # Expand search
            if depth < max_depth:
                # Follow required relations if specified
                if required_relations:
                    for relation, direction in required_relations:
                        if direction == 'forward':
                            neighbors = self.get_neighbors(current, relation=relation)
                        else:  # backward
                            neighbors = self.get_neighbors(current, relation=relation, reverse=True)
                        
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                queue.append((neighbor, depth + 1))
                else:
                    # Explore all neighbors
                    neighbors = self.get_neighbors(current)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))
        
        return results
    
    def get_subgraph(self, center_nodes: List[str], radius: int = 1) -> Dict[str, Any]:
        """Extract a subgraph around given center nodes"""
        subgraph_nodes = set(center_nodes)
        subgraph_edges = []
        
        for center in center_nodes:
            queue = [(center, 0)]
            visited = {center}
            
            while queue:
                current, dist = queue.pop(0)
                
                if dist < radius and current in self.edges:
                    for neighbor, relation, weight in self.edges[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
                        
                        subgraph_nodes.add(neighbor)
                        subgraph_edges.append({
                            'source': current,
                            'target': neighbor,
                            'relation': relation,
                            'weight': weight
                        })
        
        nodes_dict = {
            node_id: self.get_node_features(node_id)
            for node_id in subgraph_nodes
        }
        
        return {
            'nodes': nodes_dict,
            'edges': subgraph_edges
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        num_edges = sum(len(v) for v in self.edges.values())
        
        return {
            'num_entities': len(self.nodes),
            'num_relations': len(self.relation_types),
            'num_edges': num_edges,
            'num_cvt_nodes': len(self.cvt_nodes),
            'num_domains': len(self.domains),
            'avg_degree': num_edges / len(self.nodes) if self.nodes else 0
        }
    
    def __len__(self) -> int:
        """Return number of entities in the graph"""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"GrailQAGraph(entities={stats['num_entities']}, "
                f"relations={stats['num_relations']}, "
                f"edges={stats['num_edges']})")