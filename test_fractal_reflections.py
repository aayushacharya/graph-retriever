import json
import os
from typing import List, Dict, Optional, Any, Set, Tuple
from graph.freebase import Freebase
from utils.gemini.helpers import generate_text, load_gemini_client
from collections import defaultdict
import re
from dotenv import load_dotenv
load_dotenv()

class KnowledgeGraph:
    """Interface to Freebase KG following GrailQA setup"""
    
    def __init__(self, ontology_path="ontology/"):
        self.ontology_path = ontology_path
        self.fb_roles = self._load_fb_roles()
        self.fb_types = self._load_fb_types()
        self.reverse_roles = {v: k for k, v in self.fb_roles.items()}
        self.reverse_types = {v: k for k, v in self.fb_types.items()}
        self.graph=Freebase()
        
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
    
    def search_relations(self, query: str, limit: int = 10) -> List[Tuple[str, str]]:
        """Search for relations matching a query"""
        results = []
        query_lower = query.lower()
        for mid, name in self.fb_roles.items():
            if query_lower in name.lower():
                results.append((mid, name))
                if len(results) >= limit:
                    break
        return results
    
    def search_types(self, query: str, limit: int = 10) -> List[Tuple[str, str]]:
        """Search for types matching a query"""
        results = []
        query_lower = query.lower()
        for mid, name in self.fb_types.items():
            if query_lower in name.lower():
                results.append((mid, name))
                if len(results) >= limit:
                    break
        return results
    
    def get_neighbors(self, entity: str, hops: int = 1, relations: List[str] = []) -> Dict:
        """Get neighbors of an entity within specified hops"""
        # This would interface with actual Freebase via Virtuoso
        # For now, returning structure that would come from SPARQL
        subgraph = {
            'nodes': set([entity]),
            'edges': [],
            'entity': entity
        }
        
        # Simulated neighbor retrieval
        # In real implementation, this would query Virtuoso SPARQL endpoint
        # Example query: SELECT ?rel ?obj WHERE { <entity> ?rel ?obj }

        neighbors = self.graph.get_neighbors(entity, hops=hops, relations=relations)
        for neighbor in neighbors:
            rel = neighbor.relation
            obj = neighbor.node.node_id
            # Get readable names if available
            rel_name = self.fb_roles.get(rel.replace('ns:', '').replace('http://rdf.freebase.com/ns/', ''), rel)
            subgraph['edges'].append({
                'source': entity,
                'relation': rel,
                'relation_name': rel_name,
                'target': obj
            })
            subgraph['nodes'].add(obj)

        subgraph['nodes'] = list(subgraph['nodes'])
        
        return subgraph
    
    def execute_sparql(self, query: str) -> List[Dict]:
        """Execute SPARQL query on Freebase"""
        # Interface with Virtuoso endpoint
        # In real implementation: requests.post(SPARQL_ENDPOINT, data={'query': query})
        return self.graph.execute_query(query)

class StructuralProbeGenerator:
    """SPG: Generates structural probes for graph traversal"""
    
    def __init__(self, kg: KnowledgeGraph, model_name='gemini-2.5-pro'):
        self.model = model_name
        self.llm_client=load_gemini_client()
        self.kg = kg
    
    def generate_probes(self, question: str, num_probes: int = 3) -> List[Dict]:
        """Generate structural probes for a question using actual ontology"""
        
        # First, identify key concepts in the question to find relevant relations
        concept_prompt = f"""Extract key concepts from this question that would appear in a knowledge graph:

Question: {question}

List 2-3 key concepts (entities, properties, or types). Be specific and brief.
Examples: "capital city", "country", "person", "occupation"
Generate each concept on a new line. Do not include any explanations.
"""

        try:
            response = generate_text(self.llm_client,concept_prompt,self.model)
            concepts = [line.strip() for line in response.split('\n') if line.strip()]
        except:
            concepts = []
        
        # Search for relevant relations and types in ontology
        print("   Identified Concepts:", concepts)
        relevant_relations = []
        relevant_types = []
        
        for concept in concepts[:3]:
            rels = self.kg.search_relations(concept, limit=5)
            types = self.kg.search_types(concept, limit=5)
            relevant_relations.extend(rels)
            relevant_types.extend(types)
        
        # Format ontology info for LLM
        print(relevant_relations)
        print(relevant_types)
        ontology_context = self._format_ontology_context(relevant_relations, relevant_types)
        print(ontology_context)
        
        prompt = f"""You are a graph query planner for Freebase. Generate structural probes using ONLY the provided ontology.

Question: {question}

Available Freebase Relations:
{ontology_context['relations']}

Available Freebase Types:
{ontology_context['types']}

IMPORTANT: You MUST use relation MIDs from the list above (e.g., "location.country.capital").
Do NOT invent relations like "hasCapital" or "capitalCity" - only use exact MIDs from the ontology.

Generate {num_probes} different structural probes. Each probe should specify:
1. START: The starting entity or entity type (use Freebase format like "m.02_286" or type MID)
2. RELATIONS: List of relation MIDs from the ontology above
3. PATTERN: Traversal strategy (BFS, DFS, or specific motif)
4. GOAL: Semantic constraint to satisfy
5. HOPS: Number of hops (1-2)

Output as JSON list:
[
  {{
    "probe_id": 1,
    "start": "entity_mid_or_type",
    "relations": ["relation.mid.from.ontology"],
    "pattern": "BFS",
    "goal": "description of what to find",
    "hops": 1
  }}
]

Focus on generating diverse exploration strategies using ONLY the provided relations."""

        response = generate_text(self.llm_client,prompt,self.model)
        
        try:
            # Extract JSON from response
            text = response
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                probes = json.loads(json_match.group())
                return self._validate_probes(probes)
        except:
            pass
        
        # Fallback: create default probes using ontology
        return self._create_fallback_probes(relevant_relations, relevant_types)
    
    def _format_ontology_context(self, relations: List[Tuple[str, str]], 
                                  types: List[Tuple[str, str]]) -> Dict:
        """Format ontology information for LLM"""
        
        rel_lines = []
        for mid, name in relations[:15]:  # Limit to top 15
            rel_lines.append(f"  - {mid}: {name}")
        
        type_lines = []
        for mid, name in types[:10]:  # Limit to top 10
            type_lines.append(f"  - {mid}: {name}")
        
        return {
            'relations': '\n'.join(rel_lines) if rel_lines else "  (No relevant relations found)",
            'types': '\n'.join(type_lines) if type_lines else "  (No relevant types found)"
        }
    
    def _validate_probes(self, probes: List[Dict]) -> List[Dict]:
        """Validate that probes use actual ontology relations"""
        validated = []
        
        for probe in probes:
            relations = probe.get('relations', [])
            valid_relations = []
            
            for rel in relations:
                # Check if relation exists in ontology
                if self.kg.get_relation_mid(rel) or rel in self.kg.fb_roles:
                    valid_relations.append(rel)
            
            if valid_relations or relations == ['all']:
                probe['relations'] = valid_relations if valid_relations else ['all']
                validated.append(probe)
        
        return validated if validated else probes
    
    def _create_fallback_probes(self, relations: List[Tuple[str, str]], 
                                types: List[Tuple[str, str]]) -> List[Dict]:
        """Create fallback probes using ontology"""
        probes = []
        
        if relations:
            # Create BFS probe with top relations
            probes.append({
                "probe_id": 1,
                "start": "unknown",
                "relations": [relations[0][0]] if relations else ["all"],
                "pattern": "BFS",
                "goal": "Find relevant entities",
                "hops": 1
            })
        
        if len(relations) > 1:
            # Create DFS probe with different relations
            probes.append({
                "probe_id": 2,
                "start": "unknown",
                "relations": [relations[1][0]],
                "pattern": "DFS",
                "goal": "Explore deeper connections",
                "hops": 2
            })
        
        # Create broad probe
        probes.append({
            "probe_id": 3,
            "start": "unknown",
            "relations": ["all"],
            "pattern": "BFS",
            "goal": "General exploration",
            "hops": 1
        })
        
        return probes

class ReflectionBasedExecutor:
    """RBE: Executes probes on the knowledge graph"""
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
    
    def execute_probe(self, probe: Dict) -> Dict:
        """Execute a single probe and return subgraph"""
        
        entity = probe.get('start', 'unknown')
        relations = probe.get('relations', [])
        hops = probe.get('hops', 1)
        pattern = probe.get('pattern', 'BFS')
        
        # Execute based on pattern
        if pattern == 'BFS':
            return self._execute_bfs(entity, relations, hops)
        elif pattern == 'DFS':
            return self._execute_dfs(entity, relations, hops)
        else:
            return self._execute_motif(entity, relations, probe.get('goal', ''))
    
    def _execute_bfs(self, entity: str, relations: List[str], hops: int) -> Dict:
        """BFS traversal"""
        subgraph = {
            'nodes': set([entity]),
            'edges': [],
            'pattern': 'BFS'
        }
        
        current_level = [entity]
        visited = set([entity])
        
        for hop in range(hops):
            next_level = []
            for node in current_level:
                neighbors = self.kg.get_neighbors(node, hops=1, relations=relations)
                
                for edge in neighbors.get('edges', []):
                    subgraph['edges'].append(edge)
                    target = edge.get('target')
                    if target and target not in visited:
                        subgraph['nodes'].add(target)
                        next_level.append(target)
                        visited.add(target)
            
            current_level = next_level
        
        subgraph['nodes'] = list(subgraph['nodes'])
        return subgraph
    
    def _execute_dfs(self, entity: str, relations: List[str], hops: int) -> Dict:
        """DFS traversal"""
        subgraph = {
            'nodes': set([entity]),
            'edges': [],
            'pattern': 'DFS'
        }
        
        def dfs_helper(node, depth):
            if depth >= hops:
                return
            
            neighbors = self.kg.get_neighbors(node, hops=1, relations=relations)
            for edge in neighbors.get('edges', []):
                subgraph['edges'].append(edge)
                target = edge.get('target')
                if target and target not in subgraph['nodes']:
                    subgraph['nodes'].add(target)
                    dfs_helper(target, depth + 1)
        
        dfs_helper(entity, 0)
        subgraph['nodes'] = list(subgraph['nodes'])
        return subgraph
    
    def _execute_motif(self, entity: str, relations: List[str], goal: str) -> Dict:
        """Motif-based traversal"""
        # Simplified motif matching
        return self._execute_bfs(entity, relations, 2)

class SelfConsistentSubgraphReconstructor:
    """SSR: Recursively grows and refines subgraph"""
    
    def __init__(self, model_name='gemini-2.5-pro'):
        self.llm_client=load_gemini_client()
        self.model = model_name
        self.max_iterations = 3
    
    def reconstruct(self, question: str, initial_subgraphs: List[Dict]) -> Dict:
        """Recursively grow subgraph with self-consistency"""
        
        merged_subgraph = self._merge_subgraphs(initial_subgraphs)
        
        for iteration in range(self.max_iterations):
            # Identify gaps
            gaps = self._identify_gaps(question, merged_subgraph)
            
            if not gaps:
                break
            
            # Propose new probes
            new_probes = self._propose_probes(question, merged_subgraph, gaps)
            
            # These would be executed by RBE in full system
            # For now, mark iteration complete
            merged_subgraph['iteration'] = iteration + 1
            merged_subgraph['identified_gaps'] = gaps
            merged_subgraph['proposed_probes'] = new_probes
        
        return merged_subgraph
    
    def _merge_subgraphs(self, subgraphs: List[Dict]) -> Dict:
        """Merge multiple subgraphs"""
        merged = {
            'nodes': set(),
            'edges': [],
            'sources': []
        }
        
        for sg in subgraphs:
            merged['nodes'].update(sg.get('nodes', []))
            merged['edges'].extend(sg.get('edges', []))
            merged['sources'].append(sg.get('pattern', 'unknown'))
        
        merged['nodes'] = list(merged['nodes'])
        return merged
    
    def _identify_gaps(self, question: str, subgraph: Dict) -> List[str]:
        """Use LLM to identify gaps in subgraph"""
        
        prompt = f"""You are analyzing a knowledge subgraph to answer a question.

Question: {question}

Current Subgraph:
- Nodes: {len(subgraph.get('nodes', []))} entities
- Edges: {len(subgraph.get('edges', []))} relations

Identify structural gaps or missing information needed to answer the question.
List 3 specific gaps:

1.
2.
3.

Be concise and specific."""

        try:
            response = generate_text(self.llm_client,prompt,self.model)
            gaps = [line.strip() for line in response.split('\n') if line.strip() and any(c.isdigit() for c in line[:3])]
            return gaps[:3]
        except:
            return []
    
    def _propose_probes(self, question: str, subgraph: Dict, gaps: List[str]) -> List[Dict]:
        """Propose new probes to fill gaps"""
        
        prompt = f"""Given these gaps in knowledge:

{chr(10).join(gaps)}

Propose 2 new structural probes to fill these gaps. Output as JSON:
[
  {{"probe_id": "gap1", "start": "entity", "relations": ["rel"], "pattern": "BFS", "hops": 1}},
  {{"probe_id": "gap2", "start": "entity", "relations": ["rel"], "pattern": "DFS", "hops": 2}}
]
"""

        try:
            response = generate_text(self.llm_client,prompt,self.model)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return []

class FractalConsistencyFilter:
    """FCF: Filters subgraph using multi-strategy consistency"""
    
    def __init__(self):
        self.strategies = ['BFS', 'DFS', 'Motif']
    
    def filter(self, subgraphs: List[Dict]) -> Dict:
        """Apply fractal consistency filtering"""
        
        # Group by strategy
        strategy_groups = defaultdict(list)
        for sg in subgraphs:
            pattern = sg.get('pattern', 'unknown')
            strategy_groups[pattern].append(sg)
        
        # Find intersection of nodes
        all_nodes = [set(sg.get('nodes', [])) for sg in subgraphs]
        if all_nodes:
            consistent_nodes = set.intersection(*all_nodes) if len(all_nodes) > 1 else all_nodes[0]
        else:
            consistent_nodes = set()
        
        # Find overlapping edges
        edge_counts = defaultdict(int)
        for sg in subgraphs:
            for edge in sg.get('edges', []):
                edge_key = (edge.get('source'), edge.get('relation'), edge.get('target'))
                edge_counts[edge_key] += 1
        
        # Keep edges that appear in multiple strategies
        threshold = max(1, len(subgraphs) // 2)
        consistent_edges = [
            {'source': s, 'relation': r, 'target': t}
            for (s, r, t), count in edge_counts.items()
            if count >= threshold
        ]
        
        return {
            'nodes': list(consistent_nodes),
            'edges': consistent_edges,
            'confidence': len(consistent_edges) / max(1, len(edge_counts)),
            'num_strategies': len(strategy_groups)
        }

class FractalReflections:
    """Main system orchestrator"""
    
    def __init__(self, ontology_path="ontology/", model_name='gemini-2.5-pro'):
        self.kg = KnowledgeGraph(ontology_path)
        self.spg = StructuralProbeGenerator(self.kg)
        self.rbe = ReflectionBasedExecutor(self.kg)
        self.ssr = SelfConsistentSubgraphReconstructor()
        self.fcf = FractalConsistencyFilter()
        self.model = model_name
        self.llm_client=load_gemini_client()
    
    def answer_question(self, question: str) -> Dict:
        """Main pipeline: question -> answer"""
        
        print(f"Processing question: {question}\n")
        
        # 1. Generate structural probes
        print("1. Generating structural probes...")
        probes = self.spg.generate_probes(question, num_probes=3)
        print(f"   Generated {len(probes)} probes\n")
        print(f"   Probes: {json.dumps(probes, indent=2)}\n")
        
        # 2. Execute probes to get subgraphs
        print("2. Executing probes...")
        subgraphs = []
        for probe in probes:
            sg = self.rbe.execute_probe(probe)
            subgraphs.append(sg)
        print(f"   Retrieved {len(subgraphs)} subgraphs\n")
        print(f"   Subgraphs: {[{'nodes': len(sg['nodes']), 'edges': len(sg['edges']), 'pattern': sg.get('pattern', '')} for sg in subgraphs]}\n")
        
        # 3. Reconstruct with self-consistency
        print("3. Reconstructing subgraph...")
        reconstructed = self.ssr.reconstruct(question, subgraphs)
        print(f"   Completed {reconstructed.get('iteration', 0)} iterations\n")
        print(f"   Identified Gaps: {reconstructed.get('identified_gaps', [])}\n")
        print(f"   Proposed Probes: {json.dumps(reconstructed.get('proposed_probes', []), indent=2)}\n")
        
        # 4. Apply fractal consistency filter
        print("4. Applying consistency filter...")
        consistent_subgraph = self.fcf.filter(subgraphs)
        print(f"   Confidence: {consistent_subgraph.get('confidence', 0):.2f}\n")
        
        # 5. Generate final answer
        print("5. Generating final answer...")
        answer = self._generate_answer(question, consistent_subgraph)
        
        return {
            'question': question,
            'probes': probes,
            'subgraph': consistent_subgraph,
            'answer': answer
        }
    
    def _generate_answer(self, question: str, subgraph: Dict) -> str:
        """Generate final answer using only subgraph information"""
        
        prompt = f"""You must answer the question using ONLY the information in the provided subgraph.

Question: {question}

Subgraph Information:
- Nodes: {subgraph.get('nodes', [])}
- Edges: {json.dumps(subgraph.get('edges', []), indent=2)}
- Confidence: {subgraph.get('confidence', 0):.2f}

Provide a clear, explainable answer based strictly on the subgraph.
If the subgraph doesn't contain enough information, say so.

Answer:"""

        try:
            response = generate_text(self.llm_client,prompt,self.model)
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Initialize system
    system = FractalReflections(ontology_path="ontology/")
    
    # Example questions
    questions = [
        "what is the role of opera designer gig who designed the telephone / the medium?",
        "What is the capital of France?",
        "Who directed The Matrix?",
        "What chemical elements did Marie Curie discover?"
    ]
    
    for question in questions[:1]:  # Process first question
        result = system.answer_question(question)
        
        print("\n" + "="*80)
        print("FINAL RESULT")
        print("="*80)
        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSubgraph Summary:")
        print(f"  - Nodes: {len(result['subgraph']['nodes'])}")
        print(f"  - Edges: {len(result['subgraph']['edges'])}")
        print(f"  - Confidence: {result['subgraph']['confidence']:.2f}")
        print("="*80 + "\n")