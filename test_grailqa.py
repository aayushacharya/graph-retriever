from SPARQLWrapper import SPARQLWrapper, JSON
import json
from typing import Dict, Any, cast
import os

# Setup Virtuoso connection
sparql = SPARQLWrapper("http://localhost:3001/sparql")
sparql.setReturnFormat(JSON)

# Load a sample question from GrailQA dev set
data_dir=os.path.join("data","grailqa","GrailQA_v1.0")
with open(os.path.join(data_dir,'grailqa_v1.0_dev.json'), 'r') as f:
    data = json.load(f)
    sample = data[1]  # Get second question

print(f"Question: {sample['question']}")
print(f"SPARQL Query:\n{sample['sparql_query']}\n")

# Execute the SPARQL query
sparql.setQuery(sample['sparql_query'])

try:
    results = sparql.query().convert()
    results = cast(Dict[str, Any], results)
    
    # Print results
    print("Results:")
    for result in results["results"]["bindings"]:
        for var in result:
            print(f"{var}: {result[var]['value']}")
        print()
        
except Exception as e:
    print(f"Error: {e}")