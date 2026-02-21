import sys
import json
sys.path.append('./webapp')
from model_utils import handler

filepath = './synthetic_logs_10k.csv'
res = handler.analyze(filepath)
print("Keys:", res.keys())
if "autoencoder" in res:
    clusters = res['autoencoder'].get('clusters', [])
    print("AE Clusters:", len(clusters))
    if clusters:
        print("First AE Cluster:", json.dumps(clusters[0]['analysis'], indent=2))
        print("First AE Cluster logs:", len(clusters[0]['logs']))
if "isolation_forest" in res:
    clusters = res['isolation_forest'].get('clusters', [])
    print("Iso Clusters:", len(clusters))
