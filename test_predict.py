import torch, pickle, json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load everything
with open("encoders.pkl","rb") as f: encoders = pickle.load(f)
gd = torch.load("graph_data.pt", map_location="cpu")
with open("model_results.json") as f: results = json.load(f)

print("✅ encoders.pkl loaded")
print("✅ graph_data.pt loaded")
print(f"✅ model_results.json: GNN={results['gnn_acc']}% | RF={results['rf_acc']}% | MLP={results['mlp_acc']}%")
print(f"✅ Graph: {gd['x'].shape[0]} nodes, {gd['edge_index'].shape[1]} edges")
print(f"✅ Dosha classes: {list(gd['dosha_names'])}")
print("\n🎉 All files OK — ready for app.py!")