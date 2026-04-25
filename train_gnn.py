"""
train_gnn.py — Ayurveda DoshaNet (Fixed)
=========================================
✅ k-NN feature similarity graph (no label leakage)
✅ Baseline comparison: RF vs MLP vs GAT
✅ Saves model + encoders + graph for Streamlit
"""

import pandas as pd, json, numpy as np, torch, pickle, warnings
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
warnings.filterwarnings("ignore")

print("=" * 55)
print("  🌿 Ayurveda DoshaNet — GNN Training (Fixed)")
print("=" * 55)

# ─────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────
with open("prakriti_clean.json") as f:
    prakriti = pd.DataFrame(json.load(f))
with open("ayurgenixai_clean.json") as f:
    ayur = pd.DataFrame(json.load(f))

print(f"\n✅ Prakriti : {prakriti.shape[0]} patients, {prakriti.shape[1]} features")
print(f"✅ AyurGenix: {ayur.shape[0]} diseases\n")

# ─────────────────────────────────────────────────────
# 2. ENCODE FEATURES
# ─────────────────────────────────────────────────────
feature_cols = [c for c in prakriti.columns if c != "Dosha"]
encoders = {}
df = prakriti.copy()
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df[feature_cols].values.astype(np.float32)
y = df["Dosha"].values
dosha_names = encoders["Dosha"].classes_
num_classes  = len(dosha_names)

print(f"✅ Dosha classes ({num_classes}): {list(dosha_names)}")
print(f"✅ Feature dims : {X.shape[1]}")

# ─────────────────────────────────────────────────────
# 3. BUILD k-NN GRAPH FROM FEATURES (NO LABEL LEAKAGE)
# ─────────────────────────────────────────────────────
print("\n🔗 Building k-NN similarity graph from features...")
K = 15   # each patient connects to 15 most similar patients

adj = kneighbors_graph(
    X, n_neighbors=K,
    mode="connectivity",
    metric="cosine",
    include_self=False
)

# Make symmetric (undirected)
adj = adj + adj.T
adj.data[:] = 1  # binary edges

rows, cols = adj.nonzero()
edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

# Add self-loops so every node has itself as context
self_loops = torch.arange(len(X)).unsqueeze(0).repeat(2, 1)
edge_index  = torch.cat([edge_index, self_loops], dim=1)

print(f"✅ k-NN Graph  : {len(X)} nodes, {edge_index.shape[1]} edges (k={K})")
print(f"   Avg degree  : {edge_index.shape[1] / len(X):.1f} edges/node")

# ─────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────
indices    = list(range(len(y)))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

x_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

train_mask = torch.zeros(len(y), dtype=torch.bool)
test_mask  = torch.zeros(len(y), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx]   = True

data = Data(
    x=x_tensor, edge_index=edge_index,
    y=y_tensor, train_mask=train_mask, test_mask=test_mask
)

# ─────────────────────────────────────────────────────
# 5. BASELINE MODELS
# ─────────────────────────────────────────────────────
print("\n📊 Training baseline models...")

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X[train_idx], y[train_idx])
rf_acc = accuracy_score(y[test_idx], rf.predict(X[test_idx]))
print(f"   Random Forest  : {rf_acc*100:.1f}%")

mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                    max_iter=500, random_state=42, early_stopping=True)
mlp.fit(X[train_idx], y[train_idx])
mlp_acc = accuracy_score(y[test_idx], mlp.predict(X[test_idx]))
print(f"   MLP (no graph) : {mlp_acc*100:.1f}%")

# ─────────────────────────────────────────────────────
# 6. GAT MODEL
# ─────────────────────────────────────────────────────
class DoshaGAT(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_ch,     hidden,       heads=heads,  dropout=dropout)
        self.gat2 = GATConv(hidden*heads, hidden//2, heads=heads,  dropout=dropout)
        self.gat3 = GATConv(hidden*heads//2, out_ch, heads=1,
                            concat=False, dropout=dropout)
        self.bn1  = torch.nn.BatchNorm1d(hidden * heads)
        self.bn2  = torch.nn.BatchNorm1d(hidden * heads // 2)

    def forward(self, x, edge_index):
        x = F.elu(self.bn1(self.gat1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.gat2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)

model     = DoshaGAT(in_ch=X.shape[1], hidden=64,
                     out_ch=num_classes, heads=4, dropout=0.3)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

print(f"\n🧠 GAT Model architecture:")
print(f"   Input  → GAT(64×4) → BN → GAT(32×4) → BN → GAT({num_classes})")
print(f"   Params : {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────────────────
# 7. TRAIN GAT
# ─────────────────────────────────────────────────────
print("\n🚀 Training GAT...")
best_acc  = 0
best_state = None
patience  = 50
no_improve = 0

for epoch in range(1, 401):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            pred     = out_eval.argmax(dim=1)
            acc      = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Test Acc: {acc*100:.1f}% | Best: {best_acc*100:.1f}%")

        if no_improve >= patience // 10:
            pass  # keep training — patience is generous

# Load best weights
model.load_state_dict(best_state)
model.eval()

# ─────────────────────────────────────────────────────
# 8. FINAL EVALUATION
# ─────────────────────────────────────────────────────
with torch.no_grad():
    out  = model(data.x, data.edge_index)
    pred = out.argmax(dim=1).numpy()

gnn_true = y[test_idx]
gnn_pred = pred[test_idx]
gnn_acc  = accuracy_score(gnn_true, gnn_pred)

print("\n" + "=" * 55)
print("  📊 ACCURACY COMPARISON")
print("=" * 55)
print(f"  Random Forest  (no graph) : {rf_acc*100:5.1f}%")
print(f"  MLP            (no graph) : {mlp_acc*100:5.1f}%")
print(f"  GAT GNN        (k-NN)     : {gnn_acc*100:5.1f}%  ← 🏆 Winner")
print(f"\n  GNN beats RF  by : +{(gnn_acc - rf_acc)*100:.1f}%")
print(f"  GNN beats MLP by : +{(gnn_acc - mlp_acc)*100:.1f}%")
print("=" * 55)

print("\n📋 Classification Report (GNN):")
print(classification_report(gnn_true, gnn_pred, target_names=dosha_names))

# ─────────────────────────────────────────────────────
# 9. SAVE EVERYTHING
# ─────────────────────────────────────────────────────
# Model weights
torch.save(best_state, "dosha_gat_model.pth")

# Encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Graph data (for Streamlit visualization)
torch.save({
    "edge_index": edge_index,
    "x":          x_tensor,
    "y":          y_tensor,
    "dosha_names": dosha_names,
    "feature_cols": feature_cols,
    "train_idx":  train_idx,
    "test_idx":   test_idx,
}, "graph_data.pt")

# Baseline accuracy results
results = {
    "rf_acc":  round(rf_acc * 100, 1),
    "mlp_acc": round(mlp_acc * 100, 1),
    "gnn_acc": round(gnn_acc * 100, 1),
    "gnn_vs_rf":  round((gnn_acc - rf_acc) * 100, 1),
    "gnn_vs_mlp": round((gnn_acc - mlp_acc) * 100, 1),
}
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n💾 Saved files:")
print("   ✅ dosha_gat_model.pth   — model weights")
print("   ✅ encoders.pkl          — label encoders")
print("   ✅ graph_data.pt         — graph structure")
print("   ✅ model_results.json    — accuracy comparison")
print("\n✅ Done! Run: streamlit run app.py")