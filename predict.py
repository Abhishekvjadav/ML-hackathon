import json, torch, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv
import torch.nn.functional as F

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
with open('prakriti_clean.json', 'r') as f:
    prakriti = pd.DataFrame(json.load(f))
with open('ayurgenixai_clean.json', 'r') as f:
    ayur = pd.DataFrame(json.load(f))

# ─────────────────────────────────────────
# 2. REBUILD ENCODERS
# ─────────────────────────────────────────
feature_cols = [c for c in prakriti.columns if c != 'Dosha']
encoders = {}
df = prakriti.copy()
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df[feature_cols].values.astype(np.float32)
y = df['Dosha'].values
dosha_names = encoders['Dosha'].classes_

# ─────────────────────────────────────────
# 3. REBUILD GRAPH
# ─────────────────────────────────────────
edge_src, edge_dst = [], []
for i in range(len(y)):
    for j in range(i+1, len(y)):
        if y[i] == y[j]:
            edge_src += [i, j]
            edge_dst += [j, i]

max_edges = 50000
if len(edge_src) > max_edges:
    idx = np.random.choice(len(edge_src), max_edges, replace=False)
    edge_src = [edge_src[i] for i in idx]
    edge_dst = [edge_dst[i] for i in idx]

edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
x_tensor   = torch.tensor(X, dtype=torch.float)

# ─────────────────────────────────────────
# 4. LOAD MODEL
# ─────────────────────────────────────────
class DoshaGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=0.3)
        self.gat2 = GATConv(hidden*heads, out_channels, heads=1, concat=False, dropout=0.3)
    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = DoshaGAT(in_channels=X.shape[1], hidden=32, out_channels=len(dosha_names))
model.load_state_dict(torch.load('dosha_gat_model.pth'))
model.eval()
print("✅ Model loaded!")

# ─────────────────────────────────────────
# 5. REMEDY LOOKUP (fixed format)
# ─────────────────────────────────────────
def get_remedy(dosha_pred):
    # Convert "vata+pitta" → ["vata", "pitta"]
    parts = [p.strip() for p in dosha_pred.replace('+', ',').split(',')]

    # Try exact multi-dosha match first
    for _, row in ayur.iterrows():
        doshas_in_row = str(row['Doshas']).lower()
        if all(p in doshas_in_row for p in parts):
            return row

    # Fallback: match first dosha only
    for _, row in ayur.iterrows():
        if parts[0] in str(row['Doshas']).lower():
            return row

    return None

# ─────────────────────────────────────────
# 6. PREDICT FUNCTION
# ─────────────────────────────────────────
def predict_patient(patient_idx):
    with torch.no_grad():
        out  = model(x_tensor, edge_index)
        pred = out[patient_idx].argmax().item()
        prob = torch.exp(out[patient_idx])
        dosha = dosha_names[pred]

    print(f"\n{'='*50}")
    print(f"🧬 PATIENT {patient_idx} ANALYSIS")
    print(f"{'='*50}")
    print(f"📊 Predicted Dosha : {dosha.upper()}")
    print(f"📈 Confidence      : {prob[pred]*100:.1f}%")
    print(f"\n📋 Dosha Probabilities:")
    for i, name in enumerate(dosha_names):
        bar = '█' * int(prob[i]*20)
        print(f"   {name:<12} {prob[i]*100:5.1f}% {bar}")

    remedy = get_remedy(dosha)
    if remedy is not None:
        print(f"\n{'='*50}")
        print(f"🌿 AYURVEDIC REMEDY")
        print(f"{'='*50}")
        print(f"🌱 Herbs     : {remedy['Ayurvedic Herbs']}")
        print(f"💊 Formulation: {remedy['Formulation']}")
        print(f"🍽️  Diet      : {remedy['Diet and Lifestyle Recommendations']}")
        print(f"🧘 Yoga      : {remedy['Yoga & Physical Therapy']}")
        print(f"🛡️  Prevention: {remedy['Prevention']}")
    else:
        print("⚠️  No remedy found")

    return dosha

# ─────────────────────────────────────────
# 7. TEST 5 PATIENTS
# ─────────────────────────────────────────
print("\n🚀 Testing predictions on 5 patients...\n")
for i in [0, 1, 2, 10, 50]:
    predict_patient(i)

print("\n✅ Prediction engine ready!")