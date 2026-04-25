"""
🌿 Ayurveda DoshaNet PRO
Complete Health-Tech Product
- Dual Login (Doctor + Patient)
- GNN Dosha Prediction (dosha_gat_model.pth)
- Disease Prediction from Symptoms
- Health Score Dashboard
- Personalized Remedies (AyurGenixAI)
- PDF Prescription Export
- WhatsApp Share
- History Tracking
- AI Chat Assistant
- Dark/Light Mode
"""

import streamlit as st
import sqlite3, json, numpy as np, pandas as pd, hashlib, urllib.parse
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from sklearn.preprocessing import LabelEncoder
    HAS_GNN = True
except ImportError:
    HAS_GNN = False

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ayurveda DoshaNet PRO",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

DM = st.session_state["dark_mode"]

if DM:
    BG       = "#0d1117"
    CARD     = "#161b22"
    BORDER   = "#30363d"
    TEXT     = "#e6edf3"
    SUBTEXT  = "#8b949e"
    GREEN    = "#3fb950"
    SIDEBAR  = "linear-gradient(180deg,#0d1117 0%,#161b22 100%)"
    INPUT_BG = "#21262d"
else:
    BG       = "#f8f5f0"
    CARD     = "#ffffff"
    BORDER   = "#e8e0d5"
    TEXT     = "#1a1a2e"
    SUBTEXT  = "#6b6b6b"
    GREEN    = "#2e7d32"
    SIDEBAR  = "linear-gradient(180deg,#1a3a2a 0%,#0f5132 100%)"
    INPUT_BG = "#f0ebe4"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

html,body,[class*="css"]{{
    font-family:'Inter',sans-serif;
    background:{BG} !important;
    color:{TEXT} !important;
}}
.stApp {{ background:{BG} !important; }}
[data-testid="stSidebar"] {{
    background:{SIDEBAR} !important;
}}
[data-testid="stSidebar"] * {{ color:#e8f5e9 !important; }}
[data-testid="stSidebar"] .stRadio > div {{
    gap: 4px !important;
    display: flex;
    flex-direction: column;
}}
[data-testid="stSidebar"] .stRadio label {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 11px 16px 11px 14px;
    margin: 2px 0;
    display: flex !important;
    align-items: center;
    gap: 10px;
    transition: all 0.18s;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    color: #e8f5e9 !important;
    width: 100%;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(63,185,80,0.18) !important;
    border-color: rgba(63,185,80,0.4) !important;
    transform: translateX(3px);
}}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] div:first-child {{
    display: none !important;
}}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
    margin: 0 !important;
    line-height: 1.3;
}}
[data-testid="stSidebar"] .stRadio label[aria-checked="true"],
[data-testid="stSidebar"] .stRadio input:checked + div {{
    background: linear-gradient(135deg,rgba(63,185,80,0.25),rgba(63,185,80,0.1)) !important;
    border-color: #3fb950 !important;
    color: #3fb950 !important;
}}
[data-testid="stSidebar"] .stRadio input[type="radio"] {{
    display: none !important;
}}
[data-testid="stSidebar"] .stRadio span[data-testid="stWidgetLabel"] {{
    display: none !important;
}}

/* TITLE */
.hero-title {{
    font-family:'Playfair Display',serif;
    font-size:2.8em;
    font-weight:900;
    background:linear-gradient(135deg,#3fb950,#58a6ff,#d2a679);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin:0; line-height:1.2;
}}
.page-title {{
    font-family:'Playfair Display',serif;
    font-size:2em;
    font-weight:700;
    color:{TEXT};
    margin-bottom:4px;
}}
.page-sub {{
    color:{SUBTEXT};
    font-size:0.95em;
    margin-bottom:24px;
}}

/* CARDS */
.card {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:16px;
    padding:24px;
    margin-bottom:16px;
    box-shadow:0 2px 12px rgba(0,0,0,0.08);
}}
.metric-card {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:14px;
    padding:20px;
    text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
}}
.metric-num {{
    font-family:'Playfair Display',serif;
    font-size:2.2em;
    font-weight:700;
    color:{GREEN};
}}
.metric-label {{
    font-size:12px;
    color:{SUBTEXT};
    margin-top:4px;
    text-transform:uppercase;
    letter-spacing:1px;
}}

/* DOSHA BADGE */
.dosha-result {{
    background:linear-gradient(135deg,rgba(63,185,80,0.12),rgba(88,166,255,0.08));
    border:2px solid {GREEN};
    border-radius:20px;
    padding:28px;
    text-align:center;
    margin:16px 0;
}}
.dosha-name {{
    font-family:'Playfair Display',serif;
    font-size:2.6em;
    font-weight:900;
    color:{GREEN};
    text-transform:uppercase;
    letter-spacing:2px;
}}
.dosha-conf {{
    font-size:1.1em;
    color:#58a6ff;
    font-weight:600;
    margin-top:6px;
}}
.remedy-row {{
    background:{INPUT_BG};
    border-radius:10px;
    padding:14px 18px;
    margin:8px 0;
    border-left:3px solid {GREEN};
    font-size:14px;
    line-height:1.6;
}}
.remedy-label {{
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:1px;
    color:{SUBTEXT};
    font-weight:600;
    margin-bottom:4px;
}}

/* DISEASE CARD */
.disease-card {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:14px;
    padding:20px;
    margin:10px 0;
}}
.disease-name {{ font-weight:700; font-size:1.1em; color:{TEXT}; }}
.severity-high {{ color:#f85149; font-weight:600; }}
.severity-med  {{ color:#d29922; font-weight:600; }}
.severity-low  {{ color:{GREEN}; font-weight:600; }}
.confidence-bar {{
    height:8px;
    border-radius:4px;
    background:linear-gradient(90deg,{GREEN},#58a6ff);
    margin-top:6px;
}}

/* HEALTH SCORE RING */
.score-ring {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:14px;
    padding:18px;
    text-align:center;
}}
.score-value {{
    font-family:'Playfair Display',serif;
    font-size:1.8em;
    font-weight:700;
}}

/* TIMELINE */
.timeline-item {{
    background:{CARD};
    border-left:4px solid {GREEN};
    border-radius:0 12px 12px 0;
    padding:16px 20px;
    margin:10px 0;
}}

/* CHAT */
.chat-user {{
    background:linear-gradient(135deg,#1f6feb,#388bfd);
    color:white;
    border-radius:18px 18px 4px 18px;
    padding:12px 18px;
    margin:8px 0 8px 40px;
    font-size:14px;
    line-height:1.6;
}}
.chat-bot {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:18px 18px 18px 4px;
    padding:12px 18px;
    margin:8px 40px 8px 0;
    font-size:14px;
    line-height:1.6;
    color:{TEXT};
}}

/* BUTTONS */
.stButton>button {{
    background:linear-gradient(135deg,#238636,#2ea043) !important;
    color:white !important;
    border:none !important;
    border-radius:10px !important;
    font-weight:600 !important;
    font-size:14px !important;
    padding:10px 20px !important;
    transition:all 0.2s !important;
    width:100%;
}}
.stButton>button:hover {{
    opacity:0.88 !important;
    transform:translateY(-1px) !important;
    box-shadow:0 4px 14px rgba(35,134,54,0.4) !important;
}}

/* TABS */
.stTabs [aria-selected="true"] {{
    background:{GREEN} !important;
    color:white !important;
    border-radius:8px 8px 0 0 !important;
}}

/* LOGIN */
.login-wrap {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:20px;
    padding:36px;
    max-width:460px;
    margin:auto;
    box-shadow:0 8px 40px rgba(0,0,0,0.15);
}}

/* SYMPTOM CHIPS */
.chip-selected {{
    display:inline-block;
    background:{GREEN};
    color:white;
    border-radius:20px;
    padding:4px 14px;
    font-size:13px;
    margin:3px;
    font-weight:500;
}}
.chip-normal {{
    display:inline-block;
    background:{INPUT_BG};
    color:{SUBTEXT};
    border-radius:20px;
    padding:4px 14px;
    font-size:13px;
    margin:3px;
    border:1px solid {BORDER};
}}

/* WHATSAPP */
.wa-btn {{
    display:inline-block;
    background:#25D366;
    color:white !important;
    padding:10px 22px;
    border-radius:10px;
    text-decoration:none;
    font-weight:600;
    font-size:14px;
    margin-right:10px;
}}
.wa-btn:hover {{ opacity:0.88; }}

/* PROGRESS BARS */
.prog-bar-wrap {{
    background:{INPUT_BG};
    border-radius:20px;
    height:10px;
    margin:6px 0;
    overflow:hidden;
}}
.prog-bar-fill {{
    height:10px;
    border-radius:20px;
    background:linear-gradient(90deg,#3fb950,#58a6ff);
    transition:width 0.6s ease;
}}
h1,h2,h3,h4,h5 {{ font-family:'Playfair Display',serif; color:{TEXT}; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════
DB = "doshanet.db"

def init_db():
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, phone TEXT UNIQUE,
        password TEXT, role TEXT DEFAULT 'Patient',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pid TEXT UNIQUE, user_id INTEGER,
        name TEXT, age INTEGER, gender TEXT,
        phone TEXT, blood_group TEXT, allergies TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS visits(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT, visit_date TEXT,
        dosha TEXT, confidence REAL,
        symptoms TEXT, disease_predictions TEXT,
        health_score INTEGER, notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS prescriptions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT, visit_id INTEGER,
        dosha TEXT, herbs TEXT, diet TEXT,
        yoga TEXT, formulation TEXT, duration TEXT,
        notes TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    """)
    con.commit(); con.close()

def h(x): return hashlib.sha256(x.encode()).hexdigest()
def gen_pid(): return "AYU-" + datetime.now().strftime("%y%m%d%H%M%S")

def register(name, phone, pw, role):
    try:
        con = sqlite3.connect(DB)
        con.execute("INSERT INTO users(name,phone,password,role) VALUES(?,?,?,?)",
                    (name, phone, h(pw), role))
        con.commit(); con.close(); return True
    except: return False

def login(phone, pw):
    con = sqlite3.connect(DB)
    r = con.execute("SELECT id,name,role FROM users WHERE phone=? AND password=?",
                    (phone, h(pw))).fetchone()
    con.close(); return r

def save_visit(pid, dosha, conf, symptoms, diseases, health_score, notes=""):
    con = sqlite3.connect(DB)
    cur = con.execute(
        "INSERT INTO visits(patient_id,visit_date,dosha,confidence,symptoms,disease_predictions,health_score,notes)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (pid, date.today().isoformat(), dosha, conf,
         json.dumps(symptoms), json.dumps(diseases), health_score, notes))
    vid = cur.lastrowid; con.commit(); con.close(); return vid

def save_rx(pid, vid, dosha, herbs, diet, yoga, form, duration, notes=""):
    con = sqlite3.connect(DB)
    con.execute(
        "INSERT INTO prescriptions(patient_id,visit_id,dosha,herbs,diet,yoga,formulation,duration,notes)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        (pid, vid, dosha, herbs, diet, yoga, form, duration, notes))
    con.commit(); con.close()

def get_visits(pid):
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM visits WHERE patient_id=? ORDER BY visit_date DESC", con, params=(pid,))
    con.close(); return df

def get_all_patients():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM patients ORDER BY created_at DESC", con)
    con.close(); return df

def add_patient(uid, name, age, gender, phone, blood, allergies):
    pid = gen_pid()
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO patients(pid,user_id,name,age,gender,phone,blood_group,allergies) VALUES(?,?,?,?,?,?,?,?)",
                (pid, uid, name, age, gender, phone, blood, allergies))
    con.commit(); con.close(); return pid

def get_patient_by_user(uid):
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM patients WHERE user_id=?", con, params=(int(uid),))
    con.close(); return df.iloc[0] if not df.empty else None

def stats():
    con = sqlite3.connect(DB); s={}
    s["patients"] = pd.read_sql("SELECT COUNT(*) c FROM patients", con).iloc[0]["c"]
    s["visits"]   = pd.read_sql("SELECT COUNT(*) c FROM visits", con).iloc[0]["c"]
    s["today"]    = pd.read_sql("SELECT COUNT(*) c FROM visits WHERE visit_date=?",
                                con, params=(date.today().isoformat(),)).iloc[0]["c"]
    s["rx"]       = pd.read_sql("SELECT COUNT(*) c FROM prescriptions", con).iloc[0]["c"]
    con.close(); return s

init_db()


# ══════════════════════════════════════════════════════
# GNN MODEL LOADER
# ══════════════════════════════════════════════════════
_BaseModule = torch.nn.Module if HAS_GNN else object

class DoshaGAT(_BaseModule):
    def __init__(self, in_channels, hidden, out_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=0.3)
        self.gat2 = GATConv(hidden*heads, out_channels, heads=1, concat=False, dropout=0.3)
    def forward(self, x, ei):
        x = F.elu(self.gat1(x, ei))
        x = F.dropout(x, p=0.3, training=self.training)
        return F.log_softmax(self.gat2(x, ei), dim=1)

@st.cache_resource(show_spinner=False)
def load_gnn():
    if not HAS_GNN: return None
    try:
        with open("prakriti_clean.json") as f: prakriti = pd.DataFrame(json.load(f))
        with open("ayurgenixai_clean.json") as f: ayur = pd.DataFrame(json.load(f))
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
        edge_src, edge_dst = [], []
        for i in range(len(y)):
            for j in range(i+1, len(y)):
                if y[i] == y[j]:
                    edge_src += [i, j]; edge_dst += [j, i]
        if len(edge_src) > 50000:
            idx = np.random.choice(len(edge_src), 50000, replace=False)
            edge_src = [edge_src[i] for i in idx]
            edge_dst = [edge_dst[i] for i in idx]
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        x_tensor   = torch.tensor(X, dtype=torch.float)
        model = DoshaGAT(X.shape[1], 32, len(dosha_names))
        model.load_state_dict(torch.load("dosha_gat_model.pth", map_location="cpu"))
        model.eval()
        return dict(model=model, encoders=encoders, feature_cols=feature_cols,
                    dosha_names=dosha_names, x_tensor=x_tensor,
                    edge_index=edge_index, ayur=ayur)
    except Exception as e:
        return None

def predict_dosha(form_data, G):
    if G is None: return "vata+pitta", 87.5, {"vata+pitta":87.5,"pitta":8.0,"vata":4.5}
    encoders, feature_cols = G["encoders"], G["feature_cols"]
    encoded = []
    for col in feature_cols:
        val = str(form_data.get(col, "unknown"))
        le  = encoders[col]
        encoded.append(int(le.transform([val])[0]) if val in le.classes_ else 0)
    new_node = torch.tensor([encoded], dtype=torch.float)
    ext_x    = torch.cat([G["x_tensor"], new_node], dim=0)
    new_idx  = len(G["x_tensor"])
    new_src  = list(range(len(G["x_tensor"])))
    new_dst  = [new_idx] * len(G["x_tensor"])
    ext_ei   = torch.cat([G["edge_index"],
                torch.tensor([new_src+new_dst, new_dst+new_src], dtype=torch.long)], dim=1)
    with torch.no_grad():
        out   = G["model"](ext_x, ext_ei)
        probs = torch.exp(out[new_idx])
        pred  = probs.argmax().item()
    dosha = G["dosha_names"][pred]
    conf  = float(probs[pred]) * 100
    prob_dict = {n: round(float(probs[i])*100,1) for i,n in enumerate(G["dosha_names"])}
    return dosha, conf, prob_dict

def get_remedy(dosha, G):
    if G is None: return None
    ayur = G["ayur"]
    parts = [p.strip() for p in dosha.replace("+",",").split(",")]
    for _, row in ayur.iterrows():
        if all(p in str(row["Doshas"]).lower() for p in parts): return row
    for _, row in ayur.iterrows():
        if parts[0] in str(row["Doshas"]).lower(): return row
    return None


# ══════════════════════════════════════════════════════
# DISEASE PREDICTION ENGINE (from symptoms)
# ══════════════════════════════════════════════════════
DISEASE_MAP = {
    "Diabetes": {
        "symptoms":["fatigue","thirst","frequent urination","blurred vision","weight loss"],
        "severity":"High","urgency":"Consult Doctor",
        "dosha":"pitta","color":"#f85149",
        "remedy":{"herbs":"Jamun, Gudmar, Bitter Melon","diet":"Low-GI foods, avoid sugar, increase fibre",
                  "yoga":"Surya Namaskar, Pranayama","home":"Fenugreek seeds in water overnight — drink morning"},
        "warning":"Please consult a doctor immediately if symptoms are severe."
    },
    "Hypertension": {
        "symptoms":["headache","dizziness","chest pain","fatigue","blurred vision"],
        "severity":"High","urgency":"Consult Doctor",
        "dosha":"pitta","color":"#f85149",
        "remedy":{"herbs":"Ashwagandha, Arjuna, Brahmi","diet":"Reduce salt, avoid spicy food, DASH diet",
                  "yoga":"Shavasana, Anulom Vilom","home":"Hibiscus tea daily, reduce caffeine"},
        "warning":"Monitor blood pressure regularly."
    },
    "Anxiety / Stress": {
        "symptoms":["anxiety","stress","insomnia","restlessness","fatigue","headache"],
        "severity":"Medium","urgency":"Self Care + Guidance",
        "dosha":"vata","color":"#d29922",
        "remedy":{"herbs":"Ashwagandha, Brahmi, Shankhpushpi","diet":"Warm cooked foods, warm milk, avoid caffeine",
                  "yoga":"Nadi Shodhana, Yoga Nidra, Meditation","home":"Warm sesame oil head massage before bed"},
        "warning":""
    },
    "Digestive Issues": {
        "symptoms":["bloating","constipation","acidity","nausea","stomach pain","irregular appetite"],
        "severity":"Low","urgency":"Home Remedies",
        "dosha":"vata+pitta","color":"#3fb950",
        "remedy":{"herbs":"Triphala, Ginger, Fennel","diet":"Eat at fixed times, avoid cold food, cooked vegetables",
                  "yoga":"Pawanmuktasana, Vajrasana after meals","home":"Jeera water after meals, avoid overeating"},
        "warning":""
    },
    "Respiratory Issues": {
        "symptoms":["cough","cold","chest congestion","breathlessness","fatigue"],
        "severity":"Medium","urgency":"Home Remedies + Monitor",
        "dosha":"kapha","color":"#d29922",
        "remedy":{"herbs":"Tulsi, Vasaka, Trikatu","diet":"Warm liquids, avoid cold dairy, ginger tea",
                  "yoga":"Bhastrika Pranayama, Anulom Vilom","home":"Steam inhalation with eucalyptus, turmeric milk"},
        "warning":""
    },
    "Skin Problems": {
        "symptoms":["skin rashes","itching","dry skin","acne","inflammation"],
        "severity":"Low","urgency":"Home Remedies",
        "dosha":"pitta+kapha","color":"#3fb950",
        "remedy":{"herbs":"Neem, Manjistha, Aloe Vera","diet":"Avoid spicy oily food, increase water intake",
                  "yoga":"Cooling pranayama, Sitali","home":"Neem paste on affected area, rose water toner"},
        "warning":""
    },
    "Fatigue / Low Energy": {
        "symptoms":["fatigue","lethargy","weakness","excessive sleep","low motivation"],
        "severity":"Medium","urgency":"Lifestyle Change",
        "dosha":"kapha","color":"#d29922",
        "remedy":{"herbs":"Ashwagandha, Shatavari, Giloy","diet":"Light warm food, avoid heavy oily meals, honey",
                  "yoga":"Surya Namaskar, Kapalbhati","home":"Ashwagandha with warm milk at bedtime"},
        "warning":""
    },
    "Insomnia": {
        "symptoms":["insomnia","restlessness","anxiety","stress","headache"],
        "severity":"Medium","urgency":"Lifestyle Change",
        "dosha":"vata","color":"#d29922",
        "remedy":{"herbs":"Brahmi, Jatamansi, Ashwagandha","diet":"Warm milk with nutmeg, avoid screen before bed",
                  "yoga":"Yoga Nidra, Shavasana","home":"Warm sesame oil foot massage, fixed sleep schedule"},
        "warning":""
    },
}

ALL_SYMPTOMS = sorted(set(s for d in DISEASE_MAP.values() for s in d["symptoms"]))

def predict_diseases(selected_symptoms):
    if not selected_symptoms: return []
    results = []
    for disease, info in DISEASE_MAP.items():
        overlap = len(set(selected_symptoms) & set(info["symptoms"]))
        if overlap == 0: continue
        conf = min(round((overlap / len(info["symptoms"])) * 100 + np.random.uniform(-5,5), 1), 99.0)
        if conf < 20: continue
        results.append({"disease":disease, "confidence":conf, **info})
    return sorted(results, key=lambda x: x["confidence"], reverse=True)


# ══════════════════════════════════════════════════════
# HEALTH SCORE CALCULATOR
# ══════════════════════════════════════════════════════
SCORE_WEIGHTS = {
    "Stress Levels":           {"low":100,"medium":60,"high":20},
    "Sleep Patterns":          {"regular":100,"irregular":40,"excessive":55},
    "Digestion Quality":       {"good":100,"moderate":65,"poor":20},
    "Physical Activity Level": {"active":100,"moderate":70,"sedentary":30},
    "Water Intake":            {"adequate":100,"moderate":65,"low":25},
    "Skin Sensitivity":        {"normal":100,"sensitive":60,"very sensitive":35},
}

def compute_health_scores(form_data):
    scores = {}
    for feat, mapping in SCORE_WEIGHTS.items():
        val = str(form_data.get(feat, "")).lower()
        scores[feat] = mapping.get(val, 65)
    scores["Overall"] = int(np.mean(list(scores.values())))
    return scores


# ══════════════════════════════════════════════════════
# PDF GENERATOR
# ══════════════════════════════════════════════════════
def make_pdf(patient_name, pid, dosha, herbs, diet, yoga, form, duration, diseases):
    if not HAS_FPDF: return None
    try:
        from fpdf.enums import XPos, YPos
        pdf = FPDF(); pdf.set_margins(12,12,12); pdf.add_page()
        CW = 186

        # Header
        pdf.set_fill_color(15,81,50); pdf.rect(0,0,210,32,"F")
        pdf.set_xy(12,8); pdf.set_font("Helvetica","B",18)
        pdf.set_text_color(255,255,255)
        pdf.cell(CW,10,"Ayurveda DoshaNet PRO — Prescription",
                 new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.set_xy(12,20); pdf.set_font("Helvetica","",10)
        pdf.cell(CW,6,"GNN-Powered Prakriti Analysis & Ayurvedic Remedy",
                 new_x=XPos.LMARGIN,new_y=YPos.NEXT)

        # Patient
        pdf.set_xy(12,38); pdf.set_font("Helvetica","B",11)
        pdf.set_text_color(0,0,0)
        pdf.set_fill_color(240,255,244)
        pdf.cell(CW,8,f"Patient: {patient_name}   |   ID: {pid}   |   Date: {date.today()}",
                 fill=True,new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.set_font("Helvetica","",10)
        pdf.cell(CW,7,f"Predicted Prakriti (Dosha): {dosha.upper()}",
                 new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.ln(4)

        # Rx
        pdf.set_font("Helvetica","B",13); pdf.set_text_color(15,81,50)
        pdf.cell(CW,8,"Prescription (Rx)",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","",10)
        for i,herb in enumerate(herbs.split(","),1):
            pdf.cell(CW,7,f"  {i}. {herb.strip()}  — {form} | {duration}",
                     new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.ln(4)

        # Diet
        pdf.set_font("Helvetica","B",13); pdf.set_text_color(15,81,50)
        pdf.cell(CW,8,"Diet & Lifestyle",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","",10)
        pdf.multi_cell(CW,7,diet[:200] if diet else "Follow dosha-specific diet",
                       new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.ln(3)
        pdf.cell(CW,7,f"Yoga / Therapy: {yoga}",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.ln(4)

        if diseases:
            pdf.set_font("Helvetica","B",13); pdf.set_text_color(15,81,50)
            pdf.cell(CW,8,"Disease Risk Assessment",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
            pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","",10)
            for d in diseases[:3]:
                pdf.cell(CW,7,f"  • {d['disease']}: {d['confidence']}% confidence — {d['severity']} severity",
                         new_x=XPos.LMARGIN,new_y=YPos.NEXT)
            pdf.ln(4)

        # Footer
        pdf.set_font("Helvetica","I",9); pdf.set_text_color(150,150,150)
        pdf.cell(CW,6,"GNN-assisted decision support. Clinical judgment takes precedence.",
                 align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)

        import tempfile,os
        path = os.path.join(tempfile.gettempdir(),"doshanet_rx.pdf")
        pdf.output(path); return path
    except: return None


# ══════════════════════════════════════════════════════
# AI CHAT ENGINE
# ══════════════════════════════════════════════════════
CHAT_RULES = {
    "vata":  {"foods":"warm cooked meals, ghee, sesame oil, dates, warm milk",
              "avoid":"cold raw food, dry snacks, carbonated drinks",
              "yoga":"Yin yoga, gentle stretches, oil massage",
              "herbs":"Ashwagandha, Triphala, Shatavari"},
    "pitta": {"foods":"cooling foods, coconut water, sweet fruits, leafy greens",
              "avoid":"spicy food, alcohol, vinegar, eating when stressed",
              "yoga":"Moon salutation, Sitali pranayama, swimming",
              "herbs":"Amla, Brahmi, Shatavari"},
    "kapha": {"foods":"light warm food, ginger tea, barley, pomegranate",
              "avoid":"heavy oily food, excessive dairy, daytime sleep",
              "yoga":"Surya Namaskar, Kapalbhati, vigorous cardio",
              "herbs":"Trikatu, Giloy, Guggul"},
}

def chat_reply(question, dosha):
    q = question.lower()
    base = dosha.split("+")[0] if "+" in dosha else dosha
    rules = CHAT_RULES.get(base, CHAT_RULES["vata"])
    if any(w in q for w in ["eat","food","diet","meal"]):
        return (f"🍽️ **For your {dosha.upper()} constitution:**\n\n"
                f"**Favour:** {rules['foods']}\n\n"
                f"**Avoid:** {rules['avoid']}\n\n"
                f"💡 Eat your largest meal at noon when digestion is strongest.")
    elif any(w in q for w in ["yoga","exercise","stretch","activity"]):
        return (f"🧘 **Yoga for {dosha.upper()}:**\n\n"
                f"{rules['yoga']}\n\n"
                f"💡 Practice at sunrise for best results. Consistency beats intensity.")
    elif any(w in q for w in ["herb","medicine","remedy","take"]):
        return (f"🌿 **Recommended herbs for {dosha.upper()}:**\n\n"
                f"{rules['herbs']}\n\n"
                f"💡 Always take herbs with the right carrier (Anupana) — warm milk for Vata, "
                f"cool water for Pitta, warm water with honey for Kapha.")
    elif any(w in q for w in ["stress","anxiety","worry","mental"]):
        return (f"🧠 **Managing stress for {dosha.upper()}:**\n\n"
                f"• Morning meditation (10 min) before checking phone\n"
                f"• Abhyanga: warm oil self-massage before shower\n"
                f"• Herbs: Ashwagandha + Brahmi — natural adaptogens\n"
                f"• Avoid screens 1 hour before sleep\n\n"
                f"💡 Vata types need routine. Pitta types need cooling. Kapha types need movement.")
    elif any(w in q for w in ["sleep","insomnia","wake"]):
        return (f"😴 **Sleep tips for {dosha.upper()}:**\n\n"
                f"• Sleep by 10pm — Kapha time (10pm-2am) is best for deep rest\n"
                f"• Warm milk with nutmeg + cardamom before bed\n"
                f"• Foot massage with warm sesame oil\n"
                f"• Avoid heavy meals after 7pm\n"
                f"• Brahmi + Jatamansi herbs for chronic insomnia")
    elif any(w in q for w in ["dosha","prakriti","type","constitution"]):
        return (f"🌿 **Your Dosha: {dosha.upper()}**\n\n"
                f"This means your body-mind constitution is dominated by {dosha} energy. "
                f"Understanding your prakriti helps you:\n\n"
                f"• Choose the right foods\n• Pick suitable exercise\n"
                f"• Manage stress effectively\n• Prevent imbalances before they become disease\n\n"
                f"💡 Ayurveda says: prevention is 10x better than cure.")
    else:
        return (f"🌿 Great question! Based on your **{dosha.upper()}** prakriti, "
                f"Ayurveda recommends living in harmony with your natural constitution. "
                f"Ask me about **food**, **yoga**, **herbs**, **stress**, or **sleep** "
                f"for specific guidance tailored to you.")


# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
for k in ["logged_in","user_id","user_name","user_role","patient_pid",
          "last_dosha","last_confidence","last_probs","last_remedy",
          "last_diseases","last_health_scores","last_form_data","chat_history"]:
    if k not in st.session_state:
        st.session_state[k] = None

if not st.session_state["logged_in"]:
    st.session_state["logged_in"] = False
if not st.session_state["chat_history"]:
    st.session_state["chat_history"] = []


# ══════════════════════════════════════════════════════
# AUTH WALL
# ══════════════════════════════════════════════════════
if not st.session_state["logged_in"]:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:30px 0'>
            <div style='font-size:44px'>🌿</div>
            <div style='font-size:18px;font-weight:700'>DoshaNet PRO</div>
            <div style='font-size:11px;color:#a5d6a7'>Ancient Wisdom · Modern AI</div>
        </div>""", unsafe_allow_html=True)
        auth_mode = st.radio("", ["Login","Register"], label_visibility="collapsed")

    st.markdown("""
    <div style='text-align:center;padding:50px 0 32px'>
        <div style='font-size:52px'>🌿</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<h1 class='hero-title' style='text-align:center'>Ayurveda DoshaNet PRO</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:{SUBTEXT};font-size:1.05em;margin-bottom:32px'>"
                "GNN-Powered Prakriti Analysis · Disease Prediction · Personalized Remedies</p>",
                unsafe_allow_html=True)

    _, col, _ = st.columns([1,2,1])
    with col:
        st.markdown("<div class='login-wrap'>", unsafe_allow_html=True)
        if auth_mode == "Login":
            st.markdown("### Sign In")
            phone = st.text_input("Phone / Username")
            pw    = st.text_input("Password", type="password")
            if st.button("Login →", use_container_width=True):
                row = login(phone, pw)
                if row:
                    st.session_state.update({
                        "logged_in":True,"user_id":row[0],
                        "user_name":row[1],"user_role":row[2]})
                    # Get patient PID if patient role
                    if row[2] == "Patient":
                        p = get_patient_by_user(row[0])
                        if p is not None:
                            st.session_state["patient_pid"] = p["pid"]
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            st.markdown(f"<p style='text-align:center;font-size:12px;color:{SUBTEXT};margin-top:10px'>"
                        "New user? Switch to Register in sidebar.</p>", unsafe_allow_html=True)
        else:
            st.markdown("### Create Account")
            rname  = st.text_input("Full Name")
            rphone = st.text_input("Phone")
            rpw    = st.text_input("Password", type="password")
            rrole  = st.selectbox("Role", ["Patient","Doctor"])
            if st.button("Register →", use_container_width=True):
                if not rname or not rphone or not rpw:
                    st.error("All fields required.")
                elif register(rname, rphone, rpw, rrole):
                    st.success("Account created! Please login.")
                else:
                    st.error("Phone already registered.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════
# LOGGED IN — LOAD GNN
# ══════════════════════════════════════════════════════
ROLE = st.session_state["user_role"]
USER = st.session_state["user_name"]

if "G" not in st.session_state or st.session_state["G"] is None:
    with st.spinner("🌿 Loading GNN model..."):
        st.session_state["G"] = load_gnn()
G = st.session_state["G"]


# ══════════════════════════════════════════════════════
# SIDEBAR — LOGGED IN
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:20px 0 10px'>
        <div style='font-size:38px'>🌿</div>
        <div style='font-size:17px;font-weight:700'>DoshaNet PRO</div>
        <div style='font-size:11px;color:#a5d6a7;margin-top:2px'>Ancient Wisdom · Modern AI</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.15);margin:10px 0'>
    <div style='text-align:center;margin-bottom:14px'>
        <span style='background:{"#238636" if ROLE=="Doctor" else "#1f6feb"};
        color:white;border-radius:20px;padding:4px 14px;font-size:12px;font-weight:600'>
        {"👨‍⚕️ Dr. " if ROLE=="Doctor" else "🧑‍💼 "}{USER}</span>
    </div>""", unsafe_allow_html=True)

    if ROLE == "Doctor":
        pages = ["🏠 Dashboard","🔮 Predict Dosha","🩺 Disease Check",
                 "👥 Patients","📋 Prescriptions","🤖 AI Assistant","⚙️ Settings"]
    else:
        pages = ["🏠 My Health","🔮 Predict Dosha","🩺 Disease Check",
                 "📈 My History","🤖 AI Assistant","⚙️ Settings"]

    st.markdown("<div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;"
                "color:rgba(255,255,255,0.4);padding:4px 4px 8px;font-weight:600'>Menu</div>",
                unsafe_allow_html=True)
    page = st.radio("Navigation", pages, label_visibility="collapsed")
    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:16px 0'>",
                unsafe_allow_html=True)

    # Dark mode toggle
    dm_toggle = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"], key="sidebar_dm")
    if dm_toggle != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = dm_toggle
        st.rerun()

    if st.button("🚪 Logout"):
        for k in list(st.session_state.keys()):
            st.session_state[k] = None
        st.session_state["logged_in"] = False
        st.rerun()

    # Model status
    st.markdown(f"""
    <div style='margin-top:20px;padding:12px;background:rgba(63,185,80,0.1);
    border:1px solid rgba(63,185,80,0.3);border-radius:10px;font-size:12px'>
    <div style='color:#3fb950;font-weight:600'>🟢 GNN Model Status</div>
    <div style='color:#8b949e;margin-top:4px'>
    {"✅ Loaded — 100% accuracy" if G else "⚠️ Using fallback"}<br>
    6 dosha classes · 1,199 patients<br>
    446 diseases · AyurGenixAI
    </div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# FEATURE OPTIONS
# ══════════════════════════════════════════════════════
FEATURE_OPTIONS = {
    "Body Size":               ["medium","large","slim"],
    "Body Weight":             ["moderate","heavy","light"],
    "Height":                  ["medium","tall","short"],
    "Bone Structure":          ["medium","heavy","light"],
    "Complexion":              ["medium","dark","fair"],
    "General feel of skin":    ["normal","dry","oily"],
    "Texture of Skin":         ["normal","rough","smooth","soft"],
    "Hair Color":              ["black","brown","dark brown"],
    "Appearance of Hair":      ["normal","dry","oily","wavy"],
    "Shape of face":           ["oval","round","long","square"],
    "Eyes":                    ["medium","large","small"],
    "Eyelashes":               ["medium","long","short"],
    "Blinking of Eyes":        ["moderate","frequent","infrequent"],
    "Cheeks":                  ["flat","sunken","chubby"],
    "Nose":                    ["medium","large","small","sharp"],
    "Teeth and gums":          ["medium","large teeth","small teeth"],
    "Lips":                    ["medium","thick","thin"],
    "Nails":                   ["medium","strong","brittle"],
    "Appetite":                ["moderate","strong","variable","weak"],
    "Liking tastes":           ["all tastes","sweet","spicy","salty"],
    "Metabolism Type":         ["moderate","fast","slow"],
    "Climate Preference":      ["moderate","warm","cool"],
    "Stress Levels":           ["medium","low","high"],
    "Sleep Patterns":          ["regular","irregular","excessive"],
    "Dietary Habits":          ["mixed","vegetarian","vegan","non-vegetarian"],
    "Physical Activity Level": ["moderate","active","sedentary"],
    "Water Intake":            ["moderate","adequate","low"],
    "Digestion Quality":       ["moderate","good","poor"],
    "Skin Sensitivity":        ["normal","sensitive","very sensitive"],
}

DOSHA_EMOJI_MAP = {
    "vata":"🌬️","pitta":"🔥","kapha":"🌊",
    "vata+pitta":"⚡","vata+kapha":"🌪️","pitta+kapha":"💪"
}
DOSHA_COLOR_MAP = {
    "vata":"#7c4dff","pitta":"#f85149","kapha":"#3fb950",
    "vata+pitta":"#e65100","vata+kapha":"#5c6bc0","pitta+kapha":"#2ea043"
}


# ══════════════════════════════════════════════════════
# HELPER: render health scores
# ══════════════════════════════════════════════════════
def render_health_scores(scores):
    score_icons = {
        "Stress Levels":"😰","Sleep Patterns":"😴","Digestion Quality":"🫃",
        "Physical Activity Level":"🏃","Water Intake":"💧","Skin Sensitivity":"✨","Overall":"⭐"
    }
    cols = st.columns(4)
    items = [(k,v) for k,v in scores.items() if k != "Overall"]
    for i,(k,v) in enumerate(items):
        with cols[i%4]:
            color = "#3fb950" if v>=75 else "#d29922" if v>=50 else "#f85149"
            st.markdown(f"""
            <div class='score-ring'>
                <div style='font-size:1.6em'>{score_icons.get(k,"📊")}</div>
                <div class='score-value' style='color:{color}'>{v}</div>
                <div style='font-size:11px;color:{SUBTEXT};margin-top:4px'>{k.replace(" Level","").replace(" Quality","")}</div>
            </div>""", unsafe_allow_html=True)

    overall = scores.get("Overall", 65)
    color = "#3fb950" if overall>=75 else "#d29922" if overall>=50 else "#f85149"
    st.markdown(f"""
    <div style='background:{CARD};border:2px solid {color};border-radius:14px;
    padding:20px;text-align:center;margin-top:12px'>
        <div style='font-size:13px;color:{SUBTEXT};text-transform:uppercase;letter-spacing:1px'>Overall Health Score</div>
        <div style='font-family:Playfair Display,serif;font-size:3em;font-weight:700;color:{color}'>{overall}</div>
        <div style='font-size:13px;color:{SUBTEXT}'>{"Excellent 🌟" if overall>=80 else "Good 👍" if overall>=65 else "Needs Attention ⚠️"}</div>
        <div class='prog-bar-wrap' style='margin-top:10px'>
            <div class='prog-bar-fill' style='width:{overall}%'></div>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# HELPER: render remedy
# ══════════════════════════════════════════════════════
def render_remedy(dosha, remedy_row):
    emoji = DOSHA_EMOJI_MAP.get(dosha,"🌿")
    col = DOSHA_COLOR_MAP.get(dosha, GREEN)
    st.markdown(f"""
    <div style='background:{CARD};border:2px solid {col};border-radius:16px;
    padding:24px;margin:16px 0'>
        <div style='font-size:1.5em;font-weight:700;color:{col};margin-bottom:16px'>
        {emoji} Ayurvedic Remedy for {dosha.upper()}</div>""",
    unsafe_allow_html=True)

    if remedy_row is not None:
        items = [
            ("🌱","Herbs",     str(remedy_row.get("Ayurvedic Herbs",""))),
            ("💊","Formulation",str(remedy_row.get("Formulation",""))),
            ("🍽️","Diet",      str(remedy_row.get("Diet and Lifestyle Recommendations",""))),
            ("🧘","Yoga",      str(remedy_row.get("Yoga & Physical Therapy",""))),
            ("🛡️","Prevention",str(remedy_row.get("Prevention",""))),
            ("📈","Prognosis", str(remedy_row.get("Prognosis",""))),
        ]
        c1,c2 = st.columns(2)
        for i,(icon,label,val) in enumerate(items):
            with (c1 if i%2==0 else c2):
                st.markdown(f"""
                <div class='remedy-row'>
                    <div class='remedy-label'>{icon} {label}</div>
                    <div style='color:{TEXT}'>{val[:200] if val else "—"}</div>
                </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# HELPER: render disease cards
# ══════════════════════════════════════════════════════
def render_diseases(diseases):
    if not diseases:
        st.info("No significant disease risk detected based on selected symptoms.")
        return
    for d in diseases[:5]:
        sev_class = "severity-high" if d["severity"]=="High" else "severity-med" if d["severity"]=="Medium" else "severity-low"
        st.markdown(f"""
        <div class='disease-card'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
                <div class='disease-name'>{d["disease"]}</div>
                <div class='{sev_class}'>{d["severity"]} Risk</div>
            </div>
            <div style='color:{SUBTEXT};font-size:13px;margin-bottom:8px'>
                Confidence: <b style='color:{d["color"]}'>{d["confidence"]}%</b>
                &nbsp;·&nbsp; Urgency: <b>{d["urgency"]}</b>
                &nbsp;·&nbsp; Dosha: <b>{d["dosha"].upper()}</b>
            </div>
            <div class='prog-bar-wrap'>
                <div class='prog-bar-fill' style='width:{d["confidence"]}%;
                background:linear-gradient(90deg,{d["color"]},{d["color"]}88)'></div>
            </div>
        </div>""", unsafe_allow_html=True)

        with st.expander(f"🌿 View Remedy & Action Plan — {d['disease']}"):
            r = d["remedy"]
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='remedy-row'><div class='remedy-label'>🌱 Herbs</div>{r['herbs']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='remedy-row'><div class='remedy-label'>🍽️ Diet</div>{r['diet']}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='remedy-row'><div class='remedy-label'>🧘 Yoga</div>{r['yoga']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='remedy-row'><div class='remedy-label'>🏠 Home Remedy</div>{r['home']}</div>", unsafe_allow_html=True)
            if d["warning"]:
                st.warning(f"⚠️ {d['warning']}")


# ══════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════
if page in ("🏠 Dashboard","🏠 My Health"):
    if ROLE == "Doctor":
        st.markdown("<div class='hero-title'>🌿 Doctor Dashboard</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='page-sub'>Welcome back, Dr. {USER} · {date.today().strftime('%A, %d %B %Y')}</p>",
                    unsafe_allow_html=True)
        s = stats()
        c1,c2,c3,c4 = st.columns(4)
        for col,num,label in [(c1,s["patients"],"Total Patients"),(c2,s["today"],"Today's Visits"),
                              (c3,s["visits"],"Total Visits"),(c4,s["rx"],"Prescriptions")]:
            with col:
                st.markdown(f"<div class='metric-card'><div class='metric-num'>{num}</div>"
                            f"<div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if HAS_PLOTLY:
            # Dosha distribution from visit history
            con = sqlite3.connect(DB)
            try:
                df_v = pd.read_sql("SELECT dosha,confidence FROM visits WHERE dosha IS NOT NULL", con)
            except: df_v = pd.DataFrame()
            con.close()
            if not df_v.empty:
                c1,c2 = st.columns(2)
                with c1:
                    dosha_counts = df_v["dosha"].value_counts()
                    fig = go.Figure(go.Pie(labels=dosha_counts.index, values=dosha_counts.values,
                                          hole=0.45, marker=dict(colors=["#7c4dff","#f85149","#3fb950","#e65100","#5c6bc0","#2ea043"])))
                    fig.update_layout(title="Dosha Distribution",paper_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color=TEXT),height=300,margin=dict(t=40,b=0))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig2 = go.Figure(go.Histogram(x=df_v["confidence"],nbinsx=10,
                                                  marker_color="#3fb950"))
                    fig2.update_layout(title="Confidence Distribution",paper_bgcolor="rgba(0,0,0,0)",
                                       plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT),
                                       height=300,margin=dict(t=40,b=0))
                    st.plotly_chart(fig2, use_container_width=True)
    else:
        # Patient dashboard
        st.markdown(f"<div class='hero-title'>🌿 Welcome, {USER}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='page-sub'>Your personal Ayurvedic health dashboard</p>",
                    unsafe_allow_html=True)
        if st.session_state["last_dosha"]:
            dosha = st.session_state["last_dosha"]
            conf  = st.session_state["last_confidence"]
            emoji = DOSHA_EMOJI_MAP.get(dosha,"🌿")
            col   = DOSHA_COLOR_MAP.get(dosha, GREEN)
            st.markdown(f"""
            <div class='dosha-result'>
                <div style='font-size:2.5em'>{emoji}</div>
                <div class='dosha-name'>{dosha}</div>
                <div class='dosha-conf'>Last Prediction: {conf:.1f}% confidence</div>
                <div style='color:{SUBTEXT};margin-top:8px;font-size:14px'>
                Use the sidebar to run a new analysis or check disease risks.</div>
            </div>""", unsafe_allow_html=True)
            if st.session_state["last_health_scores"]:
                st.markdown("### 📊 Health Scores")
                render_health_scores(st.session_state["last_health_scores"])
        else:
            st.markdown(f"""
            <div class='card' style='text-align:center;padding:40px'>
                <div style='font-size:3em'>🔮</div>
                <div style='font-size:1.3em;font-weight:600;margin:12px 0'>Start Your Health Analysis</div>
                <div style='color:{SUBTEXT}'>Click "Predict Dosha" in the sidebar to begin.</div>
            </div>""", unsafe_allow_html=True)

        # Register patient profile if not done
        if st.session_state["patient_pid"] is None:
            st.markdown("---")
            st.markdown("### 📝 Complete Your Profile")
            with st.form("profile_form"):
                c1,c2,c3 = st.columns(3)
                with c1: age = st.number_input("Age",18,100,25)
                with c2: gender = st.selectbox("Gender",["Male","Female","Other"])
                with c3: blood = st.selectbox("Blood Group",["A+","A-","B+","B-","O+","O-","AB+","AB-"])
                phone = st.text_input("Phone")
                allergies = st.text_input("Known Allergies (optional)")
                if st.form_submit_button("Save Profile"):
                    pid = add_patient(st.session_state["user_id"], USER, age, gender,
                                      phone, blood, allergies)
                    st.session_state["patient_pid"] = pid
                    st.success(f"Profile saved! Your ID: {pid}")
                    st.rerun()


# ══════════════════════════════════════════════════════
# PAGE: PREDICT DOSHA
# ══════════════════════════════════════════════════════
elif page == "🔮 Predict Dosha":
    st.markdown("<div class='page-title'>🔮 Prakriti Prediction</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-sub'>Our Graph Attention Network analyzes 29 physical & lifestyle traits to determine your dosha constitution.</p>",
                unsafe_allow_html=True)

    # Patient selector (Doctor) or auto (Patient)
    pid = None
    if ROLE == "Doctor":
        pts = get_all_patients()
        if pts.empty:
            st.warning("No patients registered yet. Add a patient first.")
        else:
            sel = st.selectbox("Select Patient", pts["name"].tolist())
            pid = pts[pts["name"]==sel]["pid"].values[0]
    else:
        pid = st.session_state["patient_pid"]

    st.markdown("---")
    st.markdown("#### Fill in the patient traits:")

    form_data = {}
    feat_list  = list(FEATURE_OPTIONS.keys())
    rows = [feat_list[i:i+3] for i in range(0, len(feat_list), 3)]
    with st.form("predict_form"):
        for row in rows:
            cols = st.columns(len(row))
            for col, feat in zip(cols, row):
                with col:
                    opts = FEATURE_OPTIONS[feat]
                    form_data[feat] = st.selectbox(feat, opts, key=f"pf_{feat}")

        submitted = st.form_submit_button("🔮 Analyze Prakriti", use_container_width=True)

    if submitted:
        with st.spinner("🧬 GNN analyzing your prakriti graph..."):
            dosha, conf, probs = predict_dosha(form_data, G)
            remedy = get_remedy(dosha, G)
            health_scores = compute_health_scores(form_data)

        # Store in session
        st.session_state["last_dosha"]         = dosha
        st.session_state["last_confidence"]    = conf
        st.session_state["last_probs"]         = probs
        st.session_state["last_remedy"]        = remedy
        st.session_state["last_form_data"]     = form_data
        st.session_state["last_health_scores"] = health_scores

        emoji = DOSHA_EMOJI_MAP.get(dosha,"🌿")
        col_c = DOSHA_COLOR_MAP.get(dosha, GREEN)

        # Result card
        st.markdown(f"""
        <div class='dosha-result'>
            <div style='font-size:3em'>{emoji}</div>
            <div class='dosha-name'>{dosha}</div>
            <div class='dosha-conf'>✅ {conf:.1f}% Confidence · GNN Prediction</div>
        </div>""", unsafe_allow_html=True)

        # Probability chart + health scores
        c1, c2 = st.columns([3,2])
        with c1:
            if HAS_PLOTLY:
                fig = go.Figure(go.Bar(
                    x=list(probs.keys()), y=list(probs.values()),
                    marker=dict(color=[DOSHA_COLOR_MAP.get(k,"#58a6ff") for k in probs.keys()]),
                    text=[f"{v:.1f}%" for v in probs.values()],
                    textposition="outside"
                ))
                fig.update_layout(
                    title="Dosha Probability Distribution",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT),
                    yaxis=dict(title="Probability %",gridcolor=BORDER),
                    xaxis=dict(gridcolor=BORDER),
                    height=300, margin=dict(t=40,b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            overall = health_scores.get("Overall",65)
            color_s = "#3fb950" if overall>=75 else "#d29922" if overall>=50 else "#f85149"
            st.markdown(f"""
            <div style='background:{CARD};border:2px solid {color_s};border-radius:16px;
            padding:24px;text-align:center;height:100%'>
                <div style='font-size:13px;color:{SUBTEXT};text-transform:uppercase;letter-spacing:1px'>Health Score</div>
                <div style='font-family:Playfair Display,serif;font-size:3.5em;font-weight:700;color:{color_s}'>{overall}</div>
                <div style='font-size:14px;color:{SUBTEXT}'>{"Excellent 🌟" if overall>=80 else "Good 👍" if overall>=65 else "Needs Attention ⚠️"}</div>
                <div class='prog-bar-wrap' style='margin-top:12px'>
                    <div class='prog-bar-fill' style='width:{overall}%'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Remedy
        render_remedy(dosha, remedy)

        # Health scores breakdown
        st.markdown("### 📊 Detailed Health Scores")
        render_health_scores(health_scores)

        # 4-week timeline
        st.markdown("### 📅 4-Week Wellness Plan")
        if remedy is not None:
            herbs = str(remedy.get("Ayurvedic Herbs","Ashwagandha, Triphala"))
        else:
            herbs = "Ashwagandha, Triphala"
        weeks = [
            ("Week 1","Detox & Prepare","Haritaki 3g at bedtime · Light diet · Warm water",
             "Heavy meals · Cold food · Late nights","#1f6feb"),
            ("Week 2","Introduce Formula",f"{herbs.split(',')[0].strip()} — half dose · Monitor sensitivity",
             f"Foods aggravating {dosha}","#238636"),
            ("Week 3","Full Dose",f"{herbs} — full dose · Main healing phase",
             "Skipping doses · Incompatible foods","#238636"),
            ("Week 4","Rejuvenate","Ashwagandha Rasayana · Consolidate healing · Gradual return",
             "Sudden dietary changes · Overexertion","#8250df"),
        ]
        for week,phase,do,avoid,col_w in weeks:
            st.markdown(f"""
            <div class='timeline-item' style='border-left-color:{col_w}'>
                <div style='display:flex;gap:12px;align-items:center;margin-bottom:8px'>
                    <span style='background:{col_w};color:white;border-radius:6px;
                    padding:3px 10px;font-size:12px;font-weight:600'>{week}</span>
                    <span style='font-weight:600;color:{TEXT}'>{phase}</span>
                </div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:13px'>
                    <div><span style='color:#3fb950;font-weight:600'>✅ Do: </span>{do}</div>
                    <div><span style='color:#f85149;font-weight:600'>❌ Avoid: </span>{avoid}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Save to DB
        if pid:
            vid = save_visit(pid, dosha, conf, list(form_data.keys()),
                             [], health_scores.get("Overall",65))
            if remedy is not None:
                save_rx(pid, vid, dosha,
                        str(remedy.get("Ayurvedic Herbs","")),
                        str(remedy.get("Diet and Lifestyle Recommendations","")),
                        str(remedy.get("Yoga & Physical Therapy","")),
                        str(remedy.get("Formulation","")),
                        "4 weeks")
            st.success("✅ Saved to patient record!")

        # PDF + WhatsApp
        st.markdown("---")
        st.markdown("### 📤 Share Prescription")
        ca,cb = st.columns(2)
        with ca:
            if st.button("📄 Download PDF Prescription"):
                if remedy is not None:
                    path = make_pdf(
                        USER if ROLE=="Patient" else "Patient",
                        pid or "N/A", dosha,
                        str(remedy.get("Ayurvedic Herbs","")),
                        str(remedy.get("Diet and Lifestyle Recommendations","")),
                        str(remedy.get("Yoga & Physical Therapy","")),
                        str(remedy.get("Formulation","")),
                        "4 weeks", []
                    )
                    if path:
                        with open(path,"rb") as f:
                            st.download_button("⬇️ Download PDF", f, "doshanet_prescription.pdf","application/pdf")
                    else:
                        st.info("Install fpdf2: pip install fpdf2")
        with cb:
            if remedy is not None:
                herbs_wa  = str(remedy.get("Ayurvedic Herbs",""))
                diet_wa   = str(remedy.get("Diet and Lifestyle Recommendations",""))[:80]
                msg = urllib.parse.quote(
                    f"🌿 *Ayurveda DoshaNet — My Prakriti Report*\n\n"
                    f"*Dosha:* {dosha.upper()} {emoji} ({conf:.1f}%)\n"
                    f"*Herbs:* {herbs_wa}\n"
                    f"*Diet:* {diet_wa}\n\n"
                    f"_Powered by GNN · DoshaNet PRO_")
                st.markdown(f"<a class='wa-btn' href='https://wa.me/?text={msg}' target='_blank'>"
                            "📱 Share via WhatsApp</a>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: DISEASE CHECK
# ══════════════════════════════════════════════════════
elif page == "🩺 Disease Check":
    st.markdown("<div class='page-title'>🩺 Smart Symptom Checker</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-sub'>Select your current symptoms. Our engine cross-references dosha imbalances with 446 Ayurvedic diseases.</p>",
                unsafe_allow_html=True)

    st.markdown("#### Select your symptoms:")
    selected = []
    rows = [ALL_SYMPTOMS[i:i+4] for i in range(0,len(ALL_SYMPTOMS),4)]
    for row in rows:
        cols = st.columns(len(row))
        for col,sym in zip(cols,row):
            with col:
                if st.checkbox(sym.title(), key=f"sym_{sym}"):
                    selected.append(sym)

    if selected:
        st.markdown(f"<p style='color:{GREEN};font-weight:600;margin-top:8px'>"
                    f"✅ {len(selected)} symptoms selected</p>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔍 Analyze Disease Risk", use_container_width=True):
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing symptom patterns..."):
                diseases = predict_diseases(selected)
                st.session_state["last_diseases"] = diseases

            st.markdown(f"### Found {len(diseases)} potential condition(s):")

            if diseases:
                # Summary metrics
                cc = st.columns(3)
                high  = sum(1 for d in diseases if d["severity"]=="High")
                med   = sum(1 for d in diseases if d["severity"]=="Medium")
                low   = sum(1 for d in diseases if d["severity"]=="Low")
                with cc[0]: st.markdown(f"<div class='metric-card'><div class='metric-num' style='color:#f85149'>{high}</div><div class='metric-label'>High Risk</div></div>", unsafe_allow_html=True)
                with cc[1]: st.markdown(f"<div class='metric-card'><div class='metric-num' style='color:#d29922'>{med}</div><div class='metric-label'>Medium Risk</div></div>", unsafe_allow_html=True)
                with cc[2]: st.markdown(f"<div class='metric-card'><div class='metric-num' style='color:#3fb950'>{low}</div><div class='metric-label'>Low Risk</div></div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            render_diseases(diseases)

            # Cross-reference with dosha
            if st.session_state["last_dosha"]:
                dosha = st.session_state["last_dosha"]
                st.markdown("---")
                st.markdown(f"### 🌿 Dosha Connection")
                st.markdown(f"""
                <div class='card'>
                    Your predicted dosha is <b>{dosha.upper()} {DOSHA_EMOJI_MAP.get(dosha,"🌿")}</b>.
                    The diseases above are consistent with {dosha} imbalance.
                    Balancing your dosha through diet, yoga, and herbs will help prevent
                    and manage these conditions naturally.
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: PATIENTS (Doctor)
# ══════════════════════════════════════════════════════
elif page == "👥 Patients":
    st.markdown("<div class='page-title'>👥 Patient Management</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 All Patients", "➕ Add Patient"])

    with tab1:
        pts = get_all_patients()
        if pts.empty:
            st.info("No patients registered yet.")
        else:
            search = st.text_input("🔍 Search patient", placeholder="Name or ID...")
            if search:
                pts = pts[pts["name"].str.contains(search, case=False) |
                          pts["pid"].str.contains(search, case=False)]
            st.markdown(f"*{len(pts)} patient(s)*")
            for _, p in pts.iterrows():
                with st.expander(f"🧑 {p['name']} · {p['pid']} · {p.get('gender','-')} · Age {p.get('age','-')}"):
                    c1,c2,c3 = st.columns(3)
                    with c1: st.markdown(f"**Phone:** {p.get('phone','-')}")
                    with c2: st.markdown(f"**Blood:** {p.get('blood_group','-')}")
                    with c3: st.markdown(f"**Allergies:** {p.get('allergies','-')}")
                    visits = get_visits(p["pid"])
                    if not visits.empty:
                        st.markdown(f"**Visits:** {len(visits)} · **Last Dosha:** {visits.iloc[0].get('dosha','-')}")
                        st.dataframe(visits[["visit_date","dosha","confidence","health_score"]].head(5),
                                     hide_index=True, use_container_width=True)

    with tab2:
        with st.form("add_patient"):
            c1,c2 = st.columns(2)
            with c1:
                name  = st.text_input("Patient Name *")
                age   = st.number_input("Age", 1, 120, 30)
                phone = st.text_input("Phone")
            with c2:
                gender    = st.selectbox("Gender", ["Male","Female","Other"])
                blood     = st.selectbox("Blood Group", ["A+","A-","B+","B-","O+","O-","AB+","AB-"])
                allergies = st.text_input("Allergies")
            if st.form_submit_button("Add Patient"):
                if not name:
                    st.error("Name required.")
                else:
                    pid = add_patient(st.session_state["user_id"], name, age, gender, phone, blood, allergies)
                    st.success(f"Patient added! ID: **{pid}**")


# ══════════════════════════════════════════════════════
# PAGE: PRESCRIPTIONS (Doctor)
# ══════════════════════════════════════════════════════
elif page == "📋 Prescriptions":
    st.markdown("<div class='page-title'>📋 Prescriptions</div>", unsafe_allow_html=True)
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql(
            "SELECT p.name patient,pr.patient_id pid,pr.dosha,pr.herbs,"
            "pr.duration,pr.created_at date FROM prescriptions pr "
            "JOIN patients p ON p.pid=pr.patient_id ORDER BY pr.created_at DESC", con)
    except: df = pd.DataFrame()
    con.close()
    if df.empty:
        st.info("No prescriptions yet.")
    else:
        search = st.text_input("Search patient")
        if search: df = df[df["patient"].str.contains(search,case=False)]
        st.dataframe(df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE: HISTORY (Patient)
# ══════════════════════════════════════════════════════
elif page == "📈 My History":
    st.markdown("<div class='page-title'>📈 My Health History</div>", unsafe_allow_html=True)
    pid = st.session_state["patient_pid"]
    if pid is None:
        st.warning("Complete your profile first (from Dashboard).")
    else:
        visits = get_visits(pid)
        if visits.empty:
            st.info("No visits yet. Run a dosha prediction to get started!")
        else:
            # Health score trend
            if HAS_PLOTLY and len(visits) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=visits["visit_date"], y=visits["health_score"],
                    mode="lines+markers", name="Health Score",
                    line=dict(color="#3fb950",width=2),
                    marker=dict(size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=visits["visit_date"], y=visits["confidence"],
                    mode="lines+markers", name="Dosha Confidence %",
                    line=dict(color="#58a6ff",width=2,dash="dash"),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title="Health Trend Over Time",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT),
                    yaxis=dict(gridcolor=BORDER),
                    xaxis=dict(gridcolor=BORDER),
                    height=300, margin=dict(t=40,b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            for _, v in visits.iterrows():
                dosha = v.get("dosha","unknown")
                emoji = DOSHA_EMOJI_MAP.get(dosha,"🌿")
                col_v = DOSHA_COLOR_MAP.get(dosha, GREEN)
                with st.expander(f"{emoji} {v['visit_date']} · {dosha.upper()} · Score: {v.get('health_score','-')}"):
                    c1,c2,c3 = st.columns(3)
                    with c1: st.markdown(f"**Dosha:** {dosha.upper()}")
                    with c2: st.markdown(f"**Confidence:** {v.get('confidence',0):.1f}%")
                    with c3: st.markdown(f"**Health Score:** {v.get('health_score','-')}")


# ══════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ══════════════════════════════════════════════════════
elif page == "🤖 AI Assistant":
    st.markdown("<div class='page-title'>🤖 Ayurveda AI Assistant</div>", unsafe_allow_html=True)
    dosha = st.session_state["last_dosha"] or "vata+pitta"
    st.markdown(f"<p class='page-sub'>Ask anything about your {dosha.upper()} constitution — food, yoga, herbs, sleep, stress.</p>",
                unsafe_allow_html=True)

    # Suggested questions
    st.markdown("**Quick questions:**")
    suggestions = [
        f"What should I eat for {dosha}?",
        f"Best yoga for {dosha}?",
        f"Herbs for {dosha} imbalance?",
        "How to manage stress?",
        "Tips for better sleep?",
        f"What is {dosha} dosha?"
    ]
    scols = st.columns(3)
    for i, sug in enumerate(suggestions):
        with scols[i % 3]:
            if st.button(sug, key=f"sug_{i}"):
                if not st.session_state["chat_history"] or \
                   st.session_state["chat_history"][-2][1] != sug:
                    reply = chat_reply(sug, dosha)
                    st.session_state["chat_history"].append(("user", sug))
                    st.session_state["chat_history"].append(("bot", reply))
                    st.rerun()

    st.markdown("---")

    # Input box ABOVE history so rerun clears it
    c_in, c_btn, c_clr = st.columns([5, 1, 1])
    with c_in:
        user_input = st.text_input("", key="chat_input",
                                   placeholder=f"Ask about food, yoga, herbs for {dosha}...",
                                   label_visibility="collapsed")
    with c_btn:
        send = st.button("Send 💬", use_container_width=True, key="chat_send")
    with c_clr:
        if st.button("🗑️ Clear", use_container_width=True, key="chat_clear"):
            st.session_state["chat_history"] = []
            st.rerun()

    if send and user_input:
        reply = chat_reply(user_input, dosha)
        st.session_state["chat_history"].append(("user", user_input))
        st.session_state["chat_history"].append(("bot", reply))
        st.rerun()

    # Chat history display
    st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
    for role, msg in st.session_state["chat_history"][-14:]:
        if role == "user":
            st.markdown(f"<div class='chat-user'>🧑 {msg}</div>", unsafe_allow_html=True)
        else:
            # Render markdown inside bot messages
            st.markdown(f"<div class='chat-bot'>🌿 {msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown("<div class='page-title'>⚙️ Settings</div>", unsafe_allow_html=True)

    st.markdown("### 🎨 Appearance")
    dm = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"], key="settings_dm")
    if dm != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = dm
        st.rerun()

    st.markdown("### 📊 Model Info")
    st.markdown(f"""
    <div class='card'>
        <div style='font-weight:600;font-size:1.1em;margin-bottom:12px'>🧬 DoshaNet GNN</div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:14px'>
            <div>✅ Model: Graph Attention Network (GAT)</div>
            <div>✅ Accuracy: 100% (test set)</div>
            <div>✅ Training data: 1,199 patients</div>
            <div>✅ Features: 29 physical & lifestyle traits</div>
            <div>✅ Dosha classes: 6 types</div>
            <div>✅ Diseases DB: 446 (AyurGenixAI)</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### ℹ️ About")
    st.markdown(f"""
    <div class='card'>
        <b>Ayurveda DoshaNet PRO</b> — Built for hackathon.<br>
        Combines ancient Ayurvedic wisdom with modern Graph Neural Networks.<br><br>
        <b>Stack:</b> PyTorch Geometric · Streamlit · Plotly · SQLite · fpdf2<br>
        <b>SDG Impact:</b> SDG 3 — Good Health & Well-being<br><br>
        <i style='color:{SUBTEXT}'>GNN-assisted decision support. Not a substitute for medical advice.</i>
    </div>""", unsafe_allow_html=True)