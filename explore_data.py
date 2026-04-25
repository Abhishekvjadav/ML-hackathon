import pandas as pd
import json

# Load Prakriti dataset
with open('prakriti_clean.json', 'r') as f:
    prakriti = pd.DataFrame(json.load(f))

# Load AyurGenix dataset
with open('ayurgenixai_clean.json', 'r') as f:
    ayur = pd.DataFrame(json.load(f))

print("=== PRAKRITI DATASET ===")
print("Shape:", prakriti.shape)
print("Columns:", list(prakriti.columns))
print(prakriti.head(2))

print("\n=== AYURGENIX DATASET ===")
print("Shape:", ayur.shape)
print("Columns:", list(ayur.columns))
print(ayur.head(2))