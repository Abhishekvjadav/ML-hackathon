import json, pandas as pd

with open('ayurgenixai_clean.json', 'r') as f:
    ayur = pd.DataFrame(json.load(f))

# See what Doshas column actually looks like
print(ayur['Doshas'].dropna().unique()[:15])