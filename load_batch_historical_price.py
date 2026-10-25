import os
import json
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# Init Firebase
firebase_creds = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT"))
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()
col = db.collection("NVDA_prices")

df = pd.read_csv("NVDA_prices.csv", parse_dates=["Date"])

# Firestore-friendly values: dates as strings, NaN → None
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
df = df.astype(object).where(df.notna(), None)

records = df.to_dict("records")

for start in range(0, len(records), 500):
    batch = db.batch()
    for row in records[start:start + 500]:
        batch.set(col.document(row["Date"]), row)
    batch.commit()
    print(f"Committed rows {start}–{start + len(records[start:start + 500]) - 1}")