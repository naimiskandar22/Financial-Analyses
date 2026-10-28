# Load Python packages
import numpy as np
import pandas as pd
import os
import json
# Load financial analysis functions
import support_resistance_levels.detect_sr_levels as sr_levels
import triangle_patterns.detect_triangle_patterns as tri_patterns

import firebase_admin
from firebase_admin import credentials, firestore

# Init Firebase
firebase_creds = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT"))
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_db_collection_ref(collection_name = "COLLECTION_NAME", db_name = "DB_NAME"):
    return db.collection(os.environ.get(collection_name, db_name))

def load_db_collection(collection_name='COLLECTION_NAME', db_name='DB_NAME'):
    # Stream and print all documents from your collection
    docs = get_db_collection_ref(db_name=db_name).stream()
    rows = []
    for doc in docs:
        # print(f"{doc.id} => {doc.to_dict()}")
        data = doc.to_dict()
        data["id"] = doc.id          # keep the document ID (e.g. the date)
        rows.append(data)

    df = pd.DataFrame(rows)
    return df

def UpdateRow(doc_id, collection_path = "", new_data = {}):
    doc_ref = db.collection(collection_path).document(doc_id)
    doc_ref.set(new_data, merge=True)

def AddNewRow(collection_ref, doc_id, new_data = {}):
    data = {
            "Symbol": "NVDA",
            "Date": new_data["Date"],
            "Open": float(new_data["Open"]),
            "High": float(new_data["High"]),
            "Low": float(new_data["Low"]),
            "Close": float(new_data["Close"]),
            "Volume": int(new_data["Volume"]),
            "Support_Level": float(new_data["Current_Support"]),
            "Resistance_Level": float(new_data["Current_Resistance"])
        }
    collection_ref.document(doc_id).set(data)


if __name__ == "__main__":
    # Load the collection from Firestore
    prices_coll_ref = get_db_collection_ref(db_name = "NVDA_prices")
    df = load_db_collection(db_name = 'NVDA_prices')
    # Fetch the latest stored date from Firestore
    last_date = df.iloc[-1]['Date']

    # Set Support and Resistance Levels
    sr_threshold_pct = 0.03
    df = sr_levels.SetSRLevels(df, sr_threshold_pct)

    # Store support and resistance levels in Firestore collection
    # Update existing rows/Save new records
    saved = len(df[df['Date'] > last_date])
    updated = len(df[df['Date'] <= last_date])
    # for timestamp, row in df.iterrows():
    #     # date_str = timestamp.strftime("%Y-%m-%d")
    #     date_str = row['Date'].strftime("%Y-%m-%d")
        
    #     # If it's already in Firestore, update the support & resistance values
    #     if last_date and date_str <= last_date.strftime("%Y-%m-%d"):
    #         UpdateRow(date_str, "NVDA_prices", new_data={"Symbol": "NVDA"})
    #         UpdateRow(date_str, "NVDA_prices", new_data={"Support_Level": row['Current_Support']})
    #         UpdateRow(date_str, "NVDA_prices", new_data={"Resistance_Level": row['Current_Resistance']})
    #         updated += 1
    #         continue  

    #     new_data = {
    #         "Symbol": "NVDA",
    #         "Date": row["Date"],
    #         "Open": float(row["Open"]),
    #         "High": float(row["High"]),
    #         "Low": float(row["Low"]),
    #         "Close": float(row["Close"]),
    #         "Volume": int(row["Volume"]),
    #         "Support_Level": float(new_data["Current_Support"]),
    #         "Resistance_Level": float(new_data["Current_Resistance"])
    #     }
    #     prices_coll_ref.document(date_str).set(new_data)
    #     saved += 1
    for start in range(0, len(df), 500):
        batch = db.batch()
        for row in df.loc[start:start + 500]:
            date_str = row['Date'].strftime("%Y-%m-%d")
            batch.set(prices_coll_ref.document(date_str), row)
        batch.commit()
        
    print(f"Stored {saved} new daily records, and updated {updated} records.")

    print(df)