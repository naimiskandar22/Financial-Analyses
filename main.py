# Load Python packages
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
import os
import json
from datetime import date, datetime, timedelta
from datetime import timezone
from dateutil.relativedelta import relativedelta
import yfinance as yf

import firebase_admin
from firebase_admin import credentials, firestore

# Init Firebase
firebase_creds = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT"))
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Set default ticker symbol and historical start date 
ticker_symbol = 'NVDA'
start_date = '2012-01-01'

def LoadPricesFromYF(symbol, start_date):
    stock_df = pd.DataFrame([], columns = [])
        
    # print(f"CSV file of {symbol} prices aren't stored")
    stock_prices = yf.Ticker(symbol).history(start=start_date, auto_adjust=False)
    stock_df['Open'] = stock_prices['Open']
    stock_df['High'] = stock_prices['High']
    stock_df['Low'] = stock_prices['Low']
    stock_df['Close'] = stock_prices['Close']
    stock_df['Adj Close'] = stock_prices['Adj Close']
    stock_df['Volume'] = stock_prices['Volume']
    return stock_df

def LoadExistingFile(df, symbol):
    existing_orders_df = df

    if existing_orders_df is not None:
        existing_orders_df = existing_orders_df.loc[:, ~existing_orders_df.columns.str.contains('^Unnamed')]
        existing_orders_df = existing_orders_df.loc[:, ~existing_orders_df.columns.str.contains('level_0')]
    
    existing_orders_df['Date'] = pd.to_datetime(existing_orders_df['Date'], utc=True)

    stock_df = pd.DataFrame([], columns = [])
    try:
        if (datetime.today() - timedelta(days=1)).date() >= np.max(existing_orders_df['Date']).date():   
            s_date = str(np.max(existing_orders_df['Date']) + timedelta(days=1))[:10]
            
            stock_prices = yf.Ticker(symbol).history(start=str(s_date), auto_adjust=False)
            stock_df['Open'] = stock_prices['Open']
            stock_df['High'] = stock_prices['High']
            stock_df['Low'] = stock_prices['Low']
            stock_df['Close'] = stock_prices['Close']
            stock_df['Adj Close'] = stock_prices['Adj Close']
            stock_df['Volume'] = stock_prices['Volume']

            stock_df['Idx'] = np.repeat(a=[0], repeats=len(stock_df))  
            stock_df['Date'] = pd.to_datetime(stock_df.index, utc=True)
            stock_df['Year'] = stock_df['Date'].apply(lambda x: int(str.split(str(x), '-')[0]))
            #Set the Idx to be the index
            stock_df = stock_df.set_index(pd.Index(stock_df['Idx'].values))
            
            stock_df = pd.concat([existing_orders_df, stock_df], ignore_index=True)
            
            if 'level_0' in stock_df.columns:
                stock_df = stock_df.drop(columns=['level_0'], axis=0)
            stock_df = stock_df.sort_values(by = ['Date']).reset_index()
            if 'index' in list(stock_df.columns):
                stock_df = stock_df.drop(columns=['index'], axis=0)
            
        else:
            stock_df = existing_orders_df
    except:
        stock_df = existing_orders_df
        if 'level_0' in stock_df.columns:
            stock_df = stock_df.drop(columns=['level_0'], axis=0)
        stock_df = stock_df.sort_values(by = ['Date']).reset_index()
        if 'index' in list(stock_df.columns):
            stock_df = stock_df.drop(columns=['index'], axis=0)
        stock_df['Idx'] = np.repeat(a=[0], repeats=len(stock_df))  
        
    if type(stock_df['Date'][0]) != pd._libs.tslibs.timestamps.Timestamp:
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
    
    stock_df = stock_df[~stock_df['Date'].duplicated(keep='first')]
    stock_df = stock_df.sort_values(by = ['Date']).reset_index()

    return stock_df

def load_db_collection(db_path = "test_folder/NVDA_prices"):
    prices_ref = db.collection(db_path)

    return prices_ref

def fetch_and_store_nvda(symbol = 'NVDA'):
    prices_ref = load_db_collection(db_path="test_folder/NVDA_prices")

    # 1. Fetch the latest stored date from Firestore
    latest_docs = prices_ref.order_by("date", direction=firestore.Query.DESCENDING).limit(1).get()
    last_date = latest_docs[0].to_dict().get("date") if latest_docs else None
    df = LoadExistingFile(prices_ref, symbol)

    # Save new records only
    saved = 0
    for timestamp, row in df.iterrows():
        # date_str = timestamp.strftime("%Y-%m-%d")
        date_str = row['Date'].strftime("%Y-%m-%d")
        
        # Skip writing the start date if it's already in Firestore
        if last_date and date_str <= last_date:
            continue  

        data = {
            "symbol": "NVDA",
            "date": date_str,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        }
        prices_ref.document(date_str).set(data)
        saved += 1
        
    print(f"Stored {saved} new daily records.")

if __name__ == "__main__":
    fetch_and_store_nvda()
    # df = load_db_collection()
    # print(df)

    # # Initialize using your local service account JSON key file
    # cred = credentials.Certificate("serviceAccountKey.json")
    # firebase_admin.initialize_app(cred)

    # # Get your Firestore client
    # db = firestore.client()

    # # Stream and print all documents from your collection
    # docs = db.collection("test_folder/NVDA_prices").stream()
    # for doc in docs:
    #     print(f"{doc.id} => {doc.to_dict()}")

