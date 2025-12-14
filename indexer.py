# indexer.py
"""
Simple TF-IDF indexer for Brocode AI.
Reads market_data.csv and live_logs/*.csv, builds TF-IDF matrix and saves:
 - docs.pkl (list of document strings)
 - metas.pkl (list of metadata dicts)
 - vectorizer.pkl (sklearn TfidfVectorizer)
 - matrix.npz (sparse TF-IDF matrix)

Run: python indexer.py
"""

import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

CSV_FILE = "market_data.csv"
LIVE_LOG_DIR = "live_logs"
OUT_DIR = "tfidf_index"

os.makedirs(OUT_DIR, exist_ok=True)

def load_rows():
    docs = []
    metas = []
    # main CSV
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE, dtype=str).fillna("")
            for i, row in df.iterrows():
                # stringify row as short text (avoid huge json strings)
                symbol = row.get("symbol") or row.get("SYMBOL") or row.get("Symbol") or ""
                desc = " ".join([f"{k}:{v}" for k,v in row.to_dict().items() if v not in (None,"")])
                docs.append(desc)
                metas.append({"source":"market_csv","index":int(i),"symbol":symbol})
        except Exception as e:
            print("Failed reading main CSV:", e)

    # per-symbol live logs
    if os.path.isdir(LIVE_LOG_DIR):
        for fname in os.listdir(LIVE_LOG_DIR):
            if not fname.lower().endswith(".csv"):
                continue
            path = os.path.join(LIVE_LOG_DIR, fname)
            try:
                sym = os.path.splitext(fname)[0]
                df2 = pd.read_csv(path, dtype=str).fillna("")
                # keep last up to 300 rows to limit size
                df2 = df2.tail(300)
                for i, row in df2.iterrows():
                    text = f"{sym} price {row.get('price','')} qty {row.get('qty','')} time {row.get('timestamp','')}"
                    docs.append(text)
                    metas.append({"source":"live_log","symbol":sym,"row_index":int(i)})
            except Exception as e:
                print("Failed reading", fname, e)
    return docs, metas

def build_and_save():
    print("Loading rows...")
    docs, metas = load_rows()
    if not docs:
        print("No documents found. Aborting.")
        return

    print(f"Building TF-IDF matrix for {len(docs)} docs...")
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
    X = vectorizer.fit_transform(docs)  # sparse matrix

    # save artifacts
    joblib.dump(docs, os.path.join(OUT_DIR, "docs.pkl"))
    joblib.dump(metas, os.path.join(OUT_DIR, "metas.pkl"))
    joblib.dump(vectorizer, os.path.join(OUT_DIR, "vectorizer.pkl"))
    sparse.save_npz(os.path.join(OUT_DIR, "matrix.npz"), X)

    print("Index saved to", OUT_DIR)

if __name__ == "__main__":
    build_and_save()
