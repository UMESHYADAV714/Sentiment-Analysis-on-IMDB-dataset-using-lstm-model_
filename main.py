"""
save_token.py
─────────────
Run this ONCE before starting main.py.

It recreates the exact same tokenizer (`token`) that was built during
training in your notebook, and saves it to token.pkl.

Your existing config.pkl was saved from a DIFFERENT tokenizer:
    Tokenizer(num_words=5000).fit_on_texts(x_data)   ← wrong
The correct one is:
    Tokenizer(lower=False).fit_on_texts(x_train)     ← this script

Usage:
    python save_token.py
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

# ── Config ─────────────────────────────────────────────────────────────
CSV_PATH   =  "D:\\desktop\\6sem project\\IMDB Dataset.csv"  # change path if your CSV is elsewhere"
OUTPUT_PKL = "token.pkl"
# ───────────────────────────────────────────────────────────────────────

english_stops = set(stopwords.words("english"))

def load_dataset(path):
    df     = pd.read_csv(path)
    x_data = df["review"]
    y_data = df["sentiment"]

    # ── Same preprocessing as notebook load_dataset() ──
    x_data = x_data.replace({"<.*?>": ""}, regex=True)           # strip HTML
    x_data = x_data.replace({"[^A-Za-z]": " "}, regex=True)      # keep letters
    x_data = x_data.apply(lambda r: [w for w in r.split()        # remove stopwords
                                      if w not in english_stops])
    x_data = x_data.apply(lambda r: [w.lower() for w in r])      # lowercase

    y_data = y_data.replace("positive", 1).replace("negative", 0)
    return x_data, y_data

print(f"Reading dataset from: {CSV_PATH}")
x_data, y_data = load_dataset(CSV_PATH)

# Same split as notebook (no random_state → use same default behaviour)
x_train, _, _, _ = train_test_split(x_data, y_data, test_size=0.2)

# ── Build tokenizer exactly like notebook cell 13 ──
print("Fitting tokenizer on x_train ...")
token = Tokenizer(lower=False)   # lower=False because data is already lowercase
token.fit_on_texts(x_train)

# Save
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(token, f)

print(f"✅  Saved correct tokenizer to  {OUTPUT_PKL}")
print(f"    Vocabulary size: {len(token.word_index)} words")