# src/features/text_tabular.py
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# simple, explainable keywords; you can add more later after error analysis
KEYWORDS = ["pack","bundle","combo","pcs","ml","g","kg","inch","cm","l","x","set",
            "laptop","phone","case","charger","wireless","bluetooth","original","brand"]

def handcrafted(df: pd.DataFrame) -> np.ndarray:
    s = df["catalog_content"].fillna("")
    lens   = s.str.len().to_numpy()[:,None]
    words  = s.str.split().str.len().to_numpy()[:,None]
    digits = s.str.count(r"\d").to_numpy()[:,None]
    caps   = s.str.count(r"[A-Z]").to_numpy()[:,None]
    feats = [lens, words, digits, caps]
    for kw in KEYWORDS:
        feats.append(s.str.contains(fr"\b{re.escape(kw)}\b", case=False).astype(int).to_numpy()[:,None])
    return np.hstack(feats).astype(np.float32)

class TextFeaturizer:
    def __init__(self, max_features: int = 1500):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1,2),
            max_features=max_features,
            min_df=3
        )

    def fit_transform(self, df: pd.DataFrame):
        X_sparse = self.tfidf.fit_transform(df["catalog_content"].fillna(""))
        X_dense  = handcrafted(df)
        return sparse.hstack([X_sparse, sparse.csr_matrix(X_dense)])

    def transform(self, df: pd.DataFrame):
        X_sparse = self.tfidf.transform(df["catalog_content"].fillna(""))
        X_dense  = handcrafted(df)
        return sparse.hstack([X_sparse, sparse.csr_matrix(X_dense)])