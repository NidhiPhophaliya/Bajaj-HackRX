# app/utils/embedder.py

import gdown
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = None

    def download_file(self, url, out_path):
        if not os.path.exists(out_path):
            gdown.download(url, out_path, quiet=False)

    def load_from_files(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def load_from_drive(self, index_url, metadata_url):
        self.download_file(index_url, "faiss_index.idx")
        self.download_file(metadata_url, "metadata.pkl")
        self.load_from_files("faiss_index.idx", "metadata.pkl")

    def query(self, query_text, k=5):
        if self.index is None or self.metadata is None:
            raise ValueError("Index or metadata not loaded")
        
        query_embedding = self.model.encode([query_text]).astype('float32')
        D, I = self.index.search(query_embedding, k)
        results = self.metadata.iloc[I[0]].copy()
        results['score'] = D[0]
        return results
