import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = None

    def embed_chunks(self, chunk_csv_path):
        df = pd.read_csv(chunk_csv_path)
        texts = df['text'].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embedding_matrix = np.array(embeddings).astype('float32')

        dimension = embedding_matrix.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embedding_matrix)

        self.metadata = df[['chunk_id', 'source_doc', 'page', 'text']].reset_index(drop=True)
        return self.index

    def query(self, query_text, k=5):
        if self.index is None:
            raise ValueError("Index not built")
        query_embedding = self.model.encode([query_text]).astype('float32')
        D, I = self.index.search(query_embedding, k)
        results = self.metadata.iloc[I[0]].copy()
        results['score'] = D[0]
        return results
