import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Load CSV
df = pd.read_csv("app/data/chunks.csv")
texts = df["text"].tolist()
metadata = df[['chunk_id', 'source_doc', 'page', 'text']].reset_index(drop=True)

# Embed
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True)
embedding_matrix = np.array(embeddings).astype('float32')

# Build FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Save everything
faiss.write_index(index, "faiss_index.idx")
np.save("embeddings.npy", embedding_matrix)
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Saved: faiss_index.idx, embeddings.npy, texts.pkl, metadata.pkl")
