from app.utils.embedder import Embedder

class SemanticSearch:
    def __init__(self, chunk_csv_path):
        self.embedder = Embedder()
        self.embedder.embed_chunks(chunk_csv_path)

    def search(self, query_text, top_k=5):
        return self.embedder.query(query_text, k=top_k)
