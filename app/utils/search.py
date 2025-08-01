# app/utils/search.py

class SemanticSearch:
    def __init__(self, embedder):
        self.embedder = embedder

    def search(self, query_text, top_k=5):
        return self.embedder.query(query_text, k=top_k)
