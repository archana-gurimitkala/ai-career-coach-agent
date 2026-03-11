import chromadb
from rag.embedder import embed_text

class VectorStore:
    def __init__(self, collection: str):
        # PersistentClient saves data to disk — survives restarts
        self.client = chromadb.PersistentClient(path="./data/chroma")

        # Store the NAME only — always get fresh collection reference when needed
        # This prevents stale reference errors when collection is deleted/recreated
        self.collection_name = collection

    def _get_collection(self):
        """Always get a fresh collection reference — prevents stale reference errors."""
        return self.client.get_or_create_collection(self.collection_name)

    def add(self, texts: list[str], ids: list[str]):
        """Store text chunks + their vectors in the database."""
        embeddings = [embed_text(t) for t in texts]
        self._get_collection().add(
            embeddings=embeddings,
            documents=texts,
            ids=ids
        )

    def query(self, query_text: str, n: int = 5) -> list[str]:
        """Find the n most relevant chunks for a given query."""
        query_embedding = embed_text(query_text)
        results = self._get_collection().query(
            query_embeddings=[query_embedding],
            n_results=n
        )
        return results["documents"][0]

    def clear(self):
        """Delete all stored chunks — useful when uploading a new resume."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # collection may not exist yet, that's fine
