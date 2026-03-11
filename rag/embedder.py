from sentence_transformers import SentenceTransformer

# Load the model once when the file is imported
# all-MiniLM-L6-v2 is small, fast, free, runs locally — no API cost
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> list[float]:
    """Convert any text into a vector (list of numbers)."""
    return _model.encode(text).tolist()
