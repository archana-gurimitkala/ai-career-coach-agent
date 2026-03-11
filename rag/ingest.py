import PyPDF2
from rag.vector_store import VectorStore

def ingest_resume(pdf_path: str):
    """Read a PDF resume, chunk it, and store in ChromaDB."""

    store = VectorStore(collection="resume")

    # Clear old resume data first
    # Important: if user uploads a new resume, old data gets replaced
    store.clear()

    # Step 1: Extract all text from the PDF
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        # Join text from all pages into one big string
        full_text = " ".join(
            page.extract_text() for page in reader.pages
        )

    # Step 2: Split into overlapping chunks of 500 characters
    # Why overlap? So no important info gets cut off at chunk edges
    # Example: chunk 1 = chars 0-500, chunk 2 = chars 400-900 (100 char overlap)
    chunks = []
    chunk_size = 500
    overlap = 100

    for i in range(0, len(full_text), chunk_size - overlap):
        chunk = full_text[i:i + chunk_size]
        if len(chunk) > 50:  # skip tiny leftover chunks
            chunks.append(chunk)

    # Step 3: Give each chunk a unique ID
    ids = [f"resume_chunk_{i}" for i in range(len(chunks))]

    # Step 4: Store in ChromaDB
    store.add(chunks, ids)

    print(f"✅ Resume ingested: {len(chunks)} chunks stored")
    return len(chunks)
