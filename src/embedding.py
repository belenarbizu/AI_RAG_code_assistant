from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from src.config import *


def load_files(repo_path: str) -> list:
    files = []

    # rglob checks files in subdirectories too
    for file in Path(repo_path).rglob('*'):
        if file.suffix == '.md':
            files.append(file)

    return files


def chunk_files(files: list) -> list:
    chunks = []

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            file_chunks = chunk_text(f.read())
            
            for i, chunk in enumerate(file_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "file_path": str(file),
                        "chunk_id": i    
                    }
                })

    return chunks


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list:
    chunks = []

    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])

    return chunks


def create_embeddings(chunks: list) -> list:
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings


def vector_database(embeddings: list, chunks: list):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"{meta['file_path']}-{meta['chunk_id']}" for meta in metadatas]

    collection.add(
        documents=documents,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=metadatas,
        ids=ids
    )


def main():
    files = load_files(DATA_PATH)
    chunks = chunk_files(files)
    embeddings = create_embeddings(chunks)
    vector_database(embeddings, chunks)


if __name__ == "__main__":
    main()
