from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from src.config import *
import ast


def load_files(repo_path: str) -> list:
    files = []

    # rglob checks files in subdirectories too
    for file in Path(repo_path).rglob('*'):
        if file.suffix == '.md' or file.suffix == '.py':
            files.append(file)

    return files


def chunk_files(files: list) -> list:
    chunks = []

    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                if file.suffix == '.md':
                    file_chunks = chunk_markdown(f.read())
                elif file.suffix == '.py':
                    file_chunks = chunk_python(f.read())
                else:
                    file_chunks = chunk_text(f.read())
                    
                for i, chunk in enumerate(file_chunks):
                    chunks.append({
                        "text": chunk,
                        "metadata": {
                            "file_path": str(file),
                            "chunk_id": i    
                        }
                    })
        except UnicodeDecodeError:
            try:
                with open(file, 'r', encoding='latin-1') as f:
                    if file.suffix == '.md':
                        file_chunks = chunk_markdown(f.read())
                    elif file.suffix == '.py':
                        file_chunks = chunk_python(f.read())
                    else:
                        file_chunks = chunk_text(f.read())
                        
                    for i, chunk in enumerate(file_chunks):
                        chunks.append({
                            "text": chunk,
                            "metadata": {
                                "file_path": str(file),
                                "chunk_id": i    
                            }
                        })
            except:
                continue

    return chunks


def chunk_markdown(text: str, max_chars: int = 1000) -> list:
    chunks = []
    current_title = ""
    current_chunk = ""

    for line in text.splitlines():
        if line.startswith('#'):
            if current_chunk.strip():
                chunks.append(current_title + "\n" + current_chunk)
                current_chunk = ""
            current_title = line
        else:
            current_chunk += line + "\n"

            if len(current_chunk) > max_chars:
                chunks.append(current_title + "\n" + current_chunk)
                current_chunk = current_chunk[-200:]
    

    if current_chunk.strip():
        chunks.append(current_title + "\n" + current_chunk)
    
    return chunks


def chunk_python(text: str, max_chars: int = 1000) -> list:
    chunks = []
    
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_text(text, max_chars)
    
    lines = text.splitlines()
    for node in ast.walk(tree):
        # extract functions and class as independents chunks
        if isinstance(node,(ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno
            text_chunk = '\n'.join(lines[start:end])
            
            if len(text_chunk) <= max_chars:
                chunks.append(text_chunk)
            else:
                for chunk in chunk_text(text_chunk, max_chars):
                    chunks.append(chunk)
    
    return chunks


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list:
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        
        # Intentar cortar en un salto de línea natural
        if end < len(text):
            last_newline = chunk.rfind('\n')
            if last_newline > max_chars // 2:
                end = start + last_newline
                chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap  # overlap para no perder contexto
    
    return chunks


def create_embeddings(chunks: list) -> list:
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    return embeddings


def vector_database(embeddings: list, chunks: list):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"{meta['file_path']}-{meta['chunk_id']}" for meta in metadatas]
    embeddings_list = [e.tolist() for e in embeddings]

    batch_size = 5000
    total = len(chunks)

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)

        collection.add(
            documents=documents[i:end],
            embeddings=embeddings_list[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )


def main():
    files = load_files(DATA_PATH)
    chunks = chunk_files(files)
    embeddings = create_embeddings(chunks)
    vector_database(embeddings, chunks)


if __name__ == "__main__":
    main()
