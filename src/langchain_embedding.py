from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import *


def load_files(repo_path: str) -> list:
    md_loader = DirectoryLoader(
        repo_path, 
        glob="**/*.md", 
        loader_cls=TextLoader, 
        show_progress=True, 
        silent_errors=True,
        exclude=["**/.buildkite/**", "**/.github/**", "**/cmake/**", "**/tests/**", "**/tools/**"]
    )
    py_loader = DirectoryLoader(
        repo_path, 
        glob="**/*.py", 
        loader_cls=TextLoader, 
        show_progress=True, 
        silent_errors=True,
        exclude=["**/.buildkite/**", "**/.github/**", "**/cmake/**", "**/tests/**", "**/tools/**"]
    )
    
    md_docs = []
    py_docs = []
    try:
        md_docs = md_loader.load()
    except Exception as e:
        print(f"Error loading markdown files: {e}")
    try:
        py_docs = py_loader.load()
    except Exception as e:
        print(f"Error loading python files: {e}")

    return md_docs, py_docs


def chunking_files(md_files: list, py_files: list):
    md_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    py_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1000,
        chunk_overlap=200
    )

    md_chunks = md_splitter.split_documents(md_files)
    py_chunks = py_splitter.split_documents(py_files)
    chunks = md_chunks + py_chunks

    return chunks


def vector_database(chunks: list):
    print(f"\nIniciando la creación de la base de datos con {len(chunks)} chunks...")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    batch_size = 50 

    db = Chroma.from_documents(
        documents=chunks[:batch_size],
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )

    for i in range(batch_size, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        db.add_documents(chunks[i:end])
        print(f"Progreso: {end}/{len(chunks)} chunks indexados...")

    print("¡Base de datos completada con éxito!")


def main():
    md_files, py_files = load_files(DATA_PATH)
    chunks = chunking_files(md_files, py_files)
    vector_database(chunks)


if __name__ == "__main__":
    main()