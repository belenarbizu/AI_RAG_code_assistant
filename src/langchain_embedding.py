from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config import *


def load_files(repo_path: str) -> list:
    md_loader = DirectoryLoader(repo_path, glob="**/*.md")
    py_loader = DirectoryLoader(repo_path, glob="**/*.py")
    
    return md_loader.load(), py_loader.load()


def chunking_files(md_files: list, py_files: list):
    md_splitter = RecursiveCharacterTextSplitter(
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
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=CHROMA_PATH
    )


def main():
    md_files, py_files = load_files(DATA_PATH)
    chunks = chunking_files(md_files, py_files)
    vector_database(chunks)


if __name__ == "__main__":
    main()