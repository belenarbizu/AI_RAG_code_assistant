from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from config import *


def load_vectorstore():
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )


def load_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=250)


def build_rag_chain(db, llm, chunks):
    semantic_retriever = db.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4
    
    ensemble_retriever = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict RAG assistant.

        You MUST follow these rules:

        - Use ONLY the provided context.
        - If the answer is not explicitly in the context, reply EXACTLY: I don't know
        - Do NOT guess, infer, or use prior knowledge.
        - Do NOT redefine terms or expand acronyms.
        - Answer in MAX 200 characters.
        - Answer in MAX 3 sentenceS.
        - Be concise and technical.

        Any violation of these rules is incorrect."""),

            ("human", """Context:
        {context}

        Question:
        {input}

        Answer (max 200 chars):""")
    ])

    docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(ensemble_retriever, docs_chain)


def define_documents(db):
    chunks = db.get()
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(chunks["documents"], chunks["metadatas"])
    ]
    return documents


def main():
    query = "What does the _VllmLogger class do?"
    db = load_vectorstore()
    llm = load_llm()
    documents = define_documents(db)

    chain = build_rag_chain(db, llm, documents)

    result = chain.invoke({"input": query})
    print("Answer:", result["answer"])
    print("Sources:")
    for doc in result["context"]:
        print(" -", doc.metadata["source"])
        with open('content.txt', 'a') as f:
            f.write(doc.page_content)

    docs = db.similarity_search_with_score(query, k=8)
    for doc, score in docs:
        print("\nScore:", score)

if __name__ == "__main__":
    main()