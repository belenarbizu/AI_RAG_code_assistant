from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import *


def load_vectorstore():
    embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )


def load_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=250)


def build_rag_chain(db, llm):
    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that answers questions about a code repository.
        Rules:
        - Answer using ONLY the provided context.
        - Be concise, maximum 3 sentences.
        - Never suggest checking external sources.
        - If unsure, say 'I don't know'.
        - Answer using 250 words or less.

        Context: {context}"""),
        ("human", "{input}")
    ])

    docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, docs_chain)


def main():
    query = "How do I install vLLM?"
    db = load_vectorstore()
    llm = load_llm()
    chain = build_rag_chain(db, llm)

    result = chain.invoke({"input": query})
    print("Answer:", result["answer"])
    print("Sources:")
    for doc in result["context"]:
        print(" -", doc.metadata["source"])


if __name__ == "__main__":
    main()