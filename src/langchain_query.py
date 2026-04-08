from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import *


def load_vectorstore():
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    return db


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        return_full_text=False
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def build_rag_chain(db, llm):
    retriever = db.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that answers questions about a code repository.
        Answer using ONLY the provided context. If unsure, say 'I don't know'. Do not ask questions.
        Do not repeat yourself.
        
        Context: {context}"""),
        ("human", "{input}")
    ])

    docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, docs_chain)
    return qa_chain


def main():
    query = "What's the latest version of this repository?"
    db = load_vectorstore()
    llm = load_llm()
    chain = build_rag_chain(db, llm)

    result = chain.invoke({"input": query})
    print("Answer:")
    print(result["answer"])
    # print("Sources:")
    # for doc in result["context"]:
    #     print(" -", doc.metadata["source"])


if __name__ == "__main__":
    main()
 