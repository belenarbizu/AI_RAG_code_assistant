import chromadb
from src.config import *
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import bm25s


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(e)
        return None

    return collection


def load_llm() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

    return tokenizer, model


def build_context(results: list) -> str:
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(documents)

    sources = list(set([meta["file_path"] for meta in metadatas]))

    return context, sources


def generate_answer(query, context, tokenizer, llm):
    messages = [
        {"role": "system", "content": "You are an assistant that answers questions about a code repository. Answer using ONLY the provided context. If unsure, say 'I don't know'."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    # Aplicar el chat template del modelo
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    outputs = llm.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=True)
    decoded = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return decoded.strip()


def build_bm25_index(documents: list):
    """documents es una lista de strings (los textos de los chunks)"""
    tokens = bm25s.tokenize(documents)
    bm25 = bm25s.BM25()
    bm25.index(tokens)
    return bm25


def search_bm25(query: str, bm25: bm25s.BM25, documents: list, k: int = 4):
    query_tokens = bm25s.tokenize([query])
    results, _ = bm25.retrieve(query_tokens, k=k)
    # results[0] contiene los índices — los usamos para recuperar los documentos
    return [(i, documents[i]) for i in results[0]]


def hybrid_search(query: str, bm25, collection, model, documents: list, k: int = 4):
    bm25_results = search_bm25(query, bm25, documents, k=k*2)

    query_embedding = model.encode([query]).tolist()
    semantic_results = collection.query(
        query_embeddings=query_embedding,
        n_results=k*2,
        include=["documents", "metadatas", "embeddings"]
    )

    scores = {}

    # Puntuar BM25 — usamos el índice como ID temporal
    for rank, (idx, _) in enumerate(bm25_results):
        scores[f"bm25_{idx}"] = scores.get(f"bm25_{idx}", 0) + 1 / (rank + 1)

    # Puntuar semántico — usamos el ID de ChromaDB
    for rank, doc_id in enumerate(semantic_results["ids"][0]):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + 1)

    # Los mejores IDs semánticos tras fusión
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    top_chroma_ids = [id for id in sorted_ids if not id.startswith("bm25_")][:k]

    # Recuperar documentos y metadatos de ChromaDB por IDs
    final_results = collection.get(
        ids=top_chroma_ids,
        include=["documents", "metadatas"]
    )

    # Formatear igual que collection.query para que build_context funcione
    return {
        "documents": [final_results["documents"]],
        "metadatas": [final_results["metadatas"]]
    }


def rag_pipeline(query: str, collection, bm25, documents: list, model, tokenizer, llm) -> str:
    results = hybrid_search(query, bm25, collection, model, documents)
    context, sources = build_context(results)
    answer = generate_answer(query, context, tokenizer, llm)
    return {"answer": answer, "sources": sources} 


def main():
    query = "What's the latest version of this repository?"
    collection = get_collection()
    if collection is None:
        return None
    
    data = collection.get(include=["documents"])
    documents = data["documents"]
    bm25 = build_bm25_index(documents)

    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer, llm = load_llm()

    answer = rag_pipeline(query, collection, bm25, documents, model, tokenizer, llm)
    if answer is not None:
        print("Answer:")
        print(answer['answer'])
        print("Sources:")
        print(answer['sources'])


if __name__ == "__main__":
    main()