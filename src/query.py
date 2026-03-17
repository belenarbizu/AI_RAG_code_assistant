import chromadb
from src.config import *
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(e)
        return None

    return collection


def get_results(query: str, collection, model, k: int = 5) -> list:
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results


def load_llm() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

    return tokenizer, model


def build_context(results: list) -> str:
    documents = results["documents"][0]

    context = "\n\n".join(documents)

    return context


def generate_answer(query: str, context: str, tokenizer, llm) -> str:
    prompt = f"""
You are an expert AI asistant that answers questions about a codebase.

Use ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    # Convert text to numbers
    # return_tensors="pt", return pytorch tensors for the model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Generate new text with max 200 tokens
    outputs = llm.generate(
        **inputs,
        max_new_tokens=200
    )

    # Convert numbers to text
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = decoded.split("Answer:")[-1].strip()

    return answer


def rag_pipeline(query: str) -> str:
    collection = get_collection()
    if collection is None:
        return None

    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer, llm = load_llm()
    results = get_results(query, collection, model)
    context = build_context(results)
    answer = generate_answer(query, context, tokenizer, llm)

    return answer


def main():
    query = "How to run the OpenAI server in vLLM?"
    answer = rag_pipeline(query)
    if answer is not None:
        print(answer)


if __name__ == "__main__":
    main()