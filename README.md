# AI_RAG_code_assistant

Built a RAG pipeline to answer questions about large codebases.


| componente | herramienta           |
| ---------- | --------------------- |
| Embeddings | sentence-transformers |
| Vector DB  | ChromaDB              |
| Retriever  | Hybrid search         |
| LLM        | Qwen                  |
| Dataset    | vLLM repository       |


## Langchain

| componente | herramienta           |
| ---------- | --------------------- |
| Embeddings | HuggingFaceEmbeddings |
| Vector DB  | ChromaDB              |
| Retriever  | Hybrid search         |
| LLM        | Ollama                |
| Dataset    | vLLM repository       |


uvicorn src.api:app --reload
streamlit run .\src\app.py

Langchain:
install ollama
ollama pull mistral