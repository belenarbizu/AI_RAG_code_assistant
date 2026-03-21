# AI_RAG_code_assistant

Built a RAG pipeline to answer questions about large codebases.


| componente | herramienta           |
| ---------- | --------------------- |
| LLM        | Qwen                  |
| Embeddings | sentence-transformers |
| Vector DB  | ChromaDB              |
| Framework  | LangChain             |
| Dataset    | vLLM repository       |

uvicorn src.api:app --reload
streamlit run .\src\app.py