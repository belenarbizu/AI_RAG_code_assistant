import streamlit as st
import requests

st.set_page_config(
    page_title="Codebase AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Codebase AI Assistant")
st.caption("Ask questions about the vLLM repository")

# Historial de conversación en session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.caption(f"- {src}")

query = st.chat_input("Ask a question about the repo...")

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/ask",
                    json={"query": query},
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received")

                    st.write(answer)

                    # Intenta obtener fuentes de la clave 'sources' (API normal) 
                    # o extraerlas de 'context'/'source_documents' (LangChain)
                    sources = data.get("sources")
                    if not sources:
                        # Extraemos de context (LangChain). 
                        # Nota: En JSON, doc es un diccionario, no un objeto Document.
                        raw_docs = data.get("context", [])
                        sources = []
                        for doc in raw_docs:
                            meta = doc.get("metadata", {})
                            # Coalescencia: toma la primera que exista
                            path = meta.get("source") or meta.get("file_path")
                            if path:
                                sources.append(path)
                    
                    if sources:
                        with st.expander("📄 Sources"):
                            for src in set(sources):
                                st.caption(f"- {src}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                else:
                    st.error(f"Error {response.status_code}: Could not get a response")

            except requests.exceptions.Timeout:
                st.error("⏱️ The request timed out. The model is taking too long.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Could not connect to the API. Is it running?")
