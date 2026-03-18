import streamlit as st
import requests

st.title("Codebase AI Assistant")

query = st.text_input("Ask a question about the repo:")

if st.button("Ask"):
    response = requests.post("http://localhost:8000/ask", json={"query": query})

    if response.status_code == 200:
        data = response.json()
        st.subheader("Answer:")
        st.write(data["answer"])
        st.divider()
        st.write("Sources:")
        for src in data["sources"]:
            st.markdown(f"- {src}")
    else:
        st.error("Something went wrong")