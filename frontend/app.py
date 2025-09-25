import streamlit as st
import requests

st.set_page_config(page_title="Multi-Agent AI", page_icon="🤖", layout="centered")

st.title("🤖 Multi-Agent Assistant")
st.write("Ask questions using Wikipedia or run Python calculations!")

# Input
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"query": query}
            )
            if response.status_code == 200:
                st.success(response.json()["answer"])
            else:
                st.error("Something went wrong with the backend.")
