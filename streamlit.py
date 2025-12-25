import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title=" Document Q&A",
    layout="centered"
)

st.title(" LLM-Powered Document Q&A System")
st.caption(
    "Upload a PDF and ask questions using FAISS + local LLM (Phi-3 via Ollama)"
)

st.divider()

# =======================
# Upload Section
# =======================
st.header(" Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file}
        )

    if response.status_code == 200:
        st.success("✅ Document processed successfully!")
    else:
        st.error(" Failed to process document.")

st.divider()

# =======================
# Question Section
# =======================
st.header(" Ask a Question")

query = st.text_input(
    "Enter your question",
    placeholder="e.g. What is this document about?"
)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = requests.get(
                f"{API_URL}/ask",
                params={"query": query}
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data.get("answer", "No answer returned."))

            st.subheader(" Citations")
            citations = data.get("citations", [])

            if citations:
                for i, chunk in enumerate(citations, start=1):
                    with st.expander(f"Source {i}"):
                        st.write(chunk)
            else:
                st.info("No citations found.")
        else:
            st.error(" Error while getting the answer.")
