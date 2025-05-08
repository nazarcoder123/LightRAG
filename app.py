import os
import streamlit as st
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio
import nest_asyncio
import fitz  # PyMuPDF for PDF text extraction

# Solve event loop issues in Streamlit
nest_asyncio.apply()
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
WORKING_DIR = "./uploaded_docs"

# --- Gemini LLM function ---
async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    client = genai.Client(api_key=gemini_api_key)

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"
    for msg in history_messages or []:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"
    combined_prompt += f"user: {prompt}"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(max_output_tokens=5000, temperature=0.7),
    )
    return response.text


# --- Embedding function ---
async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


# --- RAG Initialization ---
@st.cache_resource(show_spinner=False)
def get_rag_instance():
    if os.path.exists(WORKING_DIR):
        import shutil
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)

    async def setup():
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=embedding_func,
            ),
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
        return rag

    return asyncio.run(setup())


# --- Extract text from PDF ---
def extract_text_from_pdf(pdf_file) -> str:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Gemini + LightRAG QA", layout="centered")
    st.title("📚 Ask Questions About Your Document (Gemini + LightRAG)")

    rag = get_rag_instance()

    uploaded_file = st.file_uploader("Upload a `.txt` or `.pdf` file", type=["txt", "pdf"])

    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        if file_ext == "txt":
            text = uploaded_file.read().decode("utf-8")
        elif file_ext == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        rag.insert(text)
        st.success("✅ Document uploaded and indexed!")

        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Thinking..."):
                response = rag.query(
                    query=query,
                    param=QueryParam(mode="mix", top_k=5, response_type="paragraph"),
                )
            st.markdown("### 📌 Answer")
            cleaned_response = response.split("References:")[0].strip()
            st.write(cleaned_response)



if __name__ == "__main__":
    main()
