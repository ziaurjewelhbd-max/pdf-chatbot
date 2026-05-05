import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


# =========================
# 1. CONFIG + API KEY
# =========================
st.set_page_config(page_title="Pro PDF Chatbot", page_icon="📄")

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ GOOGLE_API_KEY missing in Streamlit Secrets")
    st.stop()


# =========================
# 2. PDF LOADER
# =========================
def load_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


# =========================
# 3. TEXT SPLITTER
# =========================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


# =========================
# 4. VECTOR STORE (FAISS)
# =========================
def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    db = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    db.save_local("faiss_index")


# =========================
# 5. RAG CHAIN
# =========================
def get_chain():
    prompt = PromptTemplate(
        template="""
You are a professional AI assistant.

Rules:
- Answer ONLY from given context
- If not found, say: "Answer is not available in the provided context"
- Be clear, structured, and accurate

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2
    )

    return create_stuff_documents_chain(model, prompt)


# =========================
# 6. QUESTION ENGINE
# =========================
def ask_question(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Please upload and process PDF first")
        return

    try:
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"❌ Vector DB load error: {e}")
        return

    docs = db.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    chain = get_chain()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    st.markdown("### 🤖 Answer")
    st.write(response.content)


# =========================
# 7. UI DESIGN (PRO LEVEL)
# =========================
def main():

    st.title("📄 Pro PDF Chatbot (Gemini AI)")
    st.caption("Ask anything from your PDF using AI-powered search 🔍")

    # Chat input
    question = st.text_input("💬 Ask your question")

    if question:
        ask_question(question)

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload PDF Files")

        pdf_files = st.file_uploader(
            "Upload PDF",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("⚡ Process PDF"):
            if pdf_files:
                with st.spinner("Processing PDF..."):

                    raw_text = load_pdf_text(pdf_files)

                    if not raw_text.strip():
                        st.error("❌ No readable text found in PDF")
                        return

                    chunks = split_text(raw_text)
                    build_vector_store(chunks)

                    st.success("✅ PDF processed successfully!")
            else:
                st.error("⚠️ Please upload at least one PDF")


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
