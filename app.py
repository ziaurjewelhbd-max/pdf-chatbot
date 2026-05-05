import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# -----------------------------
# 1. API KEY SETUP
# -----------------------------
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("GOOGLE_API_KEY missing in Streamlit Secrets")
    st.stop()


# -----------------------------
# 2. READ PDF
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


# -----------------------------
# 3. SPLIT TEXT
# -----------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


# -----------------------------
# 4. VECTOR STORE (FAISS)
# -----------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# -----------------------------
# 5. QA CHAIN (RAG)
# -----------------------------
def get_conversational_chain():
    prompt = PromptTemplate(
        template="""
You are an AI assistant. Answer ONLY using the given context.

If answer is not in context, say:
"Answer is not available in the provided context"

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
        temperature=0.3
    )

    chain = create_stuff_documents_chain(model, prompt)
    return chain


# -----------------------------
# 6. QUESTION HANDLER
# -----------------------------
def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists("faiss_index"):
        st.error("Please upload and process PDFs first")
        return

    try:
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Vector DB load error: {e}")
        return

    docs = db.similarity_search(question)

    context_text = "\n\n".join([doc.page_content for doc in docs])

    chain = get_conversational_chain()

    response = chain.invoke({
        "context": context_text,
        "question": question
    })

    st.write("🤖 Answer:")
    st.write(response.content)


# -----------------------------
# 7. STREAMLIT UI
# -----------------------------
def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    st.title("📄 Chat with PDF using Gemini AI")

    user_question = st.text_input("Ask anything from your PDF:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("Upload PDF")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Process PDF"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)

                    if raw_text.strip() == "":
                        st.error("No readable text found in PDF")
                        return

                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)

                    st.success("PDF processed successfully!")
            else:
                st.error("Please upload at least one PDF")


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    main()
