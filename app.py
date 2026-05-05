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
    st.error("Please set GOOGLE_API_KEY in Streamlit Secrets")


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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)


# -----------------------------
# 4. CREATE VECTOR STORE
# -----------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# -----------------------------
# 5. QA CHAIN
# -----------------------------
def get_conversational_chain():
    prompt_template = """
You are a helpful assistant for answering questions based only on the given context.

Rules:
- Use ONLY the context below
- If answer is not in context, say: "Answer is not available in the provided context"
- Be clear and structured

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = create_stuff_documents_chain(model, prompt)
    return chain


# -----------------------------
# 6. USER INPUT HANDLER
# -----------------------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists("faiss_index"):
        try:
            db = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return

        docs = db.similarity_search(user_question)

        # Convert docs → text
        context_text = "\n\n".join([doc.page_content for doc in docs])

        chain = get_conversational_chain()

        response = chain.invoke({
            "context": context_text,
            "question": user_question
        })

        st.write("Reply:", response.content)

    else:
        st.error("Please upload and process PDFs first")


# -----------------------------
# 7. UI
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📄")
    st.header("Chat with PDF using Gemini 🤖")

    user_question = st.text_input("Ask a question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)

                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done! Now ask questions.")
                    else:
                        st.error("No text found in PDF")
            else:
                st.error("Upload at least one PDF")


if __name__ == "__main__":
    main()
