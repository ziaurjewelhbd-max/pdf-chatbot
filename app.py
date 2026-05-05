import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# ১. API Key সেটআপ (Streamlit Secrets থেকে)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please set the GOOGLE_API_KEY in Streamlit Secrets.")

# ২. PDF ফাইল থেকে টেক্সট পড়ার ফাংশন
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# ৩. টেক্সটকে ছোট ছোট টুকরো (Chunks) করার ফাংশন
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# ৪. টেক্সট টুকরোগুলোকে ভেক্টর স্টোরে রূপান্তর করা
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ৫. চ্যাটবটের জন্য প্রশ্ন-উত্তরের চেইন তৈরি করা
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say "answer is not available in the context", 
    don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # Gemini 1.5 Flash দ্রুত এবং কার্যকরী
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# ৬. ব্যবহারকারীর প্রশ্নের উত্তর দেওয়ার ফাংশন
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # FAISS ইনডেক্স লোড করা
    if os.path.exists("faiss_index"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        st.write("Reply: ", response.get("output_text", "No response generated."))
    else:
        st.error("Vector store not found. Please upload and process a PDF first.")

# ৭. মূল অ্যাপ ইন্টারফেস (Streamlit UI)
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="💁")
    st.header("Chat with multiple PDF using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete! You can now ask questions.")
                    else:
                        st.error("Could not extract text from the uploaded PDFs.")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
