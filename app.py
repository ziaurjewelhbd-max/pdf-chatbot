import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ১. Streamlit secrets থেকে API Key লোড করা (এটি অনলাইনে হোস্ট করার জন্য নিরাপদ)
google_api_key = st.secrets["GOOGLE_API_KEY"]

# ২. PDF ফাইল থেকে লেখা (Text) পড়ার ফাংশন
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# ৩. টেক্সটকে ছোট ছোট টুকরো (Chunks) করার ফাংশন
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# ৪. ডাটাবেস তৈরি এবং সেভ করার ফাংশন
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ৫. চ্যাটবটের লজিক এবং প্রম্পট সেটআপ
def get_conversational_chain():
    prompt_template = """
    প্রদত্ত কনটেক্সট (Context) থেকে যতটা সম্ভব বিস্তারিত উত্তর দাও। যদি উত্তরটি ফাইলের ভেতর না থাকে 
    তবে শুধু বলো "দুঃখিত, এই তথ্যটি ফাইলে নেই", ভুল উত্তর বানাবে না।
    সবসময় প্রশ্নের উত্তর বাংলায় দিবে।\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# ৬. ইউজারের ইনপুট প্রসেস করার ফাংশন
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("বট: ", response["output_text"])

# ৭. প্রধান ইউজার ইন্টারফেস (Streamlit Main App)
def main():
    st.set_page_config("Bangla PDF Chatbot")
    st.header("PDF চ্যাটবট (AI Assistant) 💬")

    user_question = st.text_input("আপনার PDF সম্পর্কে বাংলায় প্রশ্ন করুন")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("মেনু ও ফাইল আপলোড")
        pdf_docs = st.file_uploader("আপনার PDF ফাইলটি এখানে আপলোড করুন", accept_multiple_files=True)
        if st.button("ফাইল প্রসেস করুন"):
            with st.spinner("অপেক্ষা করুন, আপনার ফাইলটি পড়া হচ্ছে..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("প্রসেসিং শেষ! এখন আপনি প্রশ্ন করতে পারেন।")

if __name__ == "__main__":
    main()
