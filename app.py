import streamlit as st
import os
import json
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import transformers
import torch
from tempfile import NamedTemporaryFile

# Initialize HuggingFace LLM
@st.cache_resource
def initialize_model():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    pipeline = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipeline)

# Embed and Index Documents
def process_and_index_files(files):
    documents = []
    for uploaded_file in files:
        ext = os.path.splitext(uploaded_file.name)[1]
        with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp.name) if ext == ".pdf" else TextLoader(tmp.name)
            documents.extend(loader.load())
            os.remove(tmp.name)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    splits = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(splits, embeddings)

# Streamlit UI
st.title("Enhanced Q&A System")

# File Uploads
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
if uploaded_files:
    st.write("Processing files...")
    vectorstore = process_and_index_files(uploaded_files)
    st.success("Files processed and indexed successfully!")
    
    # LLM Initialization
    llm = initialize_model()
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
    
    # Query Input
    query = st.text_input("Enter your question:")
    if query:
        response = chain({"question": query, "chat_history": []})
        st.write("### Answer:")
        st.write(response['answer'])

        # Optional: Display sources
        if st.checkbox("Show source documents"):
            for doc in response['source_documents']:
                st.write(doc.page_content[:300] + "...")
