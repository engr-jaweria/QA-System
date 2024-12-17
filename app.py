import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.llms import HuggingFacePipeline
import transformers
import torch

# Model and tokenizer initialization
def initialize_model():
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token='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'
    )
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token='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')
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1
    )
    return HuggingFacePipeline(pipeline=generate_text)

# Initialize the model and embeddings
llm = initialize_model()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cuda"})

# Streamlit UI
st.title("Document-Based Question Answering Web App")

# File uploader and web link input
uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])
web_link = st.text_input("Enter a web link to load content:")

documents = []
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(uploaded_file)
    elif uploaded_file.type == "text/plain":
        loader = TextLoader(uploaded_file)
    documents = loader.load()

if web_link:
    st.write("Loading content from web link...")
    try:
        loader = WebBaseLoader(web_link)
        documents.extend(loader.load())
        st.success("Web content loaded successfully!")
    except Exception as e:
        st.error(f"Error loading web link: {e}")

if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    # Create vector store from documents
    vectorstore = FAISS.from_documents(all_splits, embeddings)

    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

    st.write("### Ask your question below:")
    user_question = st.text_input("Your question:")

    if user_question:
        result = chain({"question": user_question, "chat_history": []})
        st.write("### Answer:")
        st.write(result['answer'])

        # Optionally show source documents
        if st.checkbox("Show source documents"):
            for doc in result['source_documents']:
                st.write(doc.page_content)
