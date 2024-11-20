import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings  # updated imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
os.environ['HF_API_KEY']=os.getenv("HF_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM and Prompt Template
llm = OllamaLLM(model="llama3.2-vision")
prompt = ChatPromptTemplate.from_template(
    '''
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    question: {input}
    '''
)

# Function to create vector embeddings
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # specify the correct model name
        st.session_state.loader = PyPDFDirectoryLoader("paper")  # Data ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit app UI
st.title("RAG Document Q&A With Ollama and Llama3.2-Vision")

user_prompt = st.text_input("Enter your question:")
if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector database ready.") 

# Handling the user prompt and retrieving response
if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({"input": user_prompt})
    st.write(f"Response time: {time.process_time() - start:.2f} seconds")

    st.write(response["answer"])

    # Display document similarity search in an expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------")












