
# RAG Document Q&A With Ollama and Llama3.2-Vision

This project is a Streamlit-based web application that utilizes the Ollama LLM (language model) and Llama3.2-Vision to perform document-based Question and Answering (Q&A). The application takes user queries, processes the input, searches through vectorized embeddings of PDF documents (loaded using PyPDFDirectoryLoader), and retrieves the most relevant information to provide accurate responses. The search is powered by FAISS, which stores and manages the vector embeddings for efficient similarity searches.

# Features

+ Document Embedding
Efficiently vectorizes PDF documents for fast retrieval using HuggingFace embeddings and FAISS.

+ Language Model Integration
Leverage the Ollama LLM (llama3.2-vision) to generate responses based on the provided context from the documents.

+ Real-Time Query Handling
Process user questions and quickly retrieve the most relevant documents from the vector database for accurate responses.

+ Document Similarity Search
After providing an answer, the system allows users to explore similar documents that may help in further understanding the response.

+ Environment Support
 The application is easily deployable with environment variables to securely manage API keys.


# Run Locally
### Prerequisites
Ensure you have the following software installed:

+ Python 3.x (preferably 3.8 or higher)
+ pip (Python package manager)


### Setup Instructions



1. **Clone the Repository**
 First, clone the repository to your local machine:

```bash
  git clone https://github.com/pruthi-saksham/rag-document-qa-ollama-llama.git
cd rag-document-qa-ollama-llama
```


2. **Create a Virtual Environment**
It's recommended to use a virtual environment for managing dependencies:

```bash
  python -m venv venv
```


3. **Activate the Virtual Environment**:
 On Windows:
```bash
 venv\Scripts\activate
```
On macOS/Linux:
```bash
 source venv/bin/activate
```

4. **Install Dependencies**
 Install the required Python libraries via the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

5. **Setup API Keys**
 To use HuggingFace embeddings, you need a HuggingFace API key.

+ Sign up at HuggingFace and generate your API key.
+ Create a `.env` file in the root of the project and add the following line
```bash
  HF_API_KEY=your-huggingface-api-key
```

6. **Run the Application**
 Start the Streamlit application:
```bash
  streamlit run app.py
```


# Environment Variables

To run this project, you will need to add the following environment variables to your `.env` file

`LANGCHAIN_API_KEY` : Your LangChain API key

`HF_API_KEY` :  Your HuggingFace API key

`LANGCHAIN_TRACING_V2` : Set to "True" for LangChain tracing

`LANGCHAIN_PROJECT` : Name of your LangChain project


# Tech Stack

1. **Python**: Core programming language for implementing data processing and logic.
2. **Streamlit**: Framework for building the interactive web application interface.
3. **LangChain**: Utilized for handling the language model interactions, vector stores, and document processing:
  - `OllamaLLM`: A custom language model based on Llama3.2-Vision.
  - `HuggingFaceEmbeddings`: Used for converting documents into vector embeddings.
  - `FAISS`: Efficient similarity search for vectorized document embeddings.
  - `PyPDFDirectoryLoader`: Tool to load PDF documents from a local directory.
  - `RecursiveCharacterTextSplitter`: Used to break down documents into smaller chunks for easier processing.
  - `create_stuff_documents_chain`: Combines the documents with the language model to generate accurate answers.
  - `create_retrieval_chain`: Handles the process of querying the document vector store for relevant information.
4. **Environment Variables**: Used to securely store API keys for third-party services like HuggingFace.



# Usage

### 1. Document Embedding

+ **Upload Documents**: Place your PDF files in a directory (e.g., `paper/`) within the project folder.
+ **Vectorization**: Click the **"Document Embedding"** button to load and vectorize the PDFs into embeddings. This process will parse the documents, break them into chunks, and create the vector database using **FAISS**.
+ **Processing**: After embedding, the system is ready to answer user queries.

### 2. Ask a Question

- **Input Query**: Type your question into the text box provided on the Streamlit interface.
- **Get an Answer**: Upon clicking **"Submit"**, the system will retrieve the most relevant documents, generate an answer using the **Ollama LLM**, and display it on the screen.
- **Response Time**: The time taken to process the response will be displayed. The system is optimized for fast retrieval, ensuring an efficient experience.

### 3. Explore Document Similarity

- **View Similar Documents**: Once the system provides an answer, expand the **"Document Similarity Search"** section to view documents that are most similar to the context of your query.
- **Deep Dive**: Review these documents for additional context or to validate the response.



# Performance and Scalability
- The system can handle large document collections as long as the vectorization and embedding process completes successfully. For large PDFs or collections, you may need to optimize the chunking and text splitting parameters.
- **FAISS** is highly optimized for fast similarity searches, and the app is designed to scale with the size of the document corpus, allowing for efficient document search and retrieval.
# 🚀 About Me
*Hi, I’m Saksham Pruthi, an AI Engineer passionate about creating innovative AI-powered solutions. I specialize in Generative AI, designing systems that bridge cutting-edge research and practical applications. With expertise in various AI frameworks and an eye for scalable technology, I enjoy tackling challenging projects that drive real-world impact.*


## 🛠 Skills
+ **Programming Languages**: Python, C++
+ **Generative AI Technologies**:  Proficient in deploying and fine-tuning a variety of LLMs including Llama2, GPT (OpenAI), Mistral, Gemini Pro  using frameworks like Hugging Face, OpenAI,Groq and Groq. Expertise in NLP tasks like tokenization, sentiment analysis, summarization, and machine translation. Skilled in computer vision (CV) with models for image classification, object detection, and segmentation (YOLO). Expertise in MLOps, building and maintaining pipelines for model training and monitoring. Proficient in conversational AI with platforms LangChain. Skilled in synthetic data generation and code generation
+ **Vector Databases and Embedding Libraries**: Proficient in ChromaDB and FAISS for efficient vector storage, retrieval, and similarity search.
+ **Frameworks, Tools & Libraries**: LangChain, HuggingFace , OpenAI API, Groq, TensorFlow, PyTorch, Streamlit.
+ **Databases**: MongoDB, ChromaDB
+ **Version Control**: Proficient in using Git for version control and GitHub for collaborative development, repository management, and continuous integration workflows.


