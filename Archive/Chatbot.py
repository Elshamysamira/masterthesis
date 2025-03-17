import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chainlit as cl

# Load environment variables
load_dotenv()

# Check for API keys
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY is not set in the environment variables.")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if not langchain_api_key:
    raise ValueError("LANGCHAIN_API_KEY is not set in the environment variables.")

# Initialize LLM and embeddings
llm = ChatCohere(model="command-r-plus", cohere_api_key=cohere_api_key)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents from a folder
folder_path = "documents"
file_paths = list(Path(folder_path).rglob("*.txt"))

# Load documents
all_documents = []
for file_path in file_paths:
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()
    for doc in documents:
        doc.metadata["title"] = file_path.name
        all_documents.append(doc)

# Ensure documents are available
if not all_documents:
    raise ValueError("No documents found in the specified folder.")

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(all_documents)

for split in all_splits:
    if "title" not in split.metadata:
        split.metadata["title"] = "Unknown Title"

# Create FAISS vector store
vector_store = FAISS.from_documents(all_splits, embeddings)
retriever = vector_store.as_retriever()

# Load the RAG prompt
prompt = hub.pull("rlm/rag-prompt")

# Define RAG pipeline
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chainlit functionality
@cl.on_message
async def on_message(message: cl.Message):
    """Handle user input and run the RAG pipeline."""
    try:
        # Process the query using the RAG pipeline
        user_query = message.content
        response = rag_chain.invoke(user_query)

        # Send the RAG pipeline response back to the user
        await cl.Message(content=f"**Answer:** {response}").send()
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()


@cl.on_chat_start
def on_chat_start():
    """Initialize session for conversation history."""
    cl.user_session.set("message_history", [])
