import getpass
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys
import subprocess
import streamlit as st


if not os.environ.get("COHERE_API_KEY"):
  os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")

from langchain_cohere import ChatCohere

llm = ChatCohere(model="command-r-plus")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load documents from a folder
folder_path = "documents"  # Path to your folder with .txt files
file_paths = list(Path(folder_path).rglob("*.txt"))

# Load documents from each .txt file in the folder
all_documents = []
for file_path in file_paths:
    loader = TextLoader(str(file_path), encoding='utf-8')  # Convert Path object to string
    documents = loader.load()
    for doc in documents:
        # Add the file name as the document title in the metadata
        doc.metadata["title"] = file_path.name  # Store the file name as the title
        all_documents.append(doc)  # Add document with metadata

# Ensure there are documents to process
if not all_documents:
    raise ValueError("No documents found in the specified folder.")
else:
    print(f"Loaded {len(all_documents)} documents.")

# Preview the first document's content
#print(all_documents[0].page_content[:500])

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # characters
    chunk_overlap=200,  # characters
    add_start_index=True  # track index in original document
)
all_splits = text_splitter.split_documents(all_documents)
#print(f"Split documents into {len(all_splits)} chunks.")

for split in all_splits:
    if "title" not in split.metadata:
        split.metadata["title"] = "Unknown Title"
        
vector_store = FAISS.from_documents(all_splits, embeddings)
retriever = vector_store.as_retriever()


from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs