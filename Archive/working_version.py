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
from streamlit.web import cli as stcli
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Check for Cohere API key
if not os.environ.get("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY is not set in the environment variables or .env file.")
else:
    print("Cohere API key loaded successfully!")
    
    # Check for Langchain API key
if not os.environ.get("LANGCHAIN_API_KEY"):
    raise ValueError("LANGCHAIN_API_KEY is not set in the environment variables or .env file.")
else:
    print("Langchain API key loaded successfully!")

# Initialize Cohere LLM
llm = ChatCohere(model="command-r-plus")

# Initialize HuggingFace embeddings
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

# Create FAISS vector store and add documents
vector_store = FAISS.from_documents(all_splits, embeddings)

# Verify the number of documents in the vector store
#print(f"FAISS vector store contains {len(vector_store.index_to_docstore_id)} documents.")

# Example Query ### COMMENT THIS OUT FOR THE INITIAL VERSION WITHOUT SIMILARITY SCORE! from line 88 up to line 100 :)
#query = "Firts, I replaced the box, then, stripping off my little jacket, I disinterred bar after bar of the soap."
#query_embedding = embeddings.embed_query(query)

# Perform similarity search
#search_results = vector_store.similarity_search_by_vector(query_embedding, k=5)

# Display search results ###COMMENT THIS OUT WHEN I WANT TO SEE THE FIRST k RESULTS FOR A QUICK LOOK :) :) 
#for result in search_results:
#    title = result.metadata.get("title")
#    print(f"Title: {title}")
#    print(f"Query: {query}")
#    #print(f"Document content: {result.page_content[:100]}...\n\n") #first 100 characters of the retrieved document
    

def compute_ndcg(relevance_scores, k):
    """
    Compute nDCG@k given a list of relevance scores.
    """
    relevance_scores = relevance_scores[:k]
    # Compute DCG
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    
    # Compute IDCG (ideal DCG)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    
    # Compute nDCG
    return dcg / idcg if idcg > 0 else 0.0

def retrieve_passage_with_ndcg(query, documents, embeddings, top_k=3):
    """
    Retrieve passages and compute nDCG@k for evaluation.
    """
    # Generate query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Perform similarity search
    search_results = vector_store.similarity_search_by_vector(query_embedding, k=top_k)
    
    # Extract passages and compute relevance scores based on cosine similarity
    query_vector = np.array(query_embedding).reshape(1, -1)
    passages = []
    relevance_scores = []
    for result in search_results:
        title = result.metadata.get("title", "Unknown Title")
        start_idx = result.metadata.get("start_index", 0)
        end_idx = start_idx + len(result.page_content)
        passage_vector = np.array(embeddings.embed_query(result.page_content)).reshape(1, -1)
        
        # Compute cosine similarity using sklearn!!!
        similarity_score = cosine_similarity(query_vector, passage_vector)[0][0]
        
        # Relevance score (optional: define thresholds for 0, 1, 2 relevance levels)
        relevance_score = 1 if similarity_score > 0.4 else 0  # Example heuristic
        relevance_scores.append(relevance_score)
        
        # Store passage
        passage = {
            "content": result.page_content,
            "title": title,
            "position": f"{start_idx}-{end_idx}",
            "cosine_similarity": similarity_score,
            "relevance_score": relevance_score,
        }
        passages.append(passage)
    
    # Compute nDCG@k
    ndcg_score = compute_ndcg(relevance_scores, top_k)
    
    return passages, ndcg_score

# Example usage
query = "Zuerst habe ich die Box ersetzt, danach meine Jacke abgenommen."
retrieved_passages, ndcg_score = retrieve_passage_with_ndcg(query, all_splits, embeddings, top_k=3)

# Display retrieved passages and nDCG@k
print(f"Query: {query}\n")
for i, passage in enumerate(retrieved_passages, 1):
    print(f"Passage {i}:\nContent: {passage['content'][:200]}...\n")
    print(f"Title: {passage['title']}")
    print(f"Position: {passage['position']}")
    print(f"Cosine Similarity: {passage['cosine_similarity']:.4f}")
    print(f"Relevance Score: {passage['relevance_score']}\n")

print(f"nDCG@3: {ndcg_score:.4f}")

    
    
    
    
    
    
    
    
########################################    
#retriever = vector_store.as_retriever()

#prompt = hub.pull("rlm/rag-prompt")

#def format_docs(docs):
#  return "\n\n".join(doc.page_content for doc in docs)

#rag_chain = (
#{"context": retriever | format_docs, "question": RunnablePassthrough()}
#| prompt
#| llm
#| StrOutputParser()
#)

#system_prompt = """
#You are an immigration expert for Austria. Answer the user's questions about immigration policies, processes, and requirements with accurate and helpful information.
#"""

# Define the function to run the RAG pipeline with the system prompt
#def run_rag_pipeline_with_system_prompt(question):
    # Combine system prompt, retrieved documents, and user question
#    system_context = system_prompt + "\n\n"
    
#    # Combine the system prompt and the user's question
#    formatted_question = system_context + f"Question: {question}"
    
#    # Invoke the RAG pipeline with the formatted input
#    return rag_chain.invoke(formatted_question)