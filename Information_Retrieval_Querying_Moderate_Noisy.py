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
from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
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
import time

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
    
        # Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the environment variables or .env file.")
else:
    print("OPENAI_API_KEY key loaded successfully!")

# Initialize Cohere LLM
llm = ChatCohere(model="command-r-plus")

# Initialize HuggingFace embeddings
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
# Initialize Cohere embeddings (using embed-multilingual-v3.0)
#embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


# Load documents from a folder
folder_path = "documents/txt_english"  # Path to your folder with .txt files
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
    chunk_size=300,  # characters
    chunk_overlap=200,  # characters
    add_start_index=True  # track index in original document
)
all_splits = text_splitter.split_documents(all_documents)
#print(f"Split documents into {len(all_splits)} chunks.")

for split in all_splits:
    if "title" not in split.metadata:
        split.metadata["title"] = "Unknown Title"
        
BATCH_SIZE = 30
vectors = []
docs = []

for i in range(0, len(all_splits), BATCH_SIZE):
    batch = all_splits[i:i + BATCH_SIZE]
    texts = [doc.page_content for doc in batch]
    
    # Embed the batch
    batch_vectors = embeddings.embed_documents(texts)  # returns list of vectors
    
    vectors.extend(batch_vectors)
    docs.extend(batch)
    if i % 600 == 0 and i != 0:
        print(f"Processed {i} items...")
        print("Sleeping for 10 seconds...")
        time.sleep(10)

# Build the FAISS store manually with vectors and docs
index = IndexFlatL2(len(vectors[0]))
index.add(np.array(vectors).astype("float32"))

# Storing the docs in the memory
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

# creating indexes
index_to_docstore_id = {i: str(i) for i in range(len(docs))}

# Saving in FAISS vector store
vector_store = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embeddings
)

# Create FAISS vector store and add documents
#vector_store = FAISS.from_documents(all_splits, embeddings)

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
    

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load noisy queries from file
def load_noisy_queries(file_path):
    """ Load noisy queries from a text file, each query on a new line. """
    with open(file_path, "r", encoding="utf-8") as file:
        queries = [line.strip() for line in file.readlines()]
    return queries

noisy_queries_file = "queries/noisy_queries_moderate_german/noisy_queries_moderate_german.txt"  # Adjust path if necessary
noisy_queries = load_noisy_queries(noisy_queries_file)

# Function to compute Hits@K
def compute_hits_at_k(relevance_scores, k):
    """ Check if at least one relevant document appears in the top K results. """
    return 1 if sum(relevance_scores[:k]) > 0 else 0

# Function to compute Mean Reciprocal Rank (MRR)
def compute_mrr(relevance_scores):
    """ Compute MRR@K given a list of relevance scores. """
    for i, score in enumerate(relevance_scores):
        if score > 0:
            return 1 / (i + 1)  # First relevant doc rank
    return 0.0

# Function to compute Precision@K
def compute_precision_at_k(relevance_scores, k):
    """ Compute Precision@K. """
    relevant_count = sum(relevance_scores[:k])
    return relevant_count / k if k > 0 else 0.0

# Function to retrieve passages and compute metrics
def retrieve_passage_with_metrics(query, documents, embeddings, top_k=3):
    """ Retrieve passages and compute Hits@K, MRR, Precision@K. """
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
        
        # Compute cosine similarity
        similarity_score = cosine_similarity(query_vector, passage_vector)[0][0]
        
        # Define relevance score (heuristic)
        relevance_score = 1 if similarity_score > 0.6 else 0
        relevance_scores.append(relevance_score)
        
        # Store passage details
        passage = {
            "content": result.page_content,
            "title": title,
            "position": f"{start_idx}-{end_idx}",
            "cosine_similarity": similarity_score,
            "relevance_score": relevance_score,
        }
        passages.append(passage)

    # Compute evaluation metrics
    hits_at_k = compute_hits_at_k(relevance_scores, top_k)
    mrr = compute_mrr(relevance_scores)
    precision_at_k = compute_precision_at_k(relevance_scores, top_k)
   
    
    return passages, hits_at_k, mrr, precision_at_k, similarity_score

# Create a list to store results for CSV
retrieval_data = []

# Process all noisy queries and compute retrieval scores
for i, query in enumerate(noisy_queries, 1):
    print(f"\nProcessing Query {i}: {query}\n" + "-" * 60)

    # Retrieve passages and compute metrics
    retrieved_passages, hits_at_k, mrr, precision_at_k, _ = retrieve_passage_with_metrics(
        query, all_splits, embeddings, top_k=3
    )

    # Store query and evaluation metrics
    for j, passage in enumerate(retrieved_passages, 1):
        retrieval_data.append({
            "Query ID": i,
            "Query": query,
            "Retrieved Passage ID": j,
            "Document Title": passage['title'],
            "Position": passage['position'],
            "Cosine Similarity": passage['cosine_similarity'],
            "Relevance Score": passage['relevance_score'],
            "Hits@3": hits_at_k if j == 1 else "",  # Only store once per query
            "MRR@3": mrr if j == 1 else "",
            "Precision@3": precision_at_k if j == 1 else "",
        })

    print(f"Hits@3 for Query {i}: {hits_at_k}")
    print(f"MRR@3 for Query {i}: {mrr:.4f}")
    print(f"Precision@3 for Query {i}: {precision_at_k:.4f}\n")
    print(f"Cosine Similarity for Query {i}: {retrieval_data[-1]['Cosine Similarity']:.4f}\n")

    print("=" * 80)

# Convert results to a Pandas DataFrame
df = pd.DataFrame(retrieval_data)

# Save to CSV inside "results" folder
output_folder = "results_0.6_temperature/results_openai/crosslingual/moderate_errors"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

output_csv = os.path.join(output_folder, "german_english_noisy_queries_clean_documents_moderate_results.csv")
df.to_csv(output_csv, index=False, encoding="utf-8-sig", sep=";")

print(f"\n Retrieval results saved to {output_csv} successfully!")



#############################################################


# Load documents from a folder
folder_path = "documents/noisy_documents_moderate_english"  # Path to your folder with .txt files
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
    chunk_size=300,  # characters
    chunk_overlap=200,  # characters
    add_start_index=True  # track index in original document
)
all_splits = text_splitter.split_documents(all_documents)
#print(f"Split documents into {len(all_splits)} chunks.")

for split in all_splits:
    if "title" not in split.metadata:
        split.metadata["title"] = "Unknown Title"
        
BATCH_SIZE = 30
vectors = []
docs = []

for i in range(0, len(all_splits), BATCH_SIZE):
    batch = all_splits[i:i + BATCH_SIZE]
    texts = [doc.page_content for doc in batch]
    
    # Embed the batch
    batch_vectors = embeddings.embed_documents(texts)  # returns list of vectors
    
    vectors.extend(batch_vectors)
    docs.extend(batch)
    if i % 600 == 0 and i != 0:
        print(f"Processed {i} items...")
        print("Sleeping for 10 seconds...")
        time.sleep(10)

# Build the FAISS store manually with vectors and docs
index = IndexFlatL2(len(vectors[0]))
index.add(np.array(vectors).astype("float32"))

# Storing the docs in the memory
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

# creating indexes
index_to_docstore_id = {i: str(i) for i in range(len(docs))}

# Saving in FAISS vector store
vector_store = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embeddings
)

# Create FAISS vector store and add documents
#vector_store = FAISS.from_documents(all_splits, embeddings)

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
    

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load noisy queries from file
def load_noisy_queries(file_path):
    """ Load noisy queries from a text file, each query on a new line. """
    with open(file_path, "r", encoding="utf-8") as file:
        queries = [line.strip() for line in file.readlines()]
    return queries

noisy_queries_file = "queries/queries_german/questions_german.txt"  # Adjust path if necessary
noisy_queries = load_noisy_queries(noisy_queries_file)

# Function to compute Hits@K
def compute_hits_at_k(relevance_scores, k):
    """ Check if at least one relevant document appears in the top K results. """
    return 1 if sum(relevance_scores[:k]) > 0 else 0

# Function to compute Mean Reciprocal Rank (MRR)
def compute_mrr(relevance_scores):
    """ Compute MRR@K given a list of relevance scores. """
    for i, score in enumerate(relevance_scores):
        if score > 0:
            return 1 / (i + 1)  # First relevant doc rank
    return 0.0

# Function to compute Precision@K
def compute_precision_at_k(relevance_scores, k):
    """ Compute Precision@K. """
    relevant_count = sum(relevance_scores[:k])
    return relevant_count / k if k > 0 else 0.0

# Function to retrieve passages and compute metrics
def retrieve_passage_with_metrics(query, documents, embeddings, top_k=3):
    """ Retrieve passages and compute Hits@K, MRR, Precision@K. """
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
        
        # Compute cosine similarity
        similarity_score = cosine_similarity(query_vector, passage_vector)[0][0]
        
        # Define relevance score (heuristic)
        relevance_score = 1 if similarity_score > 0.6 else 0
        relevance_scores.append(relevance_score)
        
        # Store passage details
        passage = {
            "content": result.page_content,
            "title": title,
            "position": f"{start_idx}-{end_idx}",
            "cosine_similarity": similarity_score,
            "relevance_score": relevance_score,
        }
        passages.append(passage)

    # Compute evaluation metrics
    hits_at_k = compute_hits_at_k(relevance_scores, top_k)
    mrr = compute_mrr(relevance_scores)
    precision_at_k = compute_precision_at_k(relevance_scores, top_k)
   
    
    return passages, hits_at_k, mrr, precision_at_k, similarity_score

# Create a list to store results for CSV
retrieval_data = []

# Process all noisy queries and compute retrieval scores
for i, query in enumerate(noisy_queries, 1):
    print(f"\nProcessing Query {i}: {query}\n" + "-" * 60)

    # Retrieve passages and compute metrics
    retrieved_passages, hits_at_k, mrr, precision_at_k, _ = retrieve_passage_with_metrics(
        query, all_splits, embeddings, top_k=3
    )

    # Store query and evaluation metrics
    for j, passage in enumerate(retrieved_passages, 1):
        retrieval_data.append({
            "Query ID": i,
            "Query": query,
            "Retrieved Passage ID": j,
            "Document Title": passage['title'],
            "Position": passage['position'],
            "Cosine Similarity": passage['cosine_similarity'],
            "Relevance Score": passage['relevance_score'],
            "Hits@3": hits_at_k if j == 1 else "",  # Only store once per query
            "MRR@3": mrr if j == 1 else "",
            "Precision@3": precision_at_k if j == 1 else "",
        })

    print(f"Hits@3 for Query {i}: {hits_at_k}")
    print(f"MRR@3 for Query {i}: {mrr:.4f}")
    print(f"Precision@3 for Query {i}: {precision_at_k:.4f}\n")
    print(f"Cosine Similarity for Query {i}: {retrieval_data[-1]['Cosine Similarity']:.4f}\n")

    print("=" * 80)

# Convert results to a Pandas DataFrame
df = pd.DataFrame(retrieval_data)

# Save to CSV inside "results" folder
output_folder = "results_0.6_temperature/results_openai/crosslingual/moderate_errors"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

output_csv = os.path.join(output_folder, "german_english_clean_queries_noisy_documents_moderate_results.csv")
df.to_csv(output_csv, index=False, encoding="utf-8-sig", sep=";")

print(f"\n Retrieval results saved to {output_csv} successfully!")


#############################################################


# Load documents from a folder
folder_path = "documents/noisy_documents_moderate_english"  # Path to your folder with .txt files
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
    chunk_size=300,  # characters
    chunk_overlap=200,  # characters
    add_start_index=True  # track index in original document
)
all_splits = text_splitter.split_documents(all_documents)
#print(f"Split documents into {len(all_splits)} chunks.")

for split in all_splits:
    if "title" not in split.metadata:
        split.metadata["title"] = "Unknown Title"
        
BATCH_SIZE = 30
vectors = []
docs = []

for i in range(0, len(all_splits), BATCH_SIZE):
    batch = all_splits[i:i + BATCH_SIZE]
    texts = [doc.page_content for doc in batch]
    
    # Embed the batch
    batch_vectors = embeddings.embed_documents(texts)  # returns list of vectors
    
    vectors.extend(batch_vectors)
    docs.extend(batch)
    if i % 600 == 0 and i != 0:
        print(f"Processed {i} items...")
        print("Sleeping for 10 seconds...")
        time.sleep(10)

# Build the FAISS store manually with vectors and docs
index = IndexFlatL2(len(vectors[0]))
index.add(np.array(vectors).astype("float32"))

# Storing the docs in the memory
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

# creating indexes
index_to_docstore_id = {i: str(i) for i in range(len(docs))}

# Saving in FAISS vector store
vector_store = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embeddings
)

# Create FAISS vector store and add documents
#vector_store = FAISS.from_documents(all_splits, embeddings)

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
    

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load noisy queries from file
def load_noisy_queries(file_path):
    """ Load noisy queries from a text file, each query on a new line. """
    with open(file_path, "r", encoding="utf-8") as file:
        queries = [line.strip() for line in file.readlines()]
    return queries

noisy_queries_file = "queries/noisy_queries_moderate_german/noisy_queries_moderate_german.txt"  # Adjust path if necessary
noisy_queries = load_noisy_queries(noisy_queries_file)

# Function to compute Hits@K
def compute_hits_at_k(relevance_scores, k):
    """ Check if at least one relevant document appears in the top K results. """
    return 1 if sum(relevance_scores[:k]) > 0 else 0

# Function to compute Mean Reciprocal Rank (MRR)
def compute_mrr(relevance_scores):
    """ Compute MRR@K given a list of relevance scores. """
    for i, score in enumerate(relevance_scores):
        if score > 0:
            return 1 / (i + 1)  # First relevant doc rank
    return 0.0

# Function to compute Precision@K
def compute_precision_at_k(relevance_scores, k):
    """ Compute Precision@K. """
    relevant_count = sum(relevance_scores[:k])
    return relevant_count / k if k > 0 else 0.0

# Function to retrieve passages and compute metrics
def retrieve_passage_with_metrics(query, documents, embeddings, top_k=3):
    """ Retrieve passages and compute Hits@K, MRR, Precision@K. """
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
        
        # Compute cosine similarity
        similarity_score = cosine_similarity(query_vector, passage_vector)[0][0]
        
        # Define relevance score (heuristic)
        relevance_score = 1 if similarity_score > 0.6 else 0
        relevance_scores.append(relevance_score)
        
        # Store passage details
        passage = {
            "content": result.page_content,
            "title": title,
            "position": f"{start_idx}-{end_idx}",
            "cosine_similarity": similarity_score,
            "relevance_score": relevance_score,
        }
        passages.append(passage)

    # Compute evaluation metrics
    hits_at_k = compute_hits_at_k(relevance_scores, top_k)
    mrr = compute_mrr(relevance_scores)
    precision_at_k = compute_precision_at_k(relevance_scores, top_k)
   
    
    return passages, hits_at_k, mrr, precision_at_k, similarity_score

# Create a list to store results for CSV
retrieval_data = []

# Process all noisy queries and compute retrieval scores
for i, query in enumerate(noisy_queries, 1):
    print(f"\nProcessing Query {i}: {query}\n" + "-" * 60)

    # Retrieve passages and compute metrics
    retrieved_passages, hits_at_k, mrr, precision_at_k, _ = retrieve_passage_with_metrics(
        query, all_splits, embeddings, top_k=3
    )

    # Store query and evaluation metrics
    for j, passage in enumerate(retrieved_passages, 1):
        retrieval_data.append({
            "Query ID": i,
            "Query": query,
            "Retrieved Passage ID": j,
            "Document Title": passage['title'],
            "Position": passage['position'],
            "Cosine Similarity": passage['cosine_similarity'],
            "Relevance Score": passage['relevance_score'],
            "Hits@3": hits_at_k if j == 1 else "",  # Only store once per query
            "MRR@3": mrr if j == 1 else "",
            "Precision@3": precision_at_k if j == 1 else "",
        })

    print(f"Hits@3 for Query {i}: {hits_at_k}")
    print(f"MRR@3 for Query {i}: {mrr:.4f}")
    print(f"Precision@3 for Query {i}: {precision_at_k:.4f}\n")
    print(f"Cosine Similarity for Query {i}: {retrieval_data[-1]['Cosine Similarity']:.4f}\n")

    print("=" * 80)

# Convert results to a Pandas DataFrame
df = pd.DataFrame(retrieval_data)

# Save to CSV inside "results" folder
output_folder = "results_0.6_temperature/results_openai/crosslingual/moderate_errors"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

output_csv = os.path.join(output_folder, "german_english_noisy_queries_noisy_documents_moderate_results.csv")
df.to_csv(output_csv, index=False, encoding="utf-8-sig", sep=";")

print(f"\n Retrieval results saved to {output_csv} successfully!")