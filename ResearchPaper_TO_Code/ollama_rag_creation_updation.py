import os
import hashlib
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import re
from data_cleaning import clean_text, normalize_text, remove_references
from get_ollama_embedding_function import get_ollama_mxbai_embed_large_embedding_function
import config as cfg



def get_file_hash(filepath: str) -> str:
    """
    Generate a unique MD5 hash for a file based on its binary contents.

    This function reads the entire file specified by the given filepath in binary mode,
    computes its MD5 hash, and returns the hexadecimal digest as a string.

    Args:
        filepath (str): The path to the file for which the hash is to be generated.

    Returns:
        str: A hexadecimal string representing the MD5 hash of the file contents.
    """
    # Create a new MD5 hash object to accumulate file data for hashing
    hasher = hashlib.md5()

    # Open the file in binary read mode to access raw bytes
    with open(filepath, "rb") as f:
        # Read the full content of the file into memory as bytes
        buf = f.read()
        # Update the hash object with the file bytes to compute hash
        hasher.update(buf)

    # Compute the final hex digest string of the hash and return it
    return hasher.hexdigest()

def load_existing_metadata(persist_dir: str) -> Dict[str, str]:
    """
    Load existing file hashes from Chroma metadata stored in the specified directory.

    This function checks if the given persistence directory exists. If it does, it loads the Chroma database,
    retrieves metadata for all documents, and extracts a mapping from each file's basename to its hash value.

    Args:
        persist_dir (str): The directory containing Chroma's persisted data.

    Returns:
        Dict[str, str]: A dictionary mapping file basenames to their hash values.
    """
    # Check if the specified directory exists; if not, return an empty dictionary
    if not os.path.exists(persist_dir):
        return {}

    # Initialize the Chroma database object with persistent storage and an embedding function
    
    db = Chroma(
        persist_directory = persist_dir,
        embedding_function = get_ollama_mxbai_embed_large_embedding_function()
    )

    # Retrieve all document metadata from the database
    docs = db.get(include=["metadatas"])

    existing = {}  # Initialize an empty dictionary to store existing file metadata

    # Iterate over each metadata dictionary in the retrieved documents
    for meta in docs["metadatas"]:
        # Check if both 'source' (file path) and 'hash' (file hash) keys are present in metadata
        if "source" in meta and "hash" in meta:
            # Use the file's basename as the dictionary key and its hash as the value
            existing[os.path.basename(meta["source"])] = meta["hash"]

    # Return the dictionary mapping file basenames to hashes
    return existing

def get_new_pdfs(pdf_folder: str, existing_hashes: Dict[str, str]) -> List[str]:
    """
    Identify and return a list of new or modified PDF files in the specified folder.

    This function compares the MD5 hashes of PDF files in a given folder against existing hash records.
    It returns the file paths of PDFs that are either new (not present in the hash dictionary)
    or have been modified (hash has changed).

    Args:
        pdf_folder (str): The directory path containing PDF files to check.
        existing_hashes (Dict[str, str]): A mapping from PDF filenames to their stored hash values.

    Returns:
        List[str]: List of full file paths for PDFs that are new or have changed since last hash check.
    """
    new_files = []  # Initialize an empty list to store new or modified PDF file paths

    # Iterate through all files in the specified PDF folder
    for filename in os.listdir(pdf_folder):
        # Skip files that do not have a .pdf extension (case-insensitive)
        if not filename.lower().endswith(".pdf"):
            continue

        # Construct the full file path by joining folder and filename
        filepath = os.path.join(pdf_folder, filename)

        # Compute the current MD5 hash of the PDF file's contents
        file_hash = get_file_hash(filepath)

        # Check if the PDF is new (not in existing_hashes) or has changed (hash is different)
        if filename not in existing_hashes or existing_hashes[filename] != file_hash:
            # If new or modified, append the full file path to the result list
            new_files.append(filepath)

    # Return the list of new or changed PDF file paths
    return new_files

# ===================== MAIN LOGIC =====================
def process_pdfs(pdf_files: List[str], persist_dir: str):
    """
    Load PDF files, clean and normalize their text, chunk texts into smaller pieces,
    create embeddings for each chunk, and persist these into a Chroma vector store.

    Args:
        pdf_files (List[str]): List of PDF file paths to process.
        persist_dir (str): Directory path for Chroma persistence storage.

    Behavior:
        - If no PDFs provided, logs and exits early.
        - Processes each PDF by loading, cleaning, and chunking its content.
        - Adds metadata including file hash, source filepath, chunk id, and unique UUID.
        - Embeds all chunks using a large embedding function.
        - Persists all chunks into the Chroma vector database.
    """
    # Exit early if no PDF files to process
    if not pdf_files:
        print("No new PDFs found.")
        return

    # Log how many new PDFs were found, showing basenames
    print(f"Found {len(pdf_files)} new PDF(s): {[os.path.basename(f) for f in pdf_files]}")

    # Initialize the embedding function to convert text chunks into vector embeddings
    embeddings = get_ollama_mxbai_embed_large_embedding_function()

    # Initialize a recursive character-based text splitter with chunk size 1500 and overlap 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    all_docs = []  # List to accumulate all processed chunks from all PDFs

    # Loop over each PDF file path
    for filepath in pdf_files:
        print(f" Creating embeddings for {filepath.split('//')[-1]} chunking to update Chroma DB...")
        # Load PDF contents into document objects using PyPDFLoader
        loader = PyPDFLoader(filepath)
        docs = loader.load()

        # Clean and normalize the textual content of each document/page
        for d in docs:
            d.page_content = normalize_text(clean_text(d.page_content))
            # Optionally remove reference sections if configured to do so
            if cfg.CLEAN_REFERENCES:
                d.page_content = remove_references(d.page_content)

        # Split the cleaned documents into smaller chunks for embedding
        chunks = text_splitter.split_documents(docs)

        # Compute the hash of the PDF file to later track changes
        file_hash = get_file_hash(filepath)

        # Add metadata to each chunk for traceability and identification
        for idx, chunk in enumerate(chunks):
            chunk.metadata["source"] = filepath  # full original file path
            chunk.metadata["hash"] = file_hash  # file content MD5 hash
            chunk.metadata["chunk_id"] = f"{os.path.basename(filepath)}_{idx:03d}"  # unique chunk id with filename & index
            chunk.metadata["uuid"] = str(uuid.uuid4())  # universally unique id for this chunk

        # Add all chunks from current file to accumulated list
        all_docs.extend(chunks)

    # Log the number of chunks to be embedded and persisted
    
    print(f"Embedding and persisting {len(all_docs)} chunks to Chroma vector store for all the new Papers...")
    # Initialize Chroma vector database with persistence directory and embedding function
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Add all documents (chunks) into the Chroma vector store
    db.add_documents(all_docs)

    # Persist the updated database to disk
    db.persist()

    # Confirm successful update to RAG vector index
    print("RAG updated successfully!")
# =====================================================

def update_rag(pdf_folder: str, persist_dir: str):
    """
    Check for new or updated PDF files in a folder and process them to update the RAG index.

    This function orchestrates the flow of RAG updating by:
    - Loading existing metadata (hashes) from the persistent Chroma storage.
    - Identifying new or changed PDFs in the specified folder by comparing hashes.
    - Processing any new or modified PDFs to load, clean, chunk, embed, and persist their data.

    Args:
        pdf_folder (str): Path to the folder containing PDF files to check.
        persist_dir (str): Path to the directory where Chroma persistence is stored.
    """
    # Inform start of the RAG update check process
    print("Checking for new or updated PDFs...")

    # Load existing metadata (file hashes) from Chroma persistence directory
    existing = load_existing_metadata(persist_dir)

    # Find PDFs that are new or whose contents have changed compared to stored hashes
    new_pdfs = get_new_pdfs(pdf_folder, existing)

    # Process these new or updated PDFs by loading, chunking, embedding, and storing them
    process_pdfs(new_pdfs, persist_dir)
    

if __name__ == "__main__":
    # os.makedirs(cfg.PDF_FOLDER, exist_ok=True)
    os.makedirs(cfg.PERSIST_DIR, exist_ok=True)
    update_rag(cfg.PDF_FOLDER, cfg.PERSIST_DIR)