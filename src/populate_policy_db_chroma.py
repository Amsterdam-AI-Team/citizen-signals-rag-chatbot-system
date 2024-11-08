import argparse
import os
import re
import shutil

from typing import List

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from codecarbon import EmissionsTracker, track_emissions

import config as cfg
from helpers.embedding_helpers import OpenAIEmbeddingFunction

class TXTDirectoryLoader:
    """Loader class to load and preprocess TXT files from a directory."""

    def __init__(self, path: str):
        """
        Initialize the TXTDirectoryLoader with the directory path.
        
        Args:
            path (str): Path to the directory containing TXT files.
        """
        self.path = path

    def load(self) -> List[Document]:
        """
        Load and preprocess TXT files from the directory, and convert them to Document objects.
        
        Returns:
            List[Document]: List of Document objects containing the preprocessed text and metadata.
        """
        documents = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        text = self._preprocess_text(text)
                        metadata = {"source": file_path}
                        documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing unnecessary newlines and preserving specific patterns.
        
        Args:
            text (str): Raw text to preprocess.
        
        Returns:
            str: Preprocessed text.
        """
        text = re.sub(r'\n+', ' ', text)  # Replace unnecessary newlines
        text = re.sub(r'\[LINK: [^\]]+\]\([^\)]+\)', lambda m: m.group(0).replace('\n', ''), text)
        text = re.sub(r'\[IMG: [^\]]+\]', lambda m: m.group(0).replace('\n', ''), text)
        return text

def load_documents() -> List[Document]:
    """
    Load documents from different formats (PDF, TXT, HTML) in the data directory.
    
    Returns:
        List[Document]: List of loaded Document objects.
    """
    documents = []

    # Load TXT documents
    txt_loader = TXTDirectoryLoader(cfg.DOCUMENTS_PATH)
    documents.extend(txt_loader.load())

    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents (List[Document]): List of Document objects to split.
    
    Returns:
        List[Document]: List of split Document objects.
    """
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    for document in documents:
        text_chunks = text_splitter.split_text(document.page_content)
        for idx, chunk in enumerate(text_chunks):
            chunk_doc = Document(page_content=chunk, metadata=document.metadata)
            chunk_doc.metadata["chunk_index"] = idx
            chunks.append(chunk_doc)

    return chunks

def add_to_chroma(chunks: List[Document]):
    """
    Add or update document chunks in the Chroma database.
    
    Args:
        chunks (List[Document]): List of Document objects (chunks) to add.
    """
    db = Chroma(persist_directory=cfg.CHROMA_PATH, embedding_function=OpenAIEmbeddingFunction())
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        for chunk, chunk_id in tqdm(zip(new_chunks, [chunk.metadata["id"] for chunk in new_chunks]), total=len(new_chunks), desc="Adding documents"):
            db.add_documents([chunk], ids=[chunk_id])
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Calculate unique IDs for each chunk based on its source, page, and chunk index.
    
    Args:
        chunks (List[Document]): List of Document objects (chunks) to process.
    
    Returns:
        List[Document]: List of Document objects with assigned IDs.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)  # Default page to 0 for non-txt documents
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks

def clear_database():
    """Clear the existing Chroma database by removing the directory."""
    if os.path.exists(cfg.CHROMA_PATH):
        shutil.rmtree(cfg.CHROMA_PATH)


# Example usage: process a folder of txt files
if __name__ == "__main__":
    """Main function to parse arguments, load documents, process them, and manage the database."""

    # Load and process documents
    if cfg.track_emissions:
        tracker = EmissionsTracker(experiment_id = "oneoff_policy_documents_embedding")
        tracker.start()

    parser = argparse.ArgumentParser(description="Manage the document database.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database...")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    if cfg.track_emissions:
        tracker.stop()