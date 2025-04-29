import os
import sys
import numpy as np
from typing import List, Dict, Any
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import torch
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config.config_loader import load_config

class RAGSystem:
    def __init__(
        self, 
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize the RAG system with embedding model.
        
        Args:
            embedding_model_name: The name of the SentenceTransformer model to use
                                 (default model supports Chinese text)
            collection_name: Name for the ChromaDB collection
        """
        # Initialize embedding model - use a multilingual model for Chinese support
        self.config = load_config()
        self.embedding_model_name = embedding_model_name
        self.collection_name = self.config["memory_processing"]["collection_name"]
        self.database_path = os.path.join(self.config["BASE_DIR"], self.config["memory_processing"]["database_path"])
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.database_path)
        
        # Create or get collection with sentence transformer embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Created new collection: {self.collection_name}")
        
        # Initialize text splitter for chunking with settings for Chinese
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
    
    def ingest_document(self, document_text: str, metadata: Dict[str, Any] = None, batch_size=100):
        """
        Ingest a document into the RAG system using ChromaDB.
        
        Args:
            document_text: The text content of the document
            metadata: Optional metadata for the document
            batch_size: Number of chunks to process in a single batch for better performance
        """
        # Check for duplicate if source_path is provided
        if metadata and "source_path" in metadata and os.path.exists(metadata["source_path"]):
            file_stat = os.stat(metadata["source_path"])
            file_basename = os.path.basename(metadata["source_path"])
            file_identity = f"{file_basename}_{file_stat.st_size}_{file_stat.st_mtime}"
            
            # Add file identity to metadata
            if not metadata:
                metadata = {}
            metadata["file_identity"] = file_identity
            
            # Check if this file was already ingested
            try:
                existing = self.collection.get(
                    where={"file_identity": file_identity}
                )
                if existing and len(existing['ids']) > 0:
                    print(f"Document {file_basename} already ingested (found {len(existing['ids'])} chunks). Skipping.")
                    return
            except Exception as e:
                # If the query fails, continue with ingestion
                pass
        
        # Split text into chunks using LangChain's splitter
        chunks = self.text_splitter.split_text(document_text)
        
        # Prepare document IDs, metadatas for each chunk
        ids = [f"doc_{i}_{pd.Timestamp.now().timestamp()}" for i in range(len(chunks))]
        
        # Prepare metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            chunk_metadata["chunk_size"] = len(chunk)
            metadatas.append(chunk_metadata)
        
        # Add chunks in batches for better performance
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            self.collection.add(
                documents=chunks[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"Ingested document with {len(chunks)} chunks in batches of {batch_size}")
    
    def ingest_csv_chat(self, csv_path=None, skip_duplicates=True, batch_size=100):
        """
        Ingest chat data from a CSV file with sender, msg, and send_time columns.
        
        Args:
            csv_path: Path to the CSV file
            skip_duplicates: Whether to skip duplicate entries by checking file hash
            batch_size: Number of chunks to process in a single batch for better performance
        """
        if not csv_path:
            memory_paths = os.listdir(os.path.join(self.config["BASE_DIR"], self.config["memory_processing"]["memory_path"]))
            for path in memory_paths:
                if path.endswith(".csv"):
                    if path.startswith("extracted_memories"):
                        csv_path = os.path.join(os.path.join(self.config["BASE_DIR"], self.config["memory_processing"]["memory_path"]), path)
                        break
        # Check if file was already ingested
        file_basename = os.path.basename(csv_path)
        file_stat = os.stat(csv_path)
        file_identity = f"{file_basename}_{file_stat.st_size}_{file_stat.st_mtime}"
        
        # Query existing documents with this file identity
        if skip_duplicates:
            try:
                existing = self.collection.get(
                    where={"file_identity": file_identity}
                )
                if existing and len(existing['ids']) > 0:
                    print(f"File {file_basename} already ingested (found {len(existing['ids'])} chunks). Skipping.")
                    return
            except Exception as e:
                # If the query fails (e.g., column doesn't exist yet), continue with ingestion
                print(f"Checking for duplicates failed: {e}. Continuing with ingestion.")
        
        # Read CSV file - try to handle both with and without headers
        try:
            # First try reading with header
            df = pd.read_csv(csv_path)
            
            # Check if we have the expected columns
            if not all(col in df.columns for col in ["sender", "msg", "send_time"]):
                # If not the expected columns, try reading without header
                df = pd.read_csv(csv_path, header=None, names=["sender", "msg", "send_time"])
        except Exception as e:
            print(f"Error reading CSV with header: {e}")
            # Fallback to no header
            df = pd.read_csv(csv_path, header=None, names=["sender", "msg", "send_time"])
        
        # Ensure send_time is numeric
        df['send_time'] = pd.to_numeric(df['send_time'], errors='coerce')
        
        # Drop rows with invalid timestamps
        df = df.dropna(subset=['send_time'])
        
        # Group conversations by date (using timestamp)
        df['date'] = pd.to_datetime(df['send_time'], unit='s').dt.date
        
        # Process each day's conversation as a document
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for date, group in tqdm(df.groupby('date')):
            # Format the conversation as a document
            conversation = ""
            for _, row in group.iterrows():
                sender = "朋友" if row['sender'] == 0 else "你"
                conversation += f"{sender}: {row['msg']}\n\n"
            
            # Create metadata
            metadata = {
                "source": csv_path,
                "date": str(date),
                "conversation_id": f"conv_{date}",
                "document_type": "chat",
                "file_identity": file_identity  # Add file identity for duplicate detection
            }
            
            # Split into chunks (more efficient than calling ingest_document separately)
            chunks = self.text_splitter.split_text(conversation)
            
            # Prepare document IDs, metadatas for each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["chunk_size"] = len(chunk)
                
                all_chunks.append(chunk)
                all_metadatas.append(chunk_metadata)
                all_ids.append(f"doc_{date}_{i}_{pd.Timestamp.now().timestamp()}")
        
        # Add chunks in batches for better performance
        total_chunks = len(all_chunks)
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            self.collection.add(
                documents=all_chunks[i:end_idx],
                metadatas=all_metadatas[i:end_idx],
                ids=all_ids[i:end_idx]
            )
            
        print(f"Ingested {file_basename} with {total_chunks} total chunks in batches of {batch_size}")
    
    def ingest_memories_csv(self, csv_path=None, skip_duplicates=True, batch_size=100):
        """
        Ingest extracted memories from a CSV file (exported by ChatMemoryExtractor).
        
        Args:
            csv_path: Path to the memories CSV file
            skip_duplicates: Whether to skip duplicate entries by checking file hash
            batch_size: Number of chunks to process in a single batch for better performance
        """
        if not csv_path:
            memory_paths = os.listdir(os.path.join(self.config["BASE_DIR"], self.config["memory_processing"]["memory_path"]))
            for path in memory_paths:
                if path.endswith(".csv"):
                    if path.startswith("extracted_memories"):
                        csv_path = os.path.join(os.path.join(self.config["BASE_DIR"], self.config["memory_processing"]["memory_path"]), path)
                        break
        # Check if file was already ingested
        file_basename = os.path.basename(csv_path)
        file_stat = os.stat(csv_path)
        file_identity = f"{file_basename}_{file_stat.st_size}_{file_stat.st_mtime}"
        
        # Query existing documents with this file identity
        if skip_duplicates:
            try:
                existing = self.collection.get(
                    where={"file_identity": file_identity}
                )
                if existing and len(existing['ids']) > 0:
                    print(f"Memory file {file_basename} already ingested (found {len(existing['ids'])} chunks). Skipping.")
                    return
            except Exception as e:
                # If the query fails (e.g., column doesn't exist yet), continue with ingestion
                print(f"Checking for duplicates failed: {e}. Continuing with ingestion.")
        
        # Read memories CSV file
        try:
            df = pd.read_csv(csv_path)
            
            # Check if we have the expected columns
            expected_cols = ['start_time', 'end_time', 'start_datetime', 'end_datetime', 'memory', 'participants']
            if not all(col in df.columns for col in expected_cols):
                print(f"CSV file {csv_path} doesn't have expected memory columns. Available columns: {df.columns.tolist()}")
                return
        except Exception as e:
            print(f"Error reading memories CSV: {e}")
            return
        
        # Process each memory as a document
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Use the memory text directly (no need to split into chunks as memories are already concise)
            memory_text = row['memory']
            
            # Create metadata
            metadata = {
                "source": csv_path,
                "document_type": "memory",
                "file_identity": file_identity,  # Add file identity for duplicate detection
                "start_time": row['start_time'],
                "end_time": row['end_time'],
                "start_datetime": row['start_datetime'],
                "end_datetime": row['end_datetime'],
                "participants": row['participants'],
                "memory_id": f"memory_{idx}"
            }
            
            # Each memory is a single document
            all_documents.append(memory_text)
            all_metadatas.append(metadata)
            all_ids.append(f"memory_{idx}_{pd.Timestamp.now().timestamp()}")
        
        # Add memories in batches for better performance
        total_memories = len(all_documents)
        for i in range(0, total_memories, batch_size):
            end_idx = min(i + batch_size, total_memories)
            self.collection.add(
                documents=all_documents[i:end_idx],
                metadatas=all_metadatas[i:end_idx],
                ids=all_ids[i:end_idx]
            )
            
        print(f"Ingested {file_basename} with {total_memories} memories")
    
    def ingest_from_directory(self, directory_path: str):
        """
        Ingest all text files, chat CSVs and memory CSVs from a directory.
        
        Args:
            directory_path: Path to the directory containing files
        """
        for filename in tqdm(os.listdir(directory_path)):
            file_path = os.path.join(directory_path, filename)
            
            if filename.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    document_text = file.read()
                
                metadata = {
                    "filename": filename,
                    "source_path": file_path,
                    "ingestion_time": pd.Timestamp.now().isoformat()
                }
                
                self.ingest_document(document_text, metadata)
            elif filename.endswith(".csv"):
                # Try to determine if it's a memory CSV or a chat CSV
                try:
                    df = pd.read_csv(file_path, nrows=1)  # Read just one row to check columns
                    
                    # Check if it's a memories file
                    if "memory" in df.columns and "start_time" in df.columns:
                        self.ingest_memories_csv(file_path)
                    # Otherwise assume it's a chat file
                    else:
                        self.ingest_csv_chat(file_path)
                except Exception as e:
                    print(f"Failed to ingest CSV file {filename}: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks using ChromaDB.
        
        Args:
            query: The search query (supports Chinese)
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "id": results['ids'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def prepare_context(self, query: str, top_k: int = 5) -> str:
        """
        Prepare context for LLM by retrieving and formatting relevant documents.
        
        Args:
            query: The user's query
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string ready to be used with an LLM
        """
        # Retrieve relevant documents
        retrieved_docs = self.search(query, top_k=top_k)
        
        # Format context from retrieved documents
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            # Format differently based on document type
            metadata = doc['metadata']
            doc_type = metadata.get('document_type', 'unknown')
            
            if doc_type == 'memory':
                # Format memories with their date information
                start_dt = metadata.get('start_datetime', 'unknown date')
                memory_text = doc['content']
                context_parts.append(f"记忆 {i+1} [时间: {start_dt}]:\n{memory_text}")
            else:
                # Default format for other document types
                context_parts.append(f"文档 {i+1}:\n{doc['content']}")
        
        context = "\n\n".join(context_parts)
        return context
        
    def get_collection_stats(self):
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "document_count": count,
            "collection_name": self.collection.name,
            "embedding_model": self.embedding_model_name,
            "database_path": self.database_path
        }