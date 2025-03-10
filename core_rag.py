import os
import time
from needs import llm, embeddings, connection, collection_name, text_splitter
from typing import Optional, List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_postgres.vectorstores import PGVector


class BaseRAG:
    """Base RAG class for multi-agent system"""
    
    def __init__(self, 
                 docs_path: str = "./knowledge_base", #can be changed to be parameterized
                 collection_name: Optional[str] = None,):
        """
        Initialize the RAG system
        
        Args:
            docs_path: Path to the documents directory
            collection_name: Name for the vector store collection
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.docs_path = docs_path
        self.collection_name = collection_name or "default"
        
        # Initialize embeddings and vectorstore
        self.embeddings = embeddings
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> PGVector:
        """Initialize or load the vector store"""
        try:
            # Try to load existing collection from PostgreSQL
            # Try to load existing collection from PostgreSQL
            print(f"Attempting to load existing collection '{self.collection_name}' from PostgreSQL...")
            self.vectorstore = PGVector(
                collection_name=self.collection_name,
                connection=connection,
                embeddings=self.embeddings
            )
            verify = self._document_exists(self.docs_path)[0]
            if verify:
                print(f"Document similar to {self.docs_path} appears to already exist in the database.")
                return self.vectorstore
            
        except Exception as e:
            print(f"Error loading collection: {e}. Will create a new vector store.")
            
        # Create a new vector store if loading failed or documents don't exist
        return self._create_new_vectorstore()

    def _create_new_vectorstore(self):
        """Create a new vector store from documents in the knowledge base"""
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Documents directory not found: {self.docs_path}")
        
        # Load and split documents
        loader = DirectoryLoader(self.docs_path, glob="**/*.*")
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents found in the specified directory.")
        
        texts = text_splitter.split_documents(documents)
        
        print(f"Creating new vector store with {len(texts)} document chunks...")
        vector_store = PGVector.from_documents(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection=connection,
            documents=texts,
            use_jsonb=True
        )
        
        return vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Perform similarity search on the vector store
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise Exception("Vector store not initialized")
        return self.vectorstore.similarity_search(query, k=k)

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """
        Get relevant context as a formatted string
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Formatted context string
        """
        docs = self.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def generate_response(self, query: str, k: int = 2) -> str:
        """
        Generate a response using the RAG workflow
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Generated response
        """
        # Retrieve relevant context
        context = self.get_relevant_context(query, k=k)
        
        # Generate response using LLM
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = llm.invoke(prompt)
        time.sleep(2)
        
        return response.content

    def add_documents(self, file_paths: List[str]) -> None:
        """
        Add new documents to the existing vector store if they don't already exist
        
        Args:
            file_paths: List of paths to new documents
        """
        if not file_paths:
            raise ValueError("No file paths provided.")
        
        new_docs = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            verify, text = self._document_exists(file_path)

            if verify:
                print(f"Document similar to {file_path} appears to already exist in the database.")
                continue
                
            
            splits = text_splitter.split_text(text)
            new_docs.extend([Document(page_content=split) for split in splits])
        
        if new_docs:
            # Add new documents to the vector store
            self.vectorstore.add_documents(new_docs)
            print(f"Added {len(new_docs)} new document chunks to the database.")
        else:
            print("No new documents were added.")

    def _document_exists(self, file_path: str) -> tuple[bool, str]:
        """
        Check if a document already exists in the vector store
        
        Args:
            file_path: Path to the document
        
        Returns:
            Tuple of (bool, str): True if the document exists, False otherwise, and the document text
        """
        if os.path.isdir(file_path):
            # Handle directory
            loader = DirectoryLoader(file_path)
            docs = loader.load()
            if not docs:
                raise ValueError("No documents found in the specified directory.")
            text = docs[0].page_content
        else:
            # Handle individual file
            loader = TextLoader(file_path)
            docs = loader.load()
            if not docs:
                raise ValueError(f"Could not load document: {file_path}")
            text = docs[0].page_content
            
        sample_text = text[:100]  # Take first 100 chars as a signature
        try:
            existing_docs = self.vectorstore.similarity_search(sample_text, k=1)
            if existing_docs and sample_text in existing_docs[0].page_content:
                return True, text
        except Exception as e:
            print(f"Error checking for duplicates: {e}.")
        
        return False, text


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = BaseRAG(collection_name=collection_name, docs_path="./knowledge_base")
    
    # Add specific documents
    rag.add_documents(file_paths=["./new_docs/doc1.txt"])