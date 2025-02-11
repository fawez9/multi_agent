import os
from needs import llm, embeddings  
from typing import Optional, List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class BaseRAG:
    """Base RAG class for multi-agent system"""
    
    def __init__(self, 
                 docs_path: str = "./knowledge_base",
                 collection_name: Optional[str] = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG system
        
        Args:
            docs_path: Path to the documents directory
            collection_name: Name for the FAISS index
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.docs_path = docs_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings and vectorstore
        self.embeddings = embeddings
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> FAISS:
        """Initialize or load the vector store"""
        index_path = f"./vectorstores/{self.collection_name}" if self.collection_name else "./vectorstores/default"
        
        if os.path.exists(f"{index_path}.faiss"):
            print(f"Loading existing FAISS index from {index_path}...")
            return FAISS.load_local(index_path, self.embeddings)
        else:
            print("Creating new FAISS index...")
            return self._create_new_vectorstore()

    def _create_new_vectorstore(self) -> FAISS:
        """Create a new FAISS vector store from documents"""
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Documents directory not found: {self.docs_path}")
        
        # Load and split documents
        loader = DirectoryLoader(self.docs_path, glob="**/*.*")
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents found in the specified directory.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        
        # Create FAISS index
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        # Save the index
        if self.collection_name:
            self._save_vectorstore(vectorstore, self.collection_name)
        
        return vectorstore

    def _save_vectorstore(self, vectorstore: FAISS, name: str) -> None:
        """Save FAISS index and metadata"""
        index_path = f"./vectorstores/{name}"
        os.makedirs("./vectorstores", exist_ok=True)
        vectorstore.save_local(index_path)

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

    def generate_response(self, query: str, k: int = 3) -> str:
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
        
        # Generate response using a generative model (e.g., Gemini)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = llm.invoke(prompt)
        
        return response.content

    def add_documents(self, file_paths: List[str]) -> None:
        """
        Add new documents to the existing vector store
        
        Args:
            file_paths: List of paths to new documents
        """
        if not file_paths:
            raise ValueError("No file paths provided.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        new_docs = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            with open(file_path, 'r') as f:
                text = f.read()
            splits = text_splitter.split_text(text)
            new_docs.extend([Document(page_content=text) for text in splits])
        
        if new_docs:
            # Create temporary FAISS index for new documents
            new_vectorstore = FAISS.from_documents(new_docs, self.embeddings)
            
            # Merge with existing index
            self.vectorstore.merge_from(new_vectorstore)
            
            # Save updated index
            if self.collection_name:
                self._save_vectorstore(self.vectorstore, self.collection_name)
        else:
            print("No new documents were added.")

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = BaseRAG(docs_path="./knowledge_base", collection_name="my_index")
    
    # Query the system
    query = "Where is the Eiffel Tower located?"
    response = rag.generate_response(query)
    print("Response:", response)
    
    # Add new documents
    # rag.add_documents(["./new_docs/doc1.txt", "./new_docs/doc2.txt"])
    
    # Query again with the updated knowledge base
    query = "What is the highest peak in the world?"
    response = rag.generate_response(query)
    print("Updated Response:", response)