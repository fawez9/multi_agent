import os
import time
from needs import llm, embeddings, connection, collection_name, text_splitter
from typing import Optional, List, Union
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_postgres.vectorstores import PGVector


class BaseRAG:
    """Base RAG class for multi-agent system"""
    
    def __init__(self, docs_path: str = "./knowledge_base", collection_name: Optional[str] = None):
        self.docs_path = docs_path
        self.collection_name = collection_name or "default"
        self.embeddings = embeddings
        self.vectorstore = None
        self._initialize_vectorstore()
    def _initialize_vectorstore(self) -> Union[PGVector, None]:
        """Initialize the vector store."""
        try:
            print(f"Attempting to load existing collection '{self.collection_name}'...")
            vectorstore = PGVector(
                collection_name=self.collection_name,
                connection=connection,
                embeddings=self.embeddings
            )
            
            # Set temporarily to check if documents exist
            self.vectorstore = vectorstore
            if os.path.exists(self.docs_path) and self._document_exists(self.docs_path)[0]:
                print("Collection loaded successfully.")
                return
        except Exception as e:
            print(f"Error loading collection: {e}")
            
        return self._create_new_vectorstore()

    def _create_new_vectorstore(self):
        """Create a new vector store from the documents in the specified directory."""
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Documents directory not found: {self.docs_path}")
        
        loader = DirectoryLoader(self.docs_path, glob="**/*.*")
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents found in the specified directory.")
        
        texts = text_splitter.split_documents(documents)
        
        print(f"Creating new vector store with {len(texts)} document chunks...")
        return PGVector.from_documents(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection=connection,
            documents=texts,
            use_jsonb=True
        )

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents in the database."""
        if not self.vectorstore:
            raise Exception("Vector store not initialized")
        return self.vectorstore.similarity_search(query, k=k)

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context for a given query."""
        docs = self.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def generate_response(self, query: str, k: int = 2) -> str:
        """Generates a response to a given query."""
        context = self.get_relevant_context(query, k=k)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = llm.invoke(prompt)
        time.sleep(2)
        return response.content

    def add_documents(self, file_paths: List[str]) -> None:
        """Add new documents to the database."""
        if not file_paths:
            raise ValueError("No file paths provided.")
        
        new_docs = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            verify, text = self._document_exists(file_path)
            if verify:
                print(f"Document similar to {file_path} already exists in the database.")
                continue
            
            splits = text_splitter.split_text(text)
            new_docs.extend([Document(page_content=split) for split in splits])
        
        if new_docs:
            self.vectorstore.add_documents(new_docs)
            print(f"Added {len(new_docs)} new document chunks to the database.")
        else:
            print("No new documents were added.")

    def _document_exists(self, file_path: str) -> tuple[bool, str]:
        """Check if a document already exists in the database."""
        if os.path.isdir(file_path):
            docs = DirectoryLoader(file_path).load()
        else:
            docs = TextLoader(file_path).load()
            
        if not docs:
            raise ValueError(f"No documents found at: {file_path}")
            
        text = docs[0].page_content
        sample_text = text[:100]
        
        try:
            existing_docs = self.vectorstore.similarity_search(sample_text, k=1)
            if existing_docs and sample_text in existing_docs[0].page_content:
                return True, text
        except Exception as e:
            print(f"Error checking for duplicates: {e}.")
        
        return False, text


rag = BaseRAG(collection_name=collection_name, docs_path="./knowledge_base")
if __name__ == "__main__":
    rag.add_documents(file_paths=["./new_docs/doc1.txt"])
    response = rag.generate_response("What is the candidate's name?")
    print(response)
