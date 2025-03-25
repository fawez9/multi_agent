import os
import uuid
import shutil
import time
from datetime import datetime
from langchain.schema import Document
from typing import Optional, List, Union, Dict
from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from needs import (
    llm, 
    embeddings, 
    connection, 
    text_splitter, 
    knowledge_base_path, 
    engine
)

class BaseRAG:
    """Base RAG class for multi-agent system"""
    #NOTE: the collection name is optional you can add it as a parameter
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.vectorstore = None
        if collection_name:
            self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> None:
        """Initialize the vector store for an existing collection."""
        try:
            print(f"Loading collection '{self.collection_name}'...")
            self.vectorstore = PGVector(
                collection_name=self.collection_name,
                connection=connection,
                embeddings=self.embeddings
            )
            print("Collection loaded successfully.")
        except Exception as e:
            print(f"Error loading collection: {e}")
            raise

    def process_document(self, file_path: str, candidate_id: Optional[int] = None) -> Dict[str, Union[str, int]]:
        """
        Process a single document and create vector store entry.
        
        Args:
            file_path: Path to the document file
            candidate_id: Optional ID of existing candidate
            
        Returns:
            Dict containing session_id, collection_id, and collection_name
        """
        try:
            # Generate UUID for document collection
            collection_uuid = str(uuid.uuid4())
            collection_name = f"collection_{collection_uuid[:8]}"
            
            # Create and setup document directory
            doc_dir = os.path.join(knowledge_base_path, collection_uuid)
            os.makedirs(doc_dir, exist_ok=True)
            
            # Copy document to new directory
            doc_name = os.path.basename(file_path)
            new_path = os.path.join(doc_dir, doc_name)
            shutil.copy2(file_path, new_path)
            
            # Load and process document based on type
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(new_path)
            else:
                loader = TextLoader(new_path)
                
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            
            # Create database entries
            with engine.connect() as conn:
                cursor = conn.connection.cursor()
                
                # Create collection
                cursor.execute(
                    """
                    INSERT INTO langchain_pg_collection (uuid, name, cmetadata) 
                    VALUES (%s, %s, '{}'::jsonb) 
                    RETURNING uuid
                    """,
                    (collection_uuid, collection_name)
                )
                collection_id = cursor.fetchone()[0]
                
                # Create session if candidate_id is provided
                session_id = None
                #TODO : change candidate_id to be a generated and given later for him on signup (to know which session belongs to which candidate)
                if candidate_id:
                    cursor.execute(
                        """
                        INSERT INTO sessions 
                        (candidate_id, collection_id, date) 
                        VALUES (%s, %s, %s) 
                        RETURNING id
                        """,
                        (candidate_id, collection_id, datetime.now()) #TODO: change to a specific date
                    )
                    session_id = cursor.fetchone()[0]
                
                conn.connection.commit()
            
            # Create vector store
            self.vectorstore = PGVector.from_documents(
                embedding=embeddings,
                collection_name=collection_name,
                connection=connection,
                use_jsonb=True,
                documents=texts,
            )
            
            self.collection_name = collection_name
            print(f"Processed document with {len(texts)} chunks")
            
            return {
                "session_id": session_id,
                "collection_id": collection_id,
                "collection_name": collection_name
            }
            
        except Exception as e:
            print(f"Error processing document: {e}")
            raise

    def load_session(self, session_id: int) -> None:
        """Load a specific interview session."""
        with engine.connect() as conn:
            cursor = conn.connection.cursor()
            cursor.execute(
                """
                SELECT c.name as collection_name 
                FROM sessions s
                JOIN langchain_pg_collection c ON s.collection_id = c.uuid
                WHERE s.id = %s
                """,
                (session_id,)
            )
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Session {session_id} not found")
                
            self.collection_name = result[0]
            self._initialize_vectorstore()

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

# Create single instance
rag = BaseRAG() 
# rag.process_document("/home/fawez/Downloads/HATTABI_FAWEZ_RES.pdf", candidate_id=1)
rag.load_session(1) #TODO: change the behaviour of session ids for now i'm manually loading a specific session
if __name__ == "__main__":
    response = rag.generate_response("What is the candidate's name?")
    print(response)