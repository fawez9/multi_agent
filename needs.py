# Load environment variables
import os
from psycopg2 import pool
from config import config
from dotenv import load_dotenv
import google.generativeai as genai
from typing_extensions import TypedDict
from typing import Annotated,List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# State definition
class State(TypedDict):
    messages: List[str]
    applied_role: str
    skills: List[str]
    name: str
    plan: List[str]
    scores: Annotated[List[Dict[str, str]], "List of scores"]
    status: str
    current_question: str
    response: str
    technical_score: Annotated[str, "Technical score"] # Annotated is used to add metadata to the type
    report: str
    email: str
    phone: str
    conversation_history: Annotated[List[Dict[str, str]], "List of conversation history"]


db_config = config()
# Connect to PostgreSQL database
# print("Connecting to the PostgreSQL database...")
connection_pool = pool.SimpleConnectionPool(**db_config)


# Close the connection pool when the application shuts down
def close_connection_pool():
    connection_pool.closeall()
    print("Connection pool closed.")


# Configure Google GenAI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) # TODO: Change for better model 
# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

connection = "postgresql+psycopg://postgres:123321@localhost:6024/interview_db"
collection_name = "state_of_uninon_vectors"



if __name__ == "__main__":
    # Initialize the RAG system
    rag ="hello"
