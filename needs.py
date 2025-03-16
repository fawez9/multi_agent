# Load environment variables
import os
import re
from psycopg2 import pool
from sqlalchemy import create_engine
from config import config
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Annotated,List, Dict
from typing_extensions import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class State(TypedDict):
    """State of the interview"""
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

def is_uuid(folder_name):
    # UUID format validation pattern
    pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    return bool(pattern.match(folder_name))


# Configure Google GenAI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) # TODO: Change for better model  #NOTE: gemini-2.0-flash is bugging
# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

connection = "postgresql+psycopg://postgres:123321@localhost:6024/interview_db"
collection_name = None

knowledge_base_path = "./knowledge_base"
folders = [f for f in os.listdir(knowledge_base_path) if os.path.isdir(os.path.join(knowledge_base_path, f)) and is_uuid(f)]


engine = create_engine(connection)



if __name__ == "__main__":
    # Initialize the RAG system
    rag ="hello"
