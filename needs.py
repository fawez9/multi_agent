# Standard library imports
import os
import re
from typing import Annotated, List, Dict
from typing_extensions import TypedDict

# Third-party imports
from dotenv import load_dotenv
import google.generativeai as genai
from psycopg2 import pool
from sqlalchemy import create_engine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

# Local imports
from config import config

# Type definitions
class State(TypedDict):
    """State of the interview"""
    messages: List[str]
    name: str
    phone: str
    email: str
    applied_role: str
    skills: List[str]
    plan: List[str]
    status: str
    current_question: str
    response: str
    scores: Annotated[List[Dict[str, str]], "List of scores"]
    report: str
    conversation_history: Annotated[List[Dict[str, str]], "List of conversation history"]
    _internal_flags: Dict[str, bool]

# Database configuration
db_config = config()
connection_pool = pool.SimpleConnectionPool(**db_config)
connection = "postgresql+psycopg://postgres:123321@localhost:6024/interview_db"
engine = create_engine(connection)

def close_connection_pool():
    """Close the connection pool when the application shuts down."""
    connection_pool.closeall()
    print("Connection pool closed.")

# File system utilities
def is_uuid(folder_name: str) -> bool:
    """Validate if a folder name matches UUID format."""
    pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    return bool(pattern.match(folder_name))

# AI/ML Configuration
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize LLM components
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
)  # TODO: Change for better model  #BUG: gemini-2.0-flash is bugging

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# File system configuration
knowledge_base_path = "./knowledge_base"
os.makedirs(knowledge_base_path, exist_ok=True)

folders = [
    f for f in os.listdir(knowledge_base_path)
    if os.path.isdir(os.path.join(knowledge_base_path, f)) and is_uuid(f)
]

# For testing/development
if __name__ == "__main__":
    # Initialize the RAG system
    rag = "hello"
