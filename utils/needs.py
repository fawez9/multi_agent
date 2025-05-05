"""Module for managing application dependencies and configurations.

This module handles database connections, AI/ML components, and file system utilities
for the interview system. It centralizes configuration and resource management.
"""

# Standard library imports
import os
import re
from typing import Annotated, Dict, List, Optional
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
from utils.config import config

# Load environment variables at module import time
load_dotenv()


# Type definitions
class State(TypedDict):
    """State of the interview process.

    Contains all data related to the current interview session.
    """
    messages: List[str]
    name: str
    phone: str
    email: str
    skills: List[str]
    plan: List[str]
    status: str
    current_question: str
    response: str
    scores: Annotated[List[Dict[str, str]], "List of scores"]
    report: str
    conversation_history: Annotated[List[Dict[str, str]], "List of conversation history"]
    _internal_flags: Dict[str, bool]
    job_details: Dict[str, str]


# Database configuration functions
def build_connection_string(config_dict: Optional[Dict[str, str]] = None) -> str:
    """Build a database connection string from configuration.

    Args:
        config_dict: Optional dictionary with database configuration parameters.
            If not provided, environment variables will be used.

    Returns:
        A formatted connection string for SQLAlchemy.

    Raises:
        ValueError: If neither environment variables nor config_dict contain
            the required database configuration.
    """
    # First try to use environment variables
    if all(os.getenv(var) for var in ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]):
        return (
            f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
            f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
    # Fall back to config file if environment variables are not set
    elif config_dict:
        return (
            f"postgresql+psycopg://{config_dict['user']}:{config_dict['password']}@"
            f"{config_dict['host']}:{config_dict['port']}/{config_dict['database']}"
        )
    else:
        raise ValueError("Database configuration not found in environment variables or config file")


def get_db_config() -> Dict[str, str]:
    """Get database configuration from environment variables or config file.

    Returns:
        Dictionary with database configuration parameters.
    """
    # Try to use environment variables first
    if all(os.getenv(var) for var in ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]):
        return {
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD"),
            'host': os.getenv("DB_HOST"),
            'port': os.getenv("DB_PORT"),
            'database': os.getenv("DB_NAME"),
            'minconn': 1,
            'maxconn': 5
        }
    # Fall back to config file
    return config()


def close_connection_pool() -> None:
    """Close the connection pool when the application shuts down."""
    connection_pool.closeall()
    print("Connection pool closed.")


# File system utilities
def is_uuid(folder_name: str) -> bool:
    """Validate if a folder name matches UUID format.

    Args:
        folder_name: The string to check against UUID pattern.

    Returns:
        True if the string matches UUID format, False otherwise.
    """
    pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    return bool(pattern.match(folder_name))


# Initialize database connections
db_config = get_db_config()
connection_pool = pool.SimpleConnectionPool(**db_config)
connection = build_connection_string(db_config)
engine = create_engine(connection)

# Configure AI/ML components
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

# Get UUID folders in knowledge base
folders = [
    f for f in os.listdir(knowledge_base_path)
    if os.path.isdir(os.path.join(knowledge_base_path, f)) and is_uuid(f)
]


# For testing/development
if __name__ == "__main__":
    # Initialize the RAG system
    rag = "hello"
