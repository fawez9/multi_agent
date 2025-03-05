# Load environment variables
import os
from psycopg2 import pool
from config import config
from dotenv import load_dotenv
import google.generativeai as genai
from typing_extensions import TypedDict
from typing import Annotated,List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

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


db_config = config()
# Connect to PostgreSQL database
print("Connecting to the PostgreSQL database...")
connection_pool = pool.SimpleConnectionPool(**db_config)


# Close the connection pool when the application shuts down
def close_connection_pool():
    connection_pool.closeall()
    print("Connection pool closed.")


# Configure Google GenAI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) # TODO: Change model to "gemini-1.5" for better results (example: "gemini-2.0-flash")
# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


if __name__ == "__main__":
    # Initialize the RAG system
    rag ="hello"