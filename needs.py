# Load environment variables
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# shared.py
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]  # List of messages in the conversation
    applied_role: str
    technical_skills: list
    name: str
    plan: list
    scores: list
    status: str
    current_question: str
    response: str
    technical_score: str
    report: str


# Configure Google GenAI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) # TODO: Change model to "gemini-1.5" for better results (example: "gemini-2.0-flash")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")