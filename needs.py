# Load environment variables
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing_extensions import TypedDict
from typing import Annotated,List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# State definition
class State(TypedDict):
    messages: List[str]
    applied_role: str
    technical_skills: List[str]
    name: str
    plan: List[str]
    scores: Annotated[List[Dict[str, str]], "List of scores"]
    status: str
    current_question: str
    response: str
    technical_score: Annotated[str, "Technical score"] # Annotated is used to add metadata to the type
    report: str


# Configure Google GenAI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) # TODO: Change model to "gemini-1.5" for better results (example: "gemini-2.0-flash")
# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")