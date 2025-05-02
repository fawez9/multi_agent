import time
import utils.candidate as candidate
from utils.needs import State
from rag.core_rag import rag
from langgraph.graph import StateGraph
from database_store import store_interview_data
from agents.interview_agent import start_interview_agent
from agents.evaluation_agent import start_evaluation_agent
from utils.stt_tts import text_to_speech_and_play
from utils.shared_state import shared_state


def start_interview(state: State):
    """Initialize interview state."""
    return {
        'messages': [],
        'plan': [],
        'status': 'Plan Incomplete',
        'current_question': '',
        'response': '',
        '_internal_flags': {
            'needs_refinement': False,
            'question_answered': False,
            'question_refined': False
        },
        'scores': [],
        'report': '',
        'conversation_history': [],
    }

def initialize_candidate_info(state: State):
    """Initialize candidate information."""
    candidate_info = candidate.process_candidate_info()

    # Create a new state with all the information
    new_state = {
        **state,  # Include all existing state
        'name': candidate_info['name'],
        'phone': candidate_info['phone'],
        'email': candidate_info['email'],
        'skills': candidate_info['skills'],
        'job_details': candidate_info['job_details']
    }
    print("Candidate information initialized:", new_state)
    text='Welcome '+new_state['name']+' for your interview for a '+new_state['job_details']['applied_role']+' position at '+new_state['job_details']['company']+', we will generate your interview plan now'
    print(text)

    # Add the welcome message to shared state
    shared_state.add_message("assistant", text)

    # text_to_speech_and_play(text)

    return new_state

def generate_interview_plan(state: State):
    """Generates an interview plan based on the candidate's role and skills."""
    # Get candidate information from the state
    candidate_skills = state.get('skills', [])

    # If job_details is missing, try to get it from candidate.py
    if 'job_details' not in state:
        candidate_info = candidate.process_candidate_info()
        job_details = candidate_info.get('job_details', {})

        # Update the state with job_details
        state['job_details'] = job_details

    # Get job details from state
    applied_role = state['job_details'].get('applied_role', 'Unknown Role')
    job_skills = state['job_details'].get('skills', [])
    job_description = state['job_details'].get('description', "No description available")
    nb_questions = 1   #TODO : param for questions
    try:
        # Generate questions using the RAG system
        #TODO : enhance the prompt to maintain a stable TTS
        prompt = f"""
        these are infos about the candidate: {candidate_skills}
        these are infos about the job: {job_skills} {job_description}
        and this is the role that the candidate applied for: {applied_role}
        Generate {nb_questions} questions for the interview based on the candidate's profile and the job requirements.
        Format each question as a numbered item.
        """
        response = rag.generate_response(query=prompt)  # Use the RAG system to generate questions
        time.sleep(2)
        # Extract only the actual questions, filtering out any explanatory text
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 4)):  #TODO : handle questions
                questions.append(line.strip())
    except Exception as e:
        print(f"API Error: {str(e)}")
        return {'plan': ['API Error: Failed to generate questions'], 'status': 'Plan Complete'}

    text="now we will start the interview please prepare yourself and answer the questions clearly you'll get a one refinement per question make sure to understand the questions before answering, good luck"
    print(text)

    # Add the start message to shared state
    shared_state.add_message("assistant", text)

    # text_to_speech_and_play(text)
    return {'plan': questions,'nb_questions':nb_questions}


def generate_report(state: dict):
    """Generates the final interview report."""
    # Generate the report text
    report = [
        f"""
----------------Interview Report for {state['name']}------------------
Position: {state['job_details']['applied_role']}
Skills: {state['skills']}\n
-----------------------Interview Scores-------------------------------
"""
    ]

    # Check for API error
    if state['plan'] == ['API Error: Failed to generate questions']:
        print("API Error: Failed to generate questions")
        return {'report': "API Error: Failed to generate questions", 'state': state}
    else:
        for i, score in enumerate(state.get('scores', []), 1):
            report.append(f"{i}. Question: {score['question']}")
            report.append(f"   Answer: {score['response']}")
            report.append(f"   Evaluation: {score['evaluation']}\n")

    report_text = '\n'.join(report)
    return {'report': report_text, 'state': state}


def end_interview(state: State):
    """Ends the interview and displays the final report."""
    print("\n" + state['report'])

# Define the workflow using StateGraph
workflow = StateGraph(State)

# Add nodes to the workflow
nodes = [
    ("start", start_interview),
    ("init", initialize_candidate_info),
    ("gen_plan", generate_interview_plan),
    ("interview_agent", start_interview_agent),
    ("evaluation_agent",start_evaluation_agent),
    ("gen_report", generate_report),
    ("database_store", store_interview_data),
    ("end", end_interview)
]

for name, func in nodes:
    workflow.add_node(name, func)

# Configure workflow edges
workflow.set_entry_point("start")
workflow.add_edge("start", "init")
workflow.add_edge("init", "gen_plan")
workflow.add_edge("gen_plan", "interview_agent")
workflow.add_edge("interview_agent", "evaluation_agent")
workflow.add_edge("evaluation_agent","gen_report")
# workflow.add_edge("gen_report", "database_store")
# workflow.add_edge("database_store", "end")
workflow.add_edge("gen_report", "end")
workflow.set_finish_point("end")


# Compile the workflow
interview_flow = workflow.compile()

# Run the workflow
if __name__ == "__main__":
    print("Starting interview workflow...\n")
    try:
        # Increase recursion limit to handle more questions
        interview_flow.invoke({"messages": []}, config={"recursion_limit": 100})
    except KeyboardInterrupt:
        print("\nInterview interrupted by user")