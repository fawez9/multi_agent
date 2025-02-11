from needs import llm
from typing import Annotated
from candidate import candidate_info
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages



# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages] # List of messages in the conversation
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

def start_interview(state: State):
    """Initialize interview state."""
    return {
        'messages': [],
        'current_question': '',
        'response': '',
        'technical_score': '',
        'report': '',
        'scores': [],
        'plan': [],
        'status': 'Plan Incomplete'
    }

def initialize_candidate_info(state: State):
    """Initialize candidate information."""
    return {
        'applied_role': candidate_info['applied_role'],
        'technical_skills': candidate_info['technical_skills'],
        'name': candidate_info['name'],
    }

def generate_interview_plan(state: State):
    """Generates an interview plan based on the candidate's role and skills."""
    role = state.get('applied_role', state['applied_role'])
    skills = state.get('technical_skills', state['technical_skills'])
    
    try:
        prompt = f"Generate 3 technical interview questions for {role} focusing on {', '.join(skills)} also dont give long codes to implement. Format each question as a numbered item:"
        response = llm.invoke(prompt)
        # Extract only the actual questions, filtering out any explanatory text
        questions = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 4)): # Check if line starts with a number
                questions.append(line.strip())
    except Exception as e:
        print(f"API Error: {str(e)}")
        # questions = ['API Error: Failed to generate questions']
        return {'plan': ['API Error: Failed to generate questions'], 'status': 'Plan Complete'}
    
    return {'plan': questions, 'status': 'Plan Incomplete'}

def check_interview_plan(state: State):
    """Checks the status of the interview plan."""
    is_complete = not state['plan'] or len(state['plan']) == 0
    return {'status': 'Plan Complete' if is_complete else 'Plan Incomplete'}

def present_question(state: State):
    """Presents the next question in the interview plan."""
    if not state['plan']:
        return {'current_question': '', 'status': 'Plan Complete'}
    
    current_question = state['plan'][0]
    remaining_plan = state['plan'][1:]  # Remove the current question from the plan
    
    return {
        'current_question': current_question,
        'plan': remaining_plan  # Update the plan to remove the current question
    }

def collect_response(state: State):
    """Collects the candidate's response to the current question."""
    if state['current_question']:
        print(f"\nQ: {state['current_question']}")
        response = input("Your answer: ").strip()
        return {
            'response': response,
        }
    return {'response': ''}

def evaluate_technical_response(state: State):
    """Evaluates the candidate's response to the current question."""
    try:
        prompt = f"Evaluate this technical response. Question: {state['current_question']}\nResponse: {state['response']}\nProvide a score out of 100 and detailed feedback dont hesitate to be critical."
        evaluation = llm.invoke(prompt)
        return {'technical_score': evaluation.content}
    except Exception as e:
        return {'technical_score': f"Evaluation failed: {str(e)}"}

def calculate_score(state: State):
    """Calculates and stores the candidate's score for the current question."""
    new_score = {
        'question': state['current_question'],
        'response': state['response'],
        'evaluation': state['technical_score']
    }
    return {'scores': [*state['scores'], new_score]}

def generate_report(state: State):
    """Generates the final interview report."""
    report = [
        f"""
-----------------Interview Report for {state['name']}------------------
Position: {state['applied_role']}
Skills: {', '.join(state['technical_skills'])}\n
-----------------Technical Interview Scores----------------------------
"""
    ]
    if state['plan'] == ['API Error: Failed to generate questions']:
        report.append("Technical interview questions could not be generated due to an API error.\n")
    else:
        for i, score in enumerate(state['scores'], 1):
            report.append(f"{i}. Question: {score['question']}")
            report.append(f"   Answer: {score['response']}")
            report.append(f"   Evaluation: {score['evaluation']}\n")
    
    return {'report': '\n'.join(report)}

def end_interview(state: State):
    """Ends the interview and displays the final report."""
    print("\n" + state['report'])
    print("\nInterview completed. Thank you!")

# Create and configure workflow
workflow = StateGraph(State)

# Add nodes
nodes = [
    ("start", start_interview),
    ("init", initialize_candidate_info),
    ("gen_plan", generate_interview_plan),
    ("check_plan", check_interview_plan),
    ("present_q", present_question),
    ("collect_resp", collect_response),
    ("evaluate", evaluate_technical_response),
    ("calc_score", calculate_score),
    ("gen_report", generate_report),
    ("end", end_interview)
]

for name, func in nodes:
    workflow.add_node(name, func)

# Configure workflow
workflow.set_entry_point("start")
workflow.add_edge("start", "init")
workflow.add_edge("init", "gen_plan")
workflow.add_edge("gen_plan", "check_plan")
workflow.add_edge("present_q", "collect_resp")
workflow.add_edge("collect_resp", "evaluate")
workflow.add_edge("evaluate", "calc_score")
workflow.add_edge("calc_score", "check_plan")
workflow.add_edge("gen_report", "end")
workflow.set_finish_point("end")

workflow.add_conditional_edges(
    "check_plan",
    lambda s: "present_q" if s['status'] == 'Plan Incomplete' else "gen_report",
    {"present_q": "present_q", "gen_report": "gen_report"}
)

# Compile the workflow
interview_flow = workflow.compile()

# Run the workflow
if __name__ == "__main__":
    print("Starting interview workflow...\n")
    try:
        # Increase recursion limit to handle more questions
        interview_flow.invoke({ "messages": [] }, config={"recursion_limit": 100})
    except KeyboardInterrupt:
        print("\nInterview interrupted by user")