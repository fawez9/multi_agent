import candidate
from needs import State
from core_rag import rag
from langgraph.graph import StateGraph 
from interview_agent import start_interview_agent


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
        'applied_role': candidate.applied_role,
        'technical_skills': candidate.technical_skills,
        'name': candidate.name,
    }

def generate_interview_plan(state: State):
    """Generates an interview plan based on the candidate's role and skills."""
    role = state.get('applied_role', state['applied_role'])
    skills = state.get('technical_skills', state['technical_skills'])
    
    try:
        # Generate questions using the RAG system
        prompt = f"""
        Based on the job offer and candidate profile, generate 2 technical interview questions for the role of {role} make them as short as possible.
        Focus on the following skills: {', '.join(skills)}.
        Format each question as a numbered item:
        """
        response = rag.generate_response(query=prompt)  # Use the RAG system to generate questions
        # Extract only the actual questions, filtering out any explanatory text
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 3)):  # Check if line starts with a number
                questions.append(line.strip())
    except Exception as e:
        print(f"API Error: {str(e)}")
        return {'plan': ['API Error: Failed to generate questions'], 'status': 'Plan Complete'}
    
    return {'plan': questions}

def evaluate_technical_response(state: State):
    """Evaluates the candidate's response to the current question."""
    try:
        if not state.get('current_question') or not state.get('response'):
            return {'technical_score': "Cannot evaluate: missing question or response"}

        prompt = f"""
        Evaluate this technical response based on the job offer and candidate profile make the answer as short as possible.
        Question: {state['current_question']}
        Response: {state['response']}
        Provide a score out of 10.
        """
        evaluation = rag.generate_response(query=prompt)

        # Clear the current_question and response after evaluation
        return {
            'technical_score': evaluation,
            'current_question': state['current_question'],  
            'response': state['response']  
        }
    except Exception as e:
        return {'technical_score': f"Evaluation failed: {str(e)}"}

def calculate_score(state: State):
    """Calculates and stores the candidate's score for the current question."""
    new_score = {
        'question': state['current_question'],
        'response': state['response'],
        'evaluation': state['technical_score']
    }
    return {
        'scores': [*state['scores'], new_score],
        'current_question': '',  # Clear the question
        'response': ''  # Clear the response
    }

def generate_report(state: State):
    """Generates the final interview report."""
    report = [
        f"""
----------------Interview Report for {state['name']}------------------
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

# Define the workflow using StateGraph
workflow = StateGraph(State)

# Add nodes to the workflow
nodes = [
    ("start", start_interview),
    ("init", initialize_candidate_info),
    ("gen_plan", generate_interview_plan),
    ("interview_agent", start_interview_agent),
    ("evaluate", evaluate_technical_response),
    ("calc_score", calculate_score),
    ("gen_report", generate_report),
    ("end", end_interview)
]

for name, func in nodes:
    workflow.add_node(name, func)

# Configure workflow edges
workflow.set_entry_point("start")
workflow.add_edge("start", "init")
workflow.add_edge("init", "gen_plan")
workflow.add_edge("gen_plan", "interview_agent")

# Add conditional edges based on the interview status
workflow.add_conditional_edges(
    "interview_agent",
    lambda s: "evaluate" if s['status'] == 'Plan Incomplete' else "gen_report" if s['status'] == 'Plan Complete' else "end",
    {
        "evaluate": "evaluate" , # Otherwise, continue evaluation
        "gen_report": "gen_report",  # When all questions are done, generate the report
        "end": "end"  # End the interview
    }
)
workflow.add_edge("evaluate", "calc_score")
workflow.add_edge("calc_score", "interview_agent")
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
