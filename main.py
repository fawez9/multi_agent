import time
import candidate
from core_rag import rag
from langgraph.graph import StateGraph 
from interview_agent import start_interview_agent
from evaluation_agent import start_evaluation_agent
from needs import State, connection_pool ,close_connection_pool


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
        'skills': candidate.skills,
        'name': candidate.name,
        'email': candidate.email,
        'phone': candidate.phone
    }

def generate_interview_plan(state: State):
    """Generates an interview plan based on the candidate's role and skills."""
    role = state.get('applied_role', state['applied_role'])
    skills = state.get('skills')
    
    try:
        # Generate questions using the RAG system
        prompt = f"""
        Based on the job offer and candidate profile, generate 1 interview question for the role of {role} make them as short as possible.
        Focus on the following skills: {skills}.
        Format each question as a numbered item:
        """
        response = rag.generate_response(query=prompt)  # Use the RAG system to generate questions
        time.sleep(2)
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


def generate_report(state: dict):
    """Generates the final interview report."""
    # Generate the report text
    report = [
        f"""
----------------Interview Report for {state['name']}------------------
Position: {state['applied_role']}
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


def store_db(state: dict):
    """Stores the report data in the database if there is no API error."""

    # Check for API error
    if state.get('plan') == ['API Error: Failed to generate questions']:
        print("API error occurred. Report not stored in the database.")
        return

    # Take the connection from the pool
    conn = connection_pool.getconn()
    cursor = conn.cursor()

    try:
        # Insert candidate information
        cursor.execute("""
            INSERT INTO candidates (name, email, phone, role, skills)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            state.get('name'),
            state.get('email'),  # Safely access 'email'
            state.get('phone'),
            state.get('applied_role'),
            state.get('skills')
        ))
        candidate_id = cursor.fetchone()[0]

        # Insert report metadata
        cursor.execute("""
            INSERT INTO reports (candidate_id)
            VALUES (%s)
            RETURNING id;
        """, (candidate_id,))
        report_id = cursor.fetchone()[0]

        # Insert interview scores
        for score in state.get('scores', []):
            cursor.execute("""
                INSERT INTO interview_scores (report_id, question, response, evaluation)
                VALUES (%s, %s, %s, %s);
            """, (
                report_id,
                score.get('question'),
                score.get('response'),
                score.get('evaluation')
            ))

        # Commit the transaction
        conn.commit()
        print("Report stored successfully in the database.")
        return

    except Exception as e:
        # Rollback in case of error
        conn.rollback()
        print(f"Error storing report in the database: {e}")

    finally:
        # Close the connection
        if conn:
            cursor.close()
            connection_pool.putconn(conn)

def end_interview(state: State):
    """Ends the interview and displays the final report."""
    close_connection_pool()
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
    ("evaluation_agent",start_evaluation_agent),
    ("gen_report", generate_report),
    ("store_db", store_db),
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
    lambda s: "evaluation_agent" if s['status'] == 'Plan Incomplete' else "gen_report" if s['status'] == 'Plan Complete' else "end",
    {
        "evaluation_agent": "evaluation_agent" , # Otherwise, continue evaluation
        "gen_report": "gen_report",  # When all questions are done, generate the report
        "end": "end"  # End the interview
    }
)
workflow.add_edge("evaluation_agent", "interview_agent")
workflow.add_edge("gen_report", "store_db")
workflow.add_edge("store_db", "end")
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
