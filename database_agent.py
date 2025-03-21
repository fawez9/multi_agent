import ast
import json
import time
import traceback
from pydantic import BaseModel, Field, field_validator

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

from needs import close_connection_pool, llm, connection_pool, State

class StateParam(BaseModel):
    """Pydantic model for the state parameter."""
    state: dict = Field(..., description="The current interview state")
    @field_validator('state', mode='before')
    @classmethod
    def parse_state(cls, value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"Input should be a valid dictionary")

@tool(args_schema=StateParam)
def check_candidate_exists(state: dict) -> dict:
    """Check if a candidate exists in the database."""
    # Make sure state is a dictionary
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            try:
                state = ast.literal_eval(state)
            except (ValueError, SyntaxError):
                return {"error": "Invalid state format"}
    
    name = state.get('name')
    email = state.get('email')
    
    if not name or not email:
        return {"error": "Name and email are required"}
    
    conn = connection_pool.getconn()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id FROM candidates
            WHERE name = %s AND email = %s;
        """, (name, email))
        result = cursor.fetchone()
        
        # Create a new state dictionary with the candidate_id
        new_state = state.copy()
        if result:
            new_state['candidate_id'] = result[0]
            return {"exists": True, "id": result[0], "state": new_state}
        return {"exists": False, "state": new_state}
    finally:
        cursor.close()
        connection_pool.putconn(conn)

@tool(args_schema=StateParam)
def create_candidate(state: dict) -> dict:
    """Create a new candidate in the database."""
    # Make sure state is a dictionary
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            try:
                state = ast.literal_eval(state)
            except (ValueError, SyntaxError):
                return {"error": "Invalid state format"}
    
    name = state.get('name')
    email = state.get('email')
    phone = state.get('phone')
    role = state.get('applied_role')  # Note: Using 'applied_role' to match original function
    skills = state.get('skills')
    
    if not name or not email:
        return {"error": "Name and email are required"}
    
    conn = connection_pool.getconn()
    cursor = conn.cursor()
    try:
        # Make sure the column names match your table structure
        cursor.execute("""
            INSERT INTO candidates (name, email, phone, role, skills)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (name, email, phone, role, skills))
        candidate_id = cursor.fetchone()[0]
        conn.commit()
        return {"id": candidate_id}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        connection_pool.putconn(conn)

@tool(args_schema=StateParam)
def create_report(state: dict) -> dict:
    """Create a new report for a candidate."""
    # Make sure state is a dictionary
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            try:
                state = ast.literal_eval(state)
            except (ValueError, SyntaxError):
                return {"error": "Invalid state format"}
    
    candidate_id = state.get('candidate_id')
    
    if not candidate_id:
        return {"error": "Candidate ID is required"}
    
    conn = connection_pool.getconn()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO reports (candidate_id)
            VALUES (%s)
            RETURNING id;
        """, (candidate_id,))
        report_id = cursor.fetchone()[0]
        conn.commit()
        
        # Return updated state with report_id
        new_state = state.copy()
        new_state['report_id'] = report_id
        return {"report_id": report_id, "state": new_state}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        connection_pool.putconn(conn)

@tool(args_schema=StateParam)
def store_interview_scores(state: dict) -> dict:
    """Store interview scores in the database."""
    # Make sure state is a dictionary
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            try:
                state = ast.literal_eval(state)
            except (ValueError, SyntaxError):
                return {"error": "Invalid state format"}
    
    report_id = state.get('report_id')
    scores = state.get('scores', [])
    
    if not report_id:
        return {"error": "Report ID is required"}
    
    conn = connection_pool.getconn()
    cursor = conn.cursor()
    try:
        for score in scores:
            cursor.execute("""
                INSERT INTO interview_scores (report_id, question, response, evaluation)
                VALUES (%s, %s, %s, %s);
            """, (
                report_id,
                score.get('question'),
                score.get('response'),
                score.get('evaluation')
            ))
        conn.commit()
        return {"status": "success", "scores_stored": len(scores)}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        connection_pool.putconn(conn)

@tool(args_schema=StateParam)
def store_conversation_history(state: dict) -> dict:
    """Store conversation history in the database."""
    # Make sure state is a dictionary
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            try:
                state = ast.literal_eval(state)
            except (ValueError, SyntaxError):
                return {"error": "Invalid state format"}
    
    report_id = state.get('report_id')
    history = state.get('conversation_history', [])
    
    if not report_id:
        return {"error": "Report ID is required"}
    
    conn = connection_pool.getconn()
    cursor = conn.cursor()
    try:
        for event in history:
            event_data = json.dumps(event)
            cursor.execute("""
                INSERT INTO conversation_events (report_id, event_type, event_data, timestamp)
                VALUES (%s, %s, %s, to_timestamp(%s));
            """, (
                report_id,
                event.get('event_type'),
                event_data,
                event.get('timestamp', time.time())
            ))
        conn.commit()
        return {"status": "success", "events_stored": len(history)}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        connection_pool.putconn(conn)

def create_db_agent():
    """Creates an agent to handle database operations."""
    tools = [
        check_candidate_exists,
        create_candidate,
        create_report,
        store_interview_scores,
        store_conversation_history
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a database operations agent responsible for storing interview data.
        Follow these steps precisely:
         {tools}{tool_names}

        1. Check if candidate exists using check_candidate_exists tool with the email from the state
        2. If exists, use the returned state (which includes candidate_id) for the next steps
        3. If not exists, create candidate using create_candidate tool with the full state
        4. Create a new report using create_report tool with the state containing candidate_id
        5. Store interview scores using store_interview_scores tool with the report ID and scores
        6. Store conversation history using store_conversation_history tool with the report ID and history

        IMPORTANT: 
        - Always use the state returned from the previous tool for the next operation
        - Make sure to pass the entire state object as a dictionary, not as a string
        - When a tool returns a new state, use that state for the next action

        Format your response as:
        Thought: analyze the current state and decide next action
        Action: the tool to use
        Action Input: the state dictionary
        Observation: the result from the tool
        ... (repeat until all data is stored)
        Final Answer: summary of all operations performed
        """),
        ("human", "State: {input}"),
        ("assistant", "Thought: {agent_scratchpad}")
    ])

    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

agent = create_db_agent()
def start_db_agent(state: State):
    """Main function to store interview data using the DB agent."""
    if state.get('plan') == ['API Error: Failed to generate questions']:
        print("API error occurred. Report not stored in the database.")
        return {"status": "error", "message": "API error occurred. Report not stored in the database."}

    try:
        
        result = agent.invoke({"input": state})
        print("\nDatabase operations completed successfully.")
        close_connection_pool()
        return result
    except Exception as e:
        print(f"Error in database operations: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Test data
    test_state = {
        'name': 'John Doe',
        'email': 'john.doe@example.com',
        'phone': '+1234567890',
        'applied_role': 'Senior Python Developer',
        'skills': ['Python', 'SQL', 'AWS'],
        'scores': [
            {
                'question': "What is your experience with Python?",
                'response': "I have 5 years of experience working with Python in production environments.",
                'evaluation': "Strong understanding of Python demonstrated. Score: 8/10"
            },
            {
                'question': "Describe a challenging project you worked on.",
                'response': "I led the migration of a monolithic application to microservices.",
                'evaluation': "Shows good project experience and leadership. Score: 9/10"
            }
        ],
        'conversation_history': [
            {
                'event_type': 'question_asked',
                'timestamp': time.time(),
                'data': {'question': "What is your experience with Python?"}
            },
            {
                'event_type': 'response_received',
                'timestamp': time.time(),
                'data': {'response': "I have 5 years of experience working with Python in production environments."}
            }
        ]
    }

    print("Starting database agent test...")
    print("\nTest state:", json.dumps(test_state, indent=2))
    
    try:
        # Let the agent handle all database operations
        print("\nExecuting database agent workflow:")
        result = start_db_agent(test_state)
        print("\nFinal Result:", json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        traceback.print_exc()
    finally:
        print("\nClosing connection pool...")
        connection_pool.closeall()
        print("Test completed.")
