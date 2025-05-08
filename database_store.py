import json
import time
import traceback
from typing import Dict, Any

from utils.needs import close_connection_pool, connection_pool, State


def check_candidate_exists(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if a candidate exists in the database."""

    conn = connection_pool.getconn()
    cursor = conn.cursor()
    try:
        name = state.get('name')
        email = state.get('email')
        if not name or not email:
            return {"error": "Name and email are required"}
        cursor.execute("""
            SELECT id FROM candidates
            WHERE name = %s AND email = %s;
        """, (name, email))
        result = cursor.fetchone()

        if result:
            print(f"Candidate exists with ID: {result[0]}")
            # Return both exists flag and id in a consistent format
            return {"exists": True, "id": result[0], "status": "success"}
        print("Candidate does not exist.")
        return {"exists": False, "status": "success"}
    except Exception as e:
        print(f"Error at check_candidate_exists: {str(e)}")
        return {"error": str(e), "status": "error"}
    finally:
        cursor.close()
        connection_pool.putconn(conn)


def create_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new report for a candidate."""

    conn = None
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        
        candidate_id = state.get('id')
        if not candidate_id:
            print("Error: Candidate ID is required")
            return {"error": "Candidate ID is required"}
            
        print(f"Creating report for candidate ID: {candidate_id}")
        cursor.execute("""
            INSERT INTO reports (candidate_id)
            VALUES (%s)
            RETURNING id;
        """, (candidate_id,))
        
        report_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Report created successfully with ID: {report_id}")
        return {"report_id": report_id}
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error creating report: {str(e)}")
        return {"error": str(e)}
    finally:
        if conn:
            cursor.close()
            connection_pool.putconn(conn)

def store_interview_scores(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store interview scores in the database."""

    conn = None
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        
        report_id = state.get('report_id')
        scores = state.get('scores', [])

        if not report_id:
            print("Error: Report ID is required for storing scores")
            return {"error": "Report ID is required"}
            
        if not scores:
            print("Warning: No scores to store")
            return {"status": "success", "message": "No scores to store"}

        print(f"Storing {len(scores)} interview scores for report ID: {report_id}")
        stored_count = 0
        for score in scores:
            try:
                cursor.execute("""
                    INSERT INTO interview_scores (report_id, question, response, evaluation)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """, (
                    report_id,
                    score.get('question'),
                    score.get('response'),
                    score.get('evaluation')
                ))
                score_id = cursor.fetchone()[0]
                stored_count += 1
                print(f"Stored score {stored_count}/{len(scores)} with ID: {score_id}")
            except Exception as e:
                print(f"Error storing individual score: {str(e)}")
                conn.rollback()
                continue

        conn.commit()
        print(f"Successfully stored {stored_count} out of {len(scores)} scores")
        return {"status": "success", "scores_stored": stored_count}
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error storing interview scores: {str(e)}")
        return {"error": str(e)}
    finally:
        if conn:
            cursor.close()
            connection_pool.putconn(conn)

def store_conversation_history(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store conversation history in the database."""

    conn = None
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        
        report_id = state.get('report_id')
        history = state.get('conversation_history', [])

        if not report_id:
            print("Error: Report ID is required to store conversation history")
            return {"error": "Report ID is required"}
            
        if not history:
            print("Warning: No conversation history to store")
            return {"status": "success", "message": "No conversation history to store"}

        print(f"Storing {len(history)} conversation events for report ID: {report_id}")
        stored_count = 0
        for event in history:
            try:
                event_data = json.dumps(event)
                cursor.execute("""
                    INSERT INTO conversation_events (report_id, event_type, event_data, timestamp)
                    VALUES (%s, %s, %s, to_timestamp(%s))
                    RETURNING id;
                """, (
                    report_id,
                    event.get('event_type'),
                    event_data,
                    event.get('timestamp', time.time())
                ))
                event_id = cursor.fetchone()[0]
                stored_count += 1
                print(f"Stored event {stored_count}/{len(history)} with ID: {event_id}")
            except Exception as e:
                print(f"Error storing individual event: {str(e)}")
                conn.rollback()
                continue

        conn.commit()
        print(f"Successfully stored {stored_count} out of {len(history)} conversation events")
        return {"status": "success", "events_stored": stored_count}
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error storing conversation history: {str(e)}")
        return {"error": str(e)}
    finally:
        if conn:
            cursor.close()
            connection_pool.putconn(conn)

def store_interview_data(state: State) -> Dict[str, Any]:
    """Main function to store interview data in the database.

    This function follows a sequential process:
    1. Check if candidate exists
    2. Create a report for the candidate
    3. Store interview scores
    4. Store conversation history

    Args:
        state: The interview state containing all data

    Returns:
        A dictionary with the operation status
    """
    print("\nStarting database operations...")
    if state.get('plan') == ['API Error: Failed to generate questions']:
        print("API error occurred. Report not stored in the database.")
        return {"status": "Error"}

    try:
        # Step 1: Check if candidate exists
        print("Checking if candidate exists...")
        result = check_candidate_exists(state)
        if result.get("error"):
            print(f"Error checking candidate: {result['error']}")
            return result
            
        # Set the candidate ID in state
        if result.get("exists"):
            state["id"] = result["id"]
        else:
            print("Error: Candidate not found")
            return {"status": "Error", "error": "Candidate not found"}

        print(f"Using candidate ID: {state['id']}")

        # Step 2: Create a report
        print("Creating report...")
        result = create_report(state)
        if "error" in result:
            print(f"Error creating report: {result['error']}")
            return result
        state["report_id"] = result["report_id"]
        print(f"Created report with ID: {state['report_id']}")

        # Step 3: Store interview scores
        if state.get("scores"):
            print("Storing interview scores...")
            result = store_interview_scores(state)
            if "error" in result:
                print(f"Error storing scores: {result['error']}")
                return result
            print(f"Stored {result.get('scores_stored', 0)} scores")

        # Step 4: Store conversation history
        if state.get("conversation_history"):
            print("Storing conversation history...")
            result = store_conversation_history(state)
            if "error" in result:
                print(f"Error storing history: {result['error']}")
                return result
            print(f"Stored {result.get('events_stored', 0)} conversation events")

        print("\nDatabase operations completed successfully.")
        return {
            "status": "Success", 
            "candidate_id": state["id"], 
            "report_id": state["report_id"]
        }
    except Exception as e:
        print(f"Error in database operations: {e}")
        traceback.print_exc()
        return {"status": "Error", "error": str(e)}
    finally:
        close_connection_pool()

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
                'data': {'question': "What is your experience with Python?"}
            },
            {
                'event_type': 'response_received',
                'data': {'response': "I have 5 years of experience working with Python in production environments."}
            }
        ]
    }

    print("Starting database operations test...")
    print("\nTest state:", json.dumps(test_state, indent=2))

    try:
        # Execute the database operations
        print("\nExecuting database operations:")
        result = store_interview_data(test_state)
        print("\nFinal Result:", json.dumps(result, indent=2))

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        traceback.print_exc()
    finally:
        print("\nClosing connection pool...")
        print("Test completed.")
