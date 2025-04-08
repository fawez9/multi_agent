import json
import time
import traceback
from typing import Dict, Any

from needs import close_connection_pool, connection_pool, State


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
            return {"exists": True, "id": result[0]}
        return {"exists": False}
    except Exception as e:
        print(f"Error at check_candidate_exists: {str(e)}")
        return {"error": str(e)}
    finally:
        cursor.close()
        connection_pool.putconn(conn)


def create_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new report for a candidate."""

    conn = connection_pool.getconn()
    cursor = conn.cursor()
    try:
        candidate_id = state.get('id')

        if not candidate_id:
            return {"error": "Candidate ID is required"}
        cursor.execute("""
            INSERT INTO reports (candidate_id)
            VALUES (%s)
            RETURNING id;
        """, (candidate_id,))
        report_id = cursor.fetchone()[0]
        conn.commit()

        return {"report_id": report_id}
    except Exception as e:
        conn.rollback()
        print(f"Error at create_report: {str(e)}")
        return {"error": str(e)}
    finally:
        cursor.close()
        connection_pool.putconn(conn)

def store_interview_scores(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store interview scores in the database."""

    conn = connection_pool.getconn()
    cursor = conn.cursor()
    try:
        report_id = state.get('report_id')
        scores = state.get('scores', [])

        if not report_id:
            return {"error": "Report ID is required"}
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
        return {"status": "success"}
    except Exception as e:
        conn.rollback()
        print(f"Error at store_interview_scores: {str(e)}")
        return {"error": str(e)}
    finally:
        cursor.close()
        connection_pool.putconn(conn)

def store_conversation_history(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store conversation history in the database."""

    conn = connection_pool.getconn()
    cursor = conn.cursor()
    try:
        report_id = state.get('report_id')
        history = state.get('conversation_history', [])

        if not report_id:
            return {"error": "Report ID is required"}
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
        return {"status": "success"}
    except Exception as e:
        conn.rollback()
        print(f"Error at store_conversation_history: {str(e)}")
        return {"error": str(e)}
    finally:
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
    if state.get('plan') == ['API Error: Failed to generate questions']:
        print("API error occurred. Report not stored in the database.")
        return {"status": "Error"}

    try:
        # Step 1: Check if candidate exists
        result = check_candidate_exists(state)
        if "error" in result:
            return result

        # Step 3: Create a report
        result = create_report(state)
        if "error" in result:
            return result
        state["report_id"] = result["report_id"]

        # Step 4: Store interview scores
        if state.get("scores"):
            result = store_interview_scores(state)
            if "error" in result:
                return result

        # Step 5: Store conversation history
        if state.get("conversation_history"):
            result = store_conversation_history(state)
            if "error" in result:
                return result

        print("\nDatabase operations completed successfully.")
        return {"status": "Success", "candidate_id": state["id"], "report_id": state["report_id"]}
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
