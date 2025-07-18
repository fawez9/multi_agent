import json
import time
from typing import Dict, Union, List, Tuple
from rag.core_rag import rag
from utils.needs import engine

def get_candidate_details(candidate_name):
    """Fetch candidate details from the database based on candidate name"""
    try:
        with engine.connect() as conn:
            cursor = conn.connection.cursor()

            query = "SELECT name, phone, email, skills FROM candidates WHERE name = %s;"
            cursor.execute(query, (candidate_name,))
            result = cursor.fetchone()

            if result:
                return {
                    "name": result[0],
                    "phone": result[1],
                    "email": result[2],
                    "skills": result[3]
                }
            else:
                return None
    except Exception as e:
        print(f"Error fetching candidate details: {e}")
        return None


def get_job_details(candidate_name):
    """Fetch job details from the database based on candidate name"""
    try:
        with engine.connect() as conn:
            cursor = conn.connection.cursor()

            # First check if the candidate exists
            check_query = "SELECT id FROM candidates WHERE name = %s;"
            cursor.execute(check_query, (candidate_name,))
            candidate_result = cursor.fetchone()

            if not candidate_result:
                return {"applied_role": "Software Developer",
                        "skills": ["Python", "JavaScript"],
                        "description": "Software development position"}

            # Get the candidate's session and job details
            query = """
            SELECT j.title, j.skills, j.description,j.company
            FROM job_offers j
            JOIN sessions s ON j.id = s.applied_role_id
            JOIN candidates c ON c.id = s.candidate_id
            WHERE c.name = %s;
            """
            cursor.execute(query, (candidate_name,))
            result = cursor.fetchone()

            if result:
                return {"applied_role": result[0],
                        "skills": result[1],
                        "description": result[2],
                        "company": result[3]}
    except Exception as e:
        print(f"Error fetching job details: {e}")
        return {"applied_role": "unknown",
                "skills": [],
                "description": "unknown",
                "company": "unknown"}

# NOTE : this is supposed to be on the backend , return candidate id that's the input for the process_document function on the core_rag to process CVs
def create_candidate(name: str, email: str, phone: str = "Unknown",role: str = "Unknown", skills: List[str] = None) -> Tuple[int, bool]:
    """Create a new candidate in the database if they don't exist.

    Args:
        name: Candidate's name
        email: Candidate's email
        phone: Candidate's phone number (optional)
        skills: List of candidate's skills (optional)

    Returns:
        Tuple of (candidate_id, is_new) where is_new indicates if a new candidate was created
    """
    if skills is None:
        skills = []

    with engine.connect() as conn:
        cursor = conn.connection.cursor()

        # Check if candidate exists
        cursor.execute(
            """SELECT id FROM candidates WHERE name = %s AND email = %s AND phone = %s;""",
            (name, email,phone)
        )
        result = cursor.fetchone()

        if result:
            # Candidate exists
            return result[0], False

        # Create new candidate
        cursor.execute(
            """INSERT INTO candidates (name, email, phone,role, skills)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;""",
            (name, email, phone, role,json.dumps(skills))
        )
        candidate_id = cursor.fetchone()[0]
        conn.connection.commit()

        print(f"Created new candidate: {name} (ID: {candidate_id})")
        return candidate_id, True


def process_candidate_info() -> Dict[str, Union[str, List[str], Dict[str, Union[str, List[str]]], int]]:
    """Main function to process candidate information

    Args:
        create_db_entry: Whether to create a database entry for the candidate

    Returns:
        Dictionary containing candidate information and optionally candidate_id
    """
    # Get candidate details from the database
    response = rag.generate_response("What is the candidate's name?")
    data = get_candidate_details(response)

    # Extract information into variables
    name = data.get("name", "Unknown")
    phone = data.get("phone", "Unknown")
    email = data.get("email", "Unknown")
    skills = data.get("skills", [])
    role = data.get("role", "Unknown")

    # Get job details
    job_details = get_job_details(name)

    result = {
        "name": name,
        "phone": phone,
        "email": email,
        "role": role,
        "skills": skills,
        "job_details": job_details
    }

    return result


if __name__ == "__main__":
    # Process candidate info and create database entry
    candidate_info = process_candidate_info()

    print(f"Name: {candidate_info['name']}")
    print(f"Applied Role: {candidate_info['job_details']['applied_role']}")
    print(f"Job Skills: {candidate_info['job_details']['skills']}")
    print(f"Candidate Skills: {candidate_info['skills']}")
    print(f"Phone: {candidate_info['phone']}")
    print(f"Email: {candidate_info['email']}")
    print(f"Job Description: {candidate_info['job_details']['description']}")

    if 'candidate_id' in candidate_info:
        print(f"Candidate ID: {candidate_info['candidate_id']}")
        print(f"Is New Candidate: {candidate_info['is_new_candidate']}")

    #NOTE : the job offer has the needed skills so no need for candidate skills here
    # Example of direct candidate creation
    # candidate_id, is_new = create_candidate(
    #     name="Jane Smith",
    #     email="jane.smith@example.com",
    #     phone="+1-555-123-4567",
    #     skills=["Python", "Machine Learning", "Data Science"]
    # )
    # print(f"Created candidate with ID: {candidate_id}, Is New: {is_new}")
