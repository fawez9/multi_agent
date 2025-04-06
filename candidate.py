import json
import time
from core_rag import rag
from needs import engine


def get_candidate_basic_info():
    """Get basic candidate information (name, phone, email) from the resume"""
    prompt = """
    1. What is the candidate's name?
    2. What is the candidate's phone?
    3. What is the candidate's email?
    """
    return rag.generate_response(prompt)


def get_candidate_skills(basic_info):
    """Get candidate's technical skills based on their resume"""
    prompt = f"""
    1. What are the candidate's technical skills?
    take these infos in consideration {basic_info}
    return json format with keys
    {{
        "name": "",
        "phone": "",
        "email": "",
        "skills": []
    }}
    """
    return rag.generate_response(prompt)


def parse_json_response(json_response):
    """Clean and parse the JSON response from the LLM"""
    # Clean the response by removing markdown code block markers
    cleaned_response = json_response.replace('```json', '').replace('```', '').strip()
    print("Raw response:", cleaned_response)

    # Parse the JSON string into a Python dictionary
    return json.loads(cleaned_response)


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
            SELECT j.title, j.skills, j.description
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
                        "description": result[2]}
            else:
                return {"applied_role": "Software Developer",
                        "skills": ["Python", "JavaScript"],
                        "description": "Software development position"}
    except Exception as e:
        print(f"Error fetching job details: {e}")
        return {"applied_role": "Software Developer",
                "skills": ["Python", "JavaScript"],
                "description": "Software development position"}


def process_candidate_info():
    """Main function to process candidate information"""
    # Get basic candidate information
    basic_info = get_candidate_basic_info()
    time.sleep(2)  # Prevent rate limiting

    # Get candidate skills
    json_response = get_candidate_skills(basic_info)
    time.sleep(2)  # Prevent rate limiting

    # Parse the response
    data = parse_json_response(json_response)

    # Extract information into variables
    name = data.get("name", "Unknown")
    phone = data.get("phone", "Unknown")
    email = data.get("email", "Unknown")
    skills = data.get("skills", [])

    # Get job details
    job_details = get_job_details(name)

    return {
        "name": name,
        "phone": phone,
        "email": email,
        "skills": skills,
        "job_details": job_details
    }


if __name__ == "__main__":
    candidate_info = process_candidate_info()

    print(f"Name: {candidate_info['name']}")
    print(f"Applied Role: {candidate_info['job_details']['applied_role']}")
    print(f"Job Skills: {candidate_info['job_details']['skills']}")
    print(f"Candidate Skills: {candidate_info['skills']}")
    print(f"Phone: {candidate_info['phone']}")
    print(f"Email: {candidate_info['email']}")
    print(f"Job Description: {candidate_info['job_details']['description']}")



