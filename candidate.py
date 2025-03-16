import json
import time
from core_rag import rag

prompt = """
1. What is the candidate's name?
2. What is the candidate's phone?
3. What is the candidate's email?
"""

res1 = rag.generate_response(prompt)
time.sleep(2)

prompt = f"""
1. What are the candidate's technical skills?
take these infos in consideration {res1}
return json format with keys 
{{
    "name": "",
    "phone": "",
    "email": "",
    "skills": []
}}
"""
json_response = rag.generate_response(prompt) #TODO: find a way to minimize these calls
time.sleep(2)


# Clean the response by removing markdown code block markers
cleaned_response = json_response.replace('```json', '').replace('```', '').strip()
print("Raw response:", cleaned_response)

# Parse the JSON string into a Python dictionary
data = json.loads(cleaned_response)
# Extract information into variables
name = data.get("name", "Unknown")
applied_role = "Software engineer"
skills = data.get("skills", [])
phone = data.get("phone", "Unknown")
email = data.get("email", "Unknown")

if __name__ == "__main__":
    print(f"Name: {name}")
    print(f"Applied Role: {applied_role}")
    print(f"Skills: {skills}")
    print(f"Phone: {phone}")
    print(f"Email: {email}")


