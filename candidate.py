import json
import time
from core_rag import rag

#TODO: fetch data from db
with open('knowledge_base/doc2.txt', 'r') as f:
    text = f.read()

prompt = f"""provide infos in json without writing (```json) i want the direct json response
        {{
        "name": "username",
        "applied_role": "role",
        "skills": ["skill1", "skill2", "..."]
        "phone": "phone_number",
        "email": "email"
        }}"""

json_response = rag.generate_response(query=prompt)
time.sleep(2)
# print(json_response)

# Parse the JSON string into a Python dictionary
data = json.loads(json_response)
# Extract information into variables
name = data.get("name")
applied_role = data.get("applied_role")
skills = data.get("skills")
phone = data.get("phone")
email = data.get("email")


if __name__ == "__main__":
    print(f"Name: {name}")
    print(f"Applied Role: {applied_role}")
    print(f"Skills: {skills}")
    print(f"Phone: {phone}")
    print(f"Email: {email}")
