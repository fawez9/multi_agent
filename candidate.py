import json
import time
from core_rag import rag

with open('knowledge_base/doc2.txt', 'r') as f:
    text = f.read()

prompt = f"""
        Extract the following  and return ONLY the following information in this format:
        {{
            "name": "candidate name",
            "applied_role": "role they're applying for",
            "technical_skills": ["skill1", "skill2", ...]
        }}
        rules: please dont write the json word at the beginning
        """

json_response = rag.generate_response(query=prompt)
time.sleep(2)
# print(json_response)

# Parse the JSON string into a Python dictionary
data = json.loads(json_response)
# Extract information into variables
name = data.get("name")
applied_role = data.get("applied_role")
technical_skills = data.get("technical_skills")


print(f"Name: {name}")
print(f"Applied Role: {applied_role}")
print(f"Technical Skills: {technical_skills}")
