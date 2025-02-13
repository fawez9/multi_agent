from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from needs import llm
from typing import List, Dict, Union

@tool
def present_question(plan: List[str]) -> Dict[str, Union[str, List[str]]]:
    """
    Presents the next question from the plan.
    Returns current question and remaining plan.
    """
    if not plan:
        return {
            'current_question': '',
            'remaining_plan': [],
            'status': 'Plan Complete'
        }
    
    return {
        'current_question': plan[0],
        'remaining_plan': plan[1:],
        'status': 'In Progress'
    }

@tool
def collect_response(response: str) -> Dict[str, str]:
    """
    Collects the candidate's response to the current question.
    Just marks the response as received since actual response comes from chat.
    """
    if response:
        return {
            'status': 'Response Collected'}
    else:
        return {
            'status': 'Response Not Collected'}

@tool
def check_plan(plan: List[str]) -> Dict[str, Union[str, int]]:
    """
    Checks if there are more questions in the plan.
    """
    remaining = len(plan)
    return {
        'status': 'Complete' if remaining == 0 else 'In Progress',
        'remaining_questions': remaining
    }

def create_interview_agent(llm):
    """Creates an agent that conducts the interview."""
    tools = [present_question, collect_response, check_plan]

    prompt = ChatPromptTemplate([
        ("system", """
        You are an interviewer conducting an interview. Use these tools to manage the interview:
        1. present_question - Get the next question from the plan
        2. collect_response - Mark the response as received
        3. check_plan - Check if there are more questions

        Rules:
         -Ask only one question at a time
        -If there's a user response collect it using the collect_response tool
        -Check the Interview Plan between each question and the next question
         
        """),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True  # Capture intermediate steps
    )

def start_interview():
    """Generate initial plan and start the interview."""
    plan = [
        "Explain how you would implement a queue using two stacks. What is the time complexity for enqueue and dequeue operations?",
        "What is the difference between bias and variance in machine learning? How do they affect model performance?",
        "Describe a challenging bug you've encountered and how you solved it."
    ]

    agent = create_interview_agent(llm)

    while plan:
        response = agent.invoke({
            "input": f"Here's the interview plan: {plan}"
        })

        # Debugging: Print the entire response
        #print("Response from agent.invoke():", response)

        # Extract the tool's output directly from intermediate steps
        if 'intermediate_steps' in response:
            for step in response['intermediate_steps']:
                if step[0].tool == "present_question":
                    tool_output = step[1]
                    if isinstance(tool_output, dict):
                        current_question = tool_output.get('current_question', '')
                        remaining_plan = tool_output.get('remaining_plan', [])
                        status = tool_output.get('status', '')

                        print(f"\nQ: {current_question}")
                        user_response = input('Your answer: ')

                        response = agent.invoke({
                            "input": f"here is the User response: {user_response}"
                        })

                        plan = remaining_plan

                        """ check_response = agent.invoke({
                            "input": f"Check if the plan is complete: {plan}"
                        })
                        print("Plan check:", check_response) """

                        if not plan:
                            print("Interview completed. No more questions in the plan.")
                            break
                    else:
                        print("Unexpected tool output format:", tool_output)
                        break
        else:
            print("Unexpected response format:", response)
            break

# Example usage
if __name__ == "__main__":
    start_interview()
