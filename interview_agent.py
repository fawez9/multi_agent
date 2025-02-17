from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from needs import llm, State
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import time

# Define the schema for the `state` parameter
class StateParam(BaseModel):
    state: dict = Field(..., description="The current interview state")

# Update the tools with the schema
@tool(args_schema=StateParam)
def check_interview_plan(state: dict):
    """Checks the status of the interview plan."""
    plan = state.get("plan", [])
    is_complete = not plan or len(plan) == 0
    return {'status': 'Plan Complete' if is_complete else 'Plan Incomplete'}

@tool(args_schema=StateParam)
def present_question(state: dict):
    """Presents the next question in the interview plan."""
    plan = state.get("plan", [])
    
    if not plan:
        return {'current_question': '', 'status': 'Plan Complete', 'plan': []}

    current_question = plan.pop(0)
    return {'current_question': current_question, 'plan': plan, 'status': 'Plan Incomplete' if plan else 'Plan Complete'}

@tool(args_schema=StateParam)
def collect_response(state: dict):
    """Collects the candidate's response to the current question."""
    if state.get("current_question"):
        print(f"\nQ: {state['current_question']}")
        response = input("Your answer: ")
        return {'response': response}
    return {'response': ''}

# (Remaining code is unchanged)


def create_interview_agent(llm):
    """Creates an agent that conducts the interview."""
    tools = [check_interview_plan, present_question, collect_response]
    prompt = ChatPromptTemplate([
        ("system", """
         You are an interviewer conducting an interview. Your task is to ask questions from the plan and collect responses.
         
         You have these tools to manage the interview:
            1. present_question - Get the next question from the plan
            2. collect_response - Get the candidate's response to the current question
            3. check_interview_plan - Check if there are more questions
         
         Important rules:
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
# Create the interview agent
agent = create_interview_agent(llm)
def start_interview_agent(state: State):
    """Manages the interview process."""
    working_state = state.copy()
    try:
        print(f"\nStarting interview for {working_state.get('name', 'candidate')} ({working_state.get('applied_role', 'unknown role')})...")

        # Present the next question
        response = agent.invoke({
            "input": f"Here is the interview state: {working_state}."
        })
        time.sleep(2)  # Add a delay for better readability
        # Process the response
        if isinstance(response, dict):
            steps = response.get("intermediate_steps", [])
            for step in steps:
                if isinstance(step, tuple) and len(step) >= 2:
                    tool_result = step[1]
                    if isinstance(tool_result, dict):
                        # Update the working state with the tool result
                        for key, value in tool_result.items():
                            if key in working_state or key == 'status':
                                working_state[key] = value

            # Collect the response if a question is presented
            if working_state.get('current_question') and not working_state.get('response'):
                collect_result = agent.invoke({
                    "input": f"Please collect the response for the following question. Here is the state: {working_state}"
                })
                time.sleep(2)

                # Update state with collected response
                if isinstance(collect_result, dict):
                    steps = collect_result.get("intermediate_steps", [])
                    for step in steps:
                        if isinstance(step, tuple) and len(step) >= 2:
                            tool_result = step[1]
                            if isinstance(tool_result, dict) and 'response' in tool_result:
                                working_state['response'] = tool_result['response']


            # Return the updated state
            return working_state

        return {"status": "Error", "message": "Unexpected response format from agent"}

    except Exception as e:
        print(f"Error in start_interview_agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "Error", "error": str(e)}

if __name__ == "__main__":
    test_state = {
        'name': 'fawez',
        'applied_role': 'Software Developer',
        'plan': ["What is your name?", "What is your greatest strength?", "Why do you want this job?"],
        'status': 'Plan Incomplete'
    }
    start_interview_agent(test_state)