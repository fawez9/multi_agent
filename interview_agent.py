from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from needs import llm, State
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union

# Define the schema for the `state` parameter
class StateParam(BaseModel):
    state: dict = Field(..., description="The current interview state")

# Update the tools with the schema
@tool(args_schema=StateParam)
def check_interview_plan(state: dict):
    """Checks the status of the interview plan."""
    try:
        plan = state.get("plan", [])
        is_complete = not plan or len(plan) == 0
        return {'status': 'Plan Complete' if is_complete else 'Plan Incomplete'}
    except Exception as e:
        print(f"Error in check_interview_plan: {str(e)}")
        return {'status': 'Error', 'error': str(e)}

@tool(args_schema=StateParam)
def present_question(state: dict):
    """Presents the next question in the interview plan."""
    try:
        plan = state.get("plan", [])
        status = state.get("status", "Plan Incomplete")
        
        if not plan:
            return {'current_question': '', 'status': 'Plan Complete', 'plan': []}
        
        current_question = plan[0]
        remaining_plan = plan[1:]  # Remove the current question from the plan
        
        return {
            'current_question': current_question,
            'plan': remaining_plan,
            'status': status
        }
    except Exception as e:
        print(f"Error in present_question: {str(e)}")
        return {'current_question': '', 'status': 'Error', 'error': str(e), 'plan': []}

@tool(args_schema=StateParam)
def collect_response(state: dict):
    """Collects the candidate's response to the current question."""
    try:
        current_question = state.get("current_question", "")
        
        if current_question:
            print(f"\nQ: {current_question}")
            response = input("Your answer: ")
            return {'response': response}
        return {'response': ''}
    except Exception as e:
        print(f"Error in collect_response: {str(e)}")
        return {'response': '', 'status': 'Error', 'error': str(e)}

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

def start_interview_agent(state: State):
    """Manages the interview process."""    
    agent = create_interview_agent(llm)
    
    # Make a copy of the state to avoid modifying the original
    working_state = state.copy()
    
    try:
        print(f"\nStarting interview for {working_state.get('name', 'candidate')} ({working_state.get('applied_role', 'unknown role')})...")
        
        # Initialize missing values with defaults if not present
        if 'current_question' not in working_state:
            working_state['current_question'] = ""
        if 'response' not in working_state:
            working_state['response'] = ""
        if 'status' not in working_state or working_state['status'] == 'started':
            working_state['status'] = 'Plan Incomplete'
        
        # First, check if we have any questions
        if not working_state.get('plan', []):
            return {"status": "Plan Complete"}
        
        # Check if we're done
        agent.invoke({
            "input": f"check interview plan status: {working_state}"
        })
        
        # Get first question
        response = agent.invoke({
            "input": f"Here is the interview state: {working_state}."
        })
        
        # Process the response
        if isinstance(response, dict):
            # Extract information from intermediate steps
            steps = response.get("intermediate_steps", [])
            for step in steps:
                if isinstance(step, tuple) and len(step) >= 2:
                    tool_result = step[1]
                    if isinstance(tool_result, dict):
                        # Update our working state with the tool result
                        for key, value in tool_result.items():
                            if key in working_state or key == 'status':
                                working_state[key] = value
            
            # Check for the current question and collect a response if needed
            if working_state.get('current_question') and not working_state.get('response'):
                collect_result = agent.invoke({
                    "input": f"Please collect the response for the following question. Here is the state: {working_state}"
                })
                
                # Update state with collected response
                if isinstance(collect_result, dict):
                    steps = collect_result.get("intermediate_steps", [])
                    for step in steps:
                        if isinstance(step, tuple) and len(step) >= 2:
                            tool_result = step[1]
                            if isinstance(tool_result, dict) and 'response' in tool_result:
                                working_state['response'] = tool_result['response']
            
            # Check if we're done
            if working_state.get('status') == 'Plan Complete' or not working_state.get('plan'):
                return {"status": "Plan Complete"}
            
            # Return the updated state
            return working_state
                
        return {"status": "Error", "message": "Unexpected response format from agent"}

    except Exception as e:
        print(f"Error in start_interview_agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "Error", "error": str(e)}

# Run the test
if __name__ == "__main__":
    test_state = {
        'name': 'fawez',
        'applied_role': 'Software Developer',
        'plan': ["What is your name?", "What is your greatest strength?", "Why do you want this job?"],
        'status': 'Plan Incomplete'
    }
    start_interview_agent(test_state)