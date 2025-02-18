from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from needs import llm, State
from pydantic import BaseModel, Field
import time

class StateParam(BaseModel):
    state: dict = Field(..., description="The current interview state")

@tool(args_schema=StateParam)
def check_interview_plan(state: dict) -> dict:
    """Checks the status of the interview plan."""
    try:
        plan = state.get("plan", [])
        if not plan:
            return {'status': 'Plan Complete'}
        return {'status': 'Plan Incomplete'}
    except Exception as e:
        print(f"Error in check_interview_plan: {str(e)}")
        return {'status': 'Error'}

@tool(args_schema=StateParam)
def present_question(state: dict) -> dict:
    """Presents the next question in the interview plan."""
    try:
        plan = state.get("plan", [])
        current_question = plan.pop(0)
        print(f"\nQ: {current_question}")
        return {
            'current_question': current_question,
            'plan': plan
        }
    except Exception as e:
        print(f"Error in present_question: {str(e)}")
        return {'status': 'Error'}

@tool(args_schema=StateParam)
def collect_response(state: dict) -> dict:
    """Collects the candidate's response to the current question."""
    try:
        if state.get("current_question"):
            response = input("Your answer: ")
            return {
                'response': response
            }
        return {'response': ''}
    except Exception as e:
        print(f"Error in collect_response: {str(e)}")
        return {'status': 'Error'}

def create_interview_agent(llm):
    """Creates an agent that conducts the interview."""
    tools = [check_interview_plan, present_question, collect_response]
    prompt = ChatPromptTemplate([
        ("system", """
        You are an interviewer conducting an interview. The state includes a 'current_step' that tells you what to do:
        
        If current_step is:
        1: You must use check_interview_plan only
        2: You must use present_question only
        3: You must use collect_response only
        
        Follow the step exactly - use only the tool specified for the current step.
        When using a tool, pass the entire state object.
        """),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

agent = create_interview_agent(llm)
def start_interview_agent(state: State):
    """Manages the interview process with exactly three steps."""
    working_state = state.copy()
    # print('Initial state:', working_state)
    
    try:
        print(f"\nStarting interview for {working_state.get('name', 'candidate')} ({working_state.get('applied_role', 'unknown role')})...")
        
        # Run through the three steps
        for i in range(1, 4):
            if working_state.get('status') == 'Plan Complete':
                break
            working_state['current_step'] = i
            
            result = agent.invoke({
                "input": working_state
            })
            
            # Update working state with any changes
            for step in result["intermediate_steps"]:
                if isinstance(step[1], dict):
                    working_state.update(step[1])
            
            #print(f"State after step {i}:", working_state)
            time.sleep(2)
            
        return working_state
        
    except Exception as e:
        print(f"Error in start_interview_agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "Error", "error": str(e)}

if __name__ == "__main__":
    test_state = {
        'name': 'fawez',
        'applied_role': 'Software Developer',
        'plan': ["What is your greatest strength?", "Why do you want this job?"],
        'current_question': '',
        'response': '',
        'status': 'Plan Incomplete'
    }
    start_interview_agent(test_state)