import time
import traceback
from needs import llm, State
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, SystemMessage

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
        current_question = plan[0]
        print(f"\nQ: {current_question}")
        if not state.get('question_refined'):
            return {
                'current_question': current_question,
                'plan': plan,
                'question_refined': False  # Reset the refined flag for the new question
            }
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
            
            if not state.get('question_refined'):
                # Create proper message format for LLM
                messages = [
                    SystemMessage(content="Answer with only 'yes' or 'no'."),
                    HumanMessage(content=f"Did the candidate understand this question: '{state['current_question']}' based on their answer: '{response}'?")
                ]
                
                check = llm.invoke(messages)
                time.sleep(2)
                if check.content.lower().strip() == 'no':
                    return {'refine': True}
            plan = state.get("plan", [])
            plan.pop(0)
            return {
                'response': response,
                'plan': plan
            }
        return {'response': ''}
    except Exception as e:
        print(f"Error in collect_response: {str(e)}")
        return {'status': 'Error'}
    
@tool(args_schema=StateParam)
def refine_question(state: dict) -> dict:
    """Refines the candidate's response to the current question."""
    try:
        if state.get("refine"):
            # Create proper message format for LLM
            messages = [
                SystemMessage(content="You are an expert interviewer. Please refine the following question to make it clearer while keeping it concise and short as possible."),
                HumanMessage(content=f"Question to refine: {state['current_question']}")
            ]
            
            refined = llm.invoke(messages)
            time.sleep(2)
            plan = state.get("plan", [])
            plan[0] = refined.content
            return {
                'plan': plan,
                'refine': False,
                'question_refined': True  # Mark the question as refined
            }
        return {'current_question': state.get('current_question', '')}
    except Exception as e:
        print(f"Error in refine_response: {str(e)}")
        return {'status': 'Error'}

def create_interview_agent(llm):
    """Creates an agent that conducts the interview."""
    tools = [check_interview_plan, present_question, collect_response, refine_question]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an interviewer conducting an interview. The state includes a 'current_step' that tells you what to do:
        
        If current_step is:
        1: You must use check_interview_plan only
        2: You must use present_question only
        3: You must use collect_response only
        4: You must use refine_question only
        
        
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
    """Manages the interview process."""
    working_state = state.copy()
    
    try:
        print(f"\nStarting interview for {working_state.get('name', 'candidate')} ({working_state.get('applied_role', 'unknown role')})...")
        
        # Run through the steps
        i = 1
        while i <= 3:
            if working_state.get('status') == 'Plan Complete':
                break
                
            working_state['current_step'] = i
            result = agent.invoke({
                "input": working_state
            })
            time.sleep(2)
            # Update working state with the result of the agent
            for step in result["intermediate_steps"]:
                if isinstance(step[1], dict):
                    working_state.update(step[1])
            
            
            if working_state.get('refine'):
                working_state['current_step'] = 4
                result = agent.invoke({
                    "input": working_state
                })
                time.sleep(2)
                # Update working state with the result of the agent
                for step in result["intermediate_steps"]:
                    if isinstance(step[1], dict):
                        working_state.update(step[1])
                i =1
            i += 1
            
        return working_state
        
    except Exception as e:
        print(f"Error in start_interview_agent: {str(e)}")
        traceback.print_exc()
        return {"status": "Error", "error": str(e)}

if __name__ == "__main__":
    test_state = {
        'name': 'fawez',
        'applied_role': 'Software Developer',
        'plan': ["What is your greatest strength?", "Why do you want this job?"],
        'current_question': '',
        'response': '',
        'status': 'Plan Incomplete',
        'refine': False,
        'question_refined': False
    }
    start_interview_agent(test_state)