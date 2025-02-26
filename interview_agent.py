import ast
import json
import time
import traceback
from needs import llm, State
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

class StateParam(BaseModel):
    state: dict = Field(..., description="The current interview state")
    @field_validator('state', mode='before')
    @classmethod
    def parse_state(cls, value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"Input should be a valid dictionary")
@tool(args_schema=StateParam)
def check_interview_plan(state: dict) -> dict:
    """Checks the status of the interview plan."""
    try:
                 # Ensure state is a dictionary
        if isinstance(state, str):
            state = ast.literal_eval(state)
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
                 # Ensure state is a dictionary
        if isinstance(state, str):
            state = ast.literal_eval(state)
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
                 # Ensure state is a dictionary
        if isinstance(state, str):
            state = ast.literal_eval(state)
        if state.get("current_question"):
            response = input("\nYour answer: ")
            
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
                'plan': plan,
                'question_answered':True
            }
        return {'response': ''}
    except Exception as e:
        print(f"Error in collect_response: {str(e)}")
        return {'status': 'Error'}
    
@tool(args_schema=StateParam)
def refine_question(state: dict) -> dict:
    """Refines the candidate's response to the current question."""
    try:
                 # Ensure state is a dictionary
        if isinstance(state, str):
            state = ast.literal_eval(state)
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
        ("system", """You are an interviewer conducting an interview to a candidate. You have the following tools:

        {tools}

        Use the following format:

        State: the state you are going to understand (python dictionary)
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: you should rethink about what to do next
        Final Answer: the final action to the original input state

         important rules:
         - You should always check the plan first to see if it is complete or not and if the question is answered or not
         - if the plan is not complete  and "question_answered" is False you should present the next question
         - if the plan is not complete and "question_answered" is False you should collect the response from the candidate
         - if the plan is not complete and "question_answered" is False and the candidate did not understand the question you should refine the question
         - If "question_answered" is True, you should stop and return the current state even if the plan is not complete

         

        Begin!"""),
        ("human", "State: {input}"),
        ("assistant", "Thought:{agent_scratchpad}")
    ])
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True)

agent = create_interview_agent(llm)
def start_interview_agent(state: State):
    """Manages the interview process."""
    working_state = state.copy()
    
    try:
        print(f"\nStarting interview for {working_state.get('name', 'candidate')} ({working_state.get('applied_role', 'unknown role')})...")
        
        # Run through the steps
        result = agent.invoke({
            "input": working_state
        })
        
        time.sleep(2)
        
        # Update working state with the result of the agent
        for step in result["intermediate_steps"]:
            if isinstance(step[1], dict):
                working_state.update(step[1])
            
        # Update the original state with all changes
        if hasattr(state, 'update'):
            state.update(working_state)
            
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
        'question_refined': False,
        'question_answered': False
    }
    start_interview_agent(test_state)
    # print(test_state)