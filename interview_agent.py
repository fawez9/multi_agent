import ast
import json
import time
import traceback
from typing import Callable
from functools import wraps
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from needs import llm, State
class StateParam(BaseModel):
    """Pydantic model for the state parameter."""
    state: dict = Field(..., description="The current interview state")

    @field_validator("state", mode="before")
    def parse_state(cls, value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)  # Convert string to dict
            except:
                try:
                    return json.loads(value.replace("'", "\""))
                except:
                    raise ValueError(f"Cannot parse state: {value}")
        return value  # If already a dict, return as is

def handle_state_conversion(func: Callable) -> Callable:
    """Decorator to handle state conversion for tools."""
    @wraps(func)
    def wrapper(state: dict | str | StateParam, *args, **kwargs) -> dict:
        try:
            # Handle string input
            if isinstance(state, str):
                state_param = StateParam(state=state)
                converted_state = state_param.state
            # Handle dict input wrapped in StateParam
            elif isinstance(state, StateParam):
                converted_state = state.state
            # Handle direct dict input
            elif isinstance(state, dict):
                converted_state = state
            else:
                raise ValueError(f"Unexpected state type: {type(state)}")
            
            return func(converted_state, *args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            traceback.print_exc()
            return {'status': 'Error', 'error': str(e)}
    return wrapper


@tool(args_schema=StateParam)
@handle_state_conversion
def check_interview_plan(state: dict) -> dict:
    """Checks the status of the interview plan."""
    plan = state.get("plan", [])
    if not plan:
        state.update({'status': 'Plan Complete'})
    else:
        state.update({'status': 'Plan Incomplete'})
    return state

@tool(args_schema=StateParam)
@handle_state_conversion
def present_question(state: dict) -> dict:
    """Presents the next question in the interview plan."""

    state['conversation_history'] = state.get('conversation_history', [])
    plan = state.get("plan", [])
    
    # Check if plan is empty
    if not plan:
        state.update({'status': 'Plan Complete'})
        return state
        
    current_question = plan[0]
    
    # Track the question presentation
    question_event = {
        'event_type': 'present_question',
        'question': current_question,
        'is_refined': state['_internal_flags'].get('question_refined', False),
    }
    state['conversation_history'].append(question_event)
    
    print(f"\nQ: {current_question}") #TODO: Replace with speech synthesis (TTS: text-to-speech)
    
    state.update({
        'current_question': current_question,
    })
    
    return state

@tool(args_schema=StateParam)
@handle_state_conversion
def collect_response(state: dict) -> dict:
    """Collects the candidate's response to the current question."""
    state['_internal_flags'] = state.get('_internal_flags', {})
    state['conversation_history'] = state.get('conversation_history', [])
    state['plan'] = state.get('plan', [])
    state['scores'] = state.get('scores',[])
    
    if state.get("current_question"):
        response = input("\nYour answer: ") #TODO: Replace with speech recognition
        
        conversation_event = {
            'event_type': 'collect_response',
            'response': response
        }
        state['conversation_history'].append(conversation_event)
        
        if not state['_internal_flags'].get('question_refined'):
            messages = [
                SystemMessage(content="You are evaluating whether a candidate understood an interview question based on their response. Answer with ONLY 'yes' or 'no'."),
                HumanMessage(content=f"Did the candidate understand this question: '{state['current_question']}' based on their answer: '{response}'?")
            ]
            
            check = llm.invoke(messages)
            time.sleep(1)
            
            if 'no' in check.content.lower():
                state['_internal_flags'].update({
                    'needs_refinement': True,
                    'question_answered': False
                })
                state.update({
                    'response': response
                })
                return state
    
        if not state['_internal_flags'].get('needs_refinement', False):
            if state['plan']:
                state['plan'].pop(0)
            state['_internal_flags']['question_answered'] = True
            
        scores={
            'question': state.get('current_question', ''),
            'response': response, #TODO: make this a list to store even the responses before the refinement
            'evaluation': ''
        }
        state['scores'].append(scores)
        state.update({
            'current_question': '',
            'response': ''
        })
        
        return state
    
    state.update({
        'response': ''
        })
    return state

@tool(args_schema=StateParam)
@handle_state_conversion
def refine_question(state: dict) -> dict:
    """Refines the current question to make it clearer for the candidate."""
    state['_internal_flags'] = state.get('_internal_flags', {})
    state['conversation_history'] = state.get('conversation_history', [])
    state['plan'] = state.get('plan',[])
    
    if state['_internal_flags'].get('needs_refinement', False):
        messages = [
            SystemMessage(content="""
            You are helping to refine an interview question that the candidate didn't understand.
            Please rewrite the question to make it clearer and more specific.
            Keep the refined question concise but add context or examples if helpful.
            Maintain the same topic and difficulty level.
            """),
            HumanMessage(content=f"""
            Original question: {state.get('current_question', '')}
            Candidate's response: {state.get('response', '')}
            
            Please provide only the refined question without any additional text.
            """)
        ]
        
        refined = llm.invoke(messages)
        time.sleep(1)
        
        refined_question = refined.content.strip()
        refinement_event = {
            'event_type': 'refine_question',
            'refined_question': refined_question
        }
        state['conversation_history'].append(refinement_event)
        
        if state['plan']:
            state['plan'][0] = refined_question
        
        state['_internal_flags'].update({
            'needs_refinement': False,
            'question_refined': True
        })
    
    return state

def create_interview_agent(llm):
    """Creates an agent that conducts the interview."""
    tools = [check_interview_plan, present_question, collect_response, refine_question]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are an interviewer conducting an interview with a candidate. Follow these steps precisely:

        Steps:
        1. First, check the interview plan status using check_interview_plan 
        2.If the status is 'Plan Complete', stop immediately.
        3. If the plan is incomplete:
           - Use present_question to show the next question
           - Use collect_response to get the candidate's answer
           - If the response needs refinement, use refine_question and repeat from step 3
        
        Available tools:
        {tools}{tool_names}

        Format your responses exactly like this:
        Thought: I will [explain your next action]
        Action: [tool name]
        Action Input: {{[complete current state]}}
        Observation: [tool result]
        
        Always include the COMPLETE state in Action Input.
        """),
        ("human", "State: {input}"),
        ("assistant", "Thought:{agent_scratchpad}")
    ])
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

agent = create_interview_agent(llm)

def start_interview_agent(state: State):
    """Starts the interview agent."""
    try:
        working_state=state.copy()
        
        print(f"\nStarting interview for {working_state.get('name', 'candidate')} ({working_state.get('applied_role', 'unknown role')})...")
        
        
        # Pass the cleaned state to the agent
        result = agent.invoke({"input": working_state})
        
        time.sleep(1)
        
        # Update working state with all intermediate steps
        if result.get("intermediate_steps"):
            for step in result["intermediate_steps"]:
                if isinstance(step[1], dict):
                    working_state.update(step[1])
        
        # Force the status update based on the plan
        if 'plan' in working_state and not working_state.get('plan'):
            working_state['status'] = 'Plan Complete'
        
        # Update the original state
        if isinstance(state, dict):
            state.clear()
            state.update(working_state)
        else:
            for key, value in working_state.items():
                setattr(state, key, value)
        
        return working_state
        
    except Exception as e:
        print(f"Error in start_interview_agent: {str(e)}")
        return {"status": "Error"}

if __name__ == "__main__":
    test_state = {
        'name': 'John Doe',
        'applied_role': 'Software Developer',
        'plan': ["What is your greatest strength?", "Why do you want this job?"],
        'current_question': '',
        'response': '',
        'status': 'Plan Incomplete',
        '_internal_flags': {
            'needs_refinement': False,
            'question_answered': False,
            'question_refined': False
        },
        'conversation_history': []
    }
    result = start_interview_agent(test_state)
    print("Final state:", result)
