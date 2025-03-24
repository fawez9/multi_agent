import ast
from functools import wraps
import time
import json
import traceback
from typing import Callable
from pydantic import BaseModel, Field, field_validator

from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

from core_rag import rag
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
def evaluate_response(state: dict) -> dict:
    """Evaluates the candidate's response with a concise score and short feedback."""
    try:
        
        conversation_history = state.get('conversation_history', [])
        scores=state.get('scores', [])
        for score in scores:

            prompt = f"""
            Evaluate this response concisely based on the knowledge base and these provided infos:
            Question: {score['question']}
            Response: {score['response']}
            Give a score (0-10) and a short justification (1-2 sentences).
            """
        
            evaluation = rag.generate_response(query=prompt)
            time.sleep(0.5)
        
            evaluation_result = evaluation.content.strip() if isinstance(evaluation, AIMessage) else str(evaluation).strip()
            score['evaluation'] = evaluation_result
        
            evaluation_event = {
                'event_type': 'evaluate_response',
                'evaluation': evaluation_result
            }
            conversation_history.append(evaluation_event)
        state.update({
            'scores': scores,
            'conversation_history': conversation_history,
        })
        return state
    except Exception as e:
        print(f"Error in evaluate_response: {str(e)}")
        return {'status': 'Error', 'error': str(e)}


def create_evaluation_agent():
    """Creates an agent to evaluate a candidate's response."""
    tools = [evaluate_response]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an evaluation agent responsible for scoring interview responses. Your task is to evaluate each response in the scores list.

        Available tools: {tools}{tool_names}

        Process:
        1. Check if there are any unevaluated responses in the scores list
        2. Use evaluate_response to score each response
        3. Stop when all responses have been evaluated

        Format your responses as:
        Thought: I will analyze the current state and determine next action
        Action: evaluate_response
        Action Input: {{complete_current_state}}
        Observation: [Result of evaluation]
        
        Rules:
        - Always pass the complete state dictionary in Action Input
        - Only use the evaluate_response tool
        - Stop when all responses have evaluations
        - Do not modify the status field

        Final Answer: Summarize the evaluations completed
        """),
        ("human", "State: {input}"),
        ("assistant", "Thought:{agent_scratchpad}")
    ])
    

    agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

agent = create_evaluation_agent()

def start_evaluation_agent(state: State):
    """Starts the evaluation agent."""
    print("Starting concise evaluation")
    
    # Create a proper copy of the state
    working_state = state.copy() if hasattr(state, 'copy') else state.copy() if isinstance(state, dict) else state
    
    try:   
        result = agent.invoke({
            "input": working_state
        })
        
        time.sleep(2)
        
        # Extract the final state from the last observation if available
        final_state = None
        if result.get("intermediate_steps") and result["intermediate_steps"]:
            final_step = result["intermediate_steps"][-1]
            if isinstance(final_step[1], dict):
                final_state = final_step[1]
        
        # If we have a valid final state, use it
        if final_state and isinstance(final_state, dict):
            if isinstance(working_state, dict):
                working_state = final_state
            else:
                # Handle case where working_state is a State object
                for key, value in final_state.items():
                    setattr(working_state, key, value)
        else:
            # Otherwise update incrementally
            for step in result["intermediate_steps"]:
                if isinstance(step[1], dict):
                    if isinstance(working_state, dict):
                        working_state.update(step[1])
                    else:
                        # Handle case where working_state is a State object
                        for key, value in step[1].items():
                            setattr(working_state, key, value)

        # Update the original state with all changes
        if hasattr(state, 'update'):
            state.update(working_state)
        elif isinstance(state, dict) and isinstance(working_state, dict):
            state.update(working_state)
        
        return working_state
    except Exception as e:
        print(f"Agent processing failed: {str(e)}")
        traceback.print_exc()
        return {"status": "Error", "error": str(e)}

if __name__ == "__main__":
    test_state = {
        'scores': [{"question":"What is your experience with Python?","response":"I have 3 years of experience with Python.","evaluation":""},{"question":"What is your experience with mobile app development?","response":"i developed a lot of apps.","evaluation":""}],
        'conversation_history': []
    }
    test_state = start_evaluation_agent(test_state)
    print("Final result:", test_state.get('scores', []))
