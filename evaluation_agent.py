import time
import traceback
from needs import llm, State
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from typing import Any
import ast
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage
from core_rag import rag


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
def evaluate_response(state: dict) -> dict:
    """Evaluates the candidate's response with a concise score and short feedback."""
    try:
         # Ensure state is a dictionary
        if isinstance(state, str):
            state = ast.literal_eval(state)
        prompt = f"""
        Evaluate this response concisely:
        Question: {state.get('current_question')}
        Response: {state.get('response')}
        Give a score (0-10) and a short justification (1-2 sentences).
        """
        
        evaluation = rag.generate_response(query=prompt)
        time.sleep(2)
        
        return {
            'technical_score': evaluation.content.strip() if isinstance(evaluation, AIMessage) else str(evaluation).strip(),
            'current_question': state.get('current_question', ''),
            'response': state.get('response', ''),
            'evaluated': True
        }
    except Exception as e:
        print(f"Error in evaluate_response: {str(e)}")
        traceback.print_exc()
        return {'technical_score': "Evaluation failed"}

@tool(args_schema=StateParam)
def calculate_score(state: dict) -> dict:
    """Calculates and stores the candidate's concise score."""
    try:
         # Ensure state is a dictionary
        if isinstance(state, str):
            state = ast.literal_eval(state)
        if state.get('evaluated'):
            new_score = {
            'question': state.get('current_question', ''),
            'response': state.get('response', ''),
            'evaluation': state.get('technical_score', '')
        }
            return {
                'scores': [*state['scores'], new_score],
                'current_question': '',
                'response': '',
                'technical_score': '',
            }
        return state
    except Exception as e:
        print(f"Error in calculate_score: {str(e)}")
        traceback.print_exc()
        return state


def create_evaluation_agent():
    tools = [evaluate_response, calculate_score]
    
    # Fixed: Use proper syntax for ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are an evaluator of an interview. You have the following tools:  

        {tools}  

        ### **Format:**  
        State: (current state as a Python dictionary)  
        Thought: Analyze the state and decide what to do next  
        Action: The action to take, must be one of [{tool_names}]  
        Action Input: The input to the action  
        Observation: The result of the action, which updates the state  
        Thought: Reevaluate based on the updated state and decide the next step  
        ... (repeat until termination)  
        Final Answer: The final decision based on the evaluated state  

        ### **Important Rules:**  

        1. **Never modify the status.** this is for the interview completion not for the evaluation. 

        2. **Always determine which tool to execute based on the state.** If an action is required, select the appropriate tool and execute it.    

        3. **Use only the provided tools.** Do not invent new actions or modify the state outside of tool operations.  

        4. **Once the evaluation reaches a conclusion, stop immediately and return the final state.**

        5. **Make sure to give the  tools the whole state , to ensure that the state is fully updated correctly.**.

        """),
        ("human", "State: {input}"),
        ("assistant", "Thought:{agent_scratchpad}")
    ])
    

    agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

agent = create_evaluation_agent()

def start_evaluation_agent(state: State):
    print("Starting concise evaluation")
    
    # Create a proper copy of the state
    working_state = state.copy()
    
    try:   
        result = agent.invoke({
            "input": working_state
        })
        
        time.sleep(2)
        
        # Update state with results
        for step in result["intermediate_steps"]:
            if isinstance(step[1], dict):
                working_state.update(step[1])


        # Update the original state with all changes
        if hasattr(state, 'update'):
            state.update(working_state)
        
        return working_state
    except Exception as e:
        print(f"Agent processing failed: {str(e)}")
        traceback.print_exc()
        return {"status": "Error", "error": str(e)}

if __name__ == "__main__":
    test_state = {
        'current_question': "What is your experience with Python?",
        'response': "I have 3 years of experience with Python.",
        'scores': [],
        'technical_score': '',
        'evaluated': False
    }
    test_state=start_evaluation_agent(test_state)
    print("Final result:", test_state.get('scores', []))