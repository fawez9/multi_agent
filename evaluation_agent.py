import ast
import time
import json
import traceback
from core_rag import rag
from needs import llm, State
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from langchain.agents import AgentExecutor, create_react_agent


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
        
        conversation_history = state.get('conversation_history', [])
        
        
        prompt = f"""
        Evaluate this response concisely:
        Question: {state.get('current_question')}
        Response: {state.get('response')}
        Give a score (0-10) and a short justification (1-2 sentences).
        """
        
        evaluation = rag.generate_response(query=prompt)
        time.sleep(2)
        
        evaluation_result = evaluation.content.strip() if isinstance(evaluation, AIMessage) else str(evaluation).strip()
        
        evaluation_event = {
            'event_type': 'evaluate_response',
            'evaluation': evaluation_result
        }
        conversation_history.append(evaluation_event)
        
        return {
            'technical_score': evaluation_result,
            'current_question': state.get('current_question', ''),
            'response': state.get('response', ''),
            'evaluated': True,
            'conversation_history': conversation_history
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
        
        conversation_history = state.get('conversation_history', [])
        
        # Check if the evaluated flag is True or if technical_score is not empty
        if state.get('evaluated', False) or state.get('technical_score', ''):
            new_score = {
                'question': state.get('current_question', ''),
                'response': state.get('response', ''),
                'evaluation': state.get('technical_score', '')
            }
            
            calculate_score_event = {
                'event_type': 'calculate_score',
                'score': new_score
            }
            
            conversation_history.append(calculate_score_event)
            
            # Return updated state with new score added and fields reset
            return {
                'scores': [*state.get('scores', []), new_score],
                'current_question': '',
                'response': '',
                'technical_score': '',
                'evaluated': False,  # Reset the evaluated flag
                'conversation_history': conversation_history
            }
        
        # If not evaluated, add error to conversation history
        conversation_history.append({
            'event_type': 'calculate_score',
            'error': 'Response not evaluated'
        })
        
        return {'conversation_history': conversation_history}
    except Exception as e:
        print(f"Error in calculate_score: {str(e)}")
        traceback.print_exc()
        return state


def create_evaluation_agent():
    tools = [evaluate_response, calculate_score]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are an evaluator of an interview. You have the following tools:  

        {tools}  

        ### **Format:**  
        State: (current state as a Python dictionary)  
        Thought: Analyze the state and decide what to do next  
        Action: The action to take, must be one of [{tool_names}]  
        Action Input: {{(the complete state as a Python dictionary) }}
        Observation: The result of the action, which updates the state  
        Thought: Reevaluate based on the updated state and decide the next step  
        ... (repeat until termination)  
        Final Answer: The final decision based on the evaluated state  

        ### **Important Rules:**  
         
        1. **Make sure to pass the whole state dictionary in the Action Input**. To avoid any data loss.

        2. **Never modify the status.** This is for the interview completion not for the evaluation. 

        3. **Always determine which tool to execute based on the state.** If an action is required, select the appropriate tool and execute it.    

        4. **Use only the provided tools.** Do not invent new actions or modify the state outside of tool operations.  

        5. **Make sure the scores are stored in the state.** The scores should be stored in the "scores" key of the state it is the most important field.
         
        6. **Once the evaluation reaches a conclusion, stop immediately and return the final state.**


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
        'current_question': "What is your experience with Python?",
        'response': "I have 3 years of experience with Python.",
        'scores': [],
        'technical_score': '',
        'evaluated': False,
        'conversation_history': []
    }
    test_state = start_evaluation_agent(test_state)
    print("Final result:", test_state.get('scores', []))