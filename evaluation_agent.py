import time
import traceback
from needs import llm, State
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import  AIMessage
from core_rag import rag



class StateParam(BaseModel):
    state: dict = Field(..., description="The current interview state")

@tool(args_schema=StateParam)
def evaluate_technical_response(state: dict) -> dict:
    """Evaluates the candidate's response with a concise score and short feedback."""
    try:
        if not state.get('current_question') or not state.get('response'):
            return {'technical_score': "Cannot evaluate: missing question or response"}
        
        prompt = f"""
        Evaluate this response concisely:
        Question: {state['current_question']}
        Response: {state['response']}
        Give a score (0-10) and a short justification (1-2 sentences).
        """
        
        evaluation = rag.generate_response(query=prompt)
        time.sleep(2)
        
        return {
            'technical_score': evaluation.content.strip() if isinstance(evaluation, AIMessage) else str(evaluation).strip(),
            'evaluated': True
        }
    except Exception as e:
        print(f"Error in evaluate_technical_response: {str(e)}")
        traceback.print_exc()
        return {'technical_score': "Evaluation failed"}

@tool(args_schema=StateParam)
def calculate_score(state: dict) -> dict:
    """Calculates and stores the candidate's concise score."""
    try:
        if state.get('evaluated'):
            new_score = {
                'question': state.get('current_question', 'Unknown question'),
                'response': state.get('response', 'No response'),
                'evaluation': state.get('technical_score', 'No score')
            }
            scores = state.get('scores', []) or []
            return {
                'scores': scores + [new_score],
                'current_question': '',
                'response': '',
            }
        return state
    except Exception as e:
        print(f"Error in calculate_score: {str(e)}")
        traceback.print_exc()
        return state

def process_state(state):
    """Processes the state and ensures concise evaluation."""
    try:
        state = state.copy() if hasattr(state, 'copy') else dict(state)
        state.setdefault('scores', [])
        state.setdefault('evaluated', False)
        
        if not state.get('evaluated'):
            state.update(evaluate_technical_response({"state": state}))
            time.sleep(2)
        
        if state.get('evaluated'):
            state.update(calculate_score({"state": state}))
        
        return state
    except Exception as e:
        print(f"Error in process_state: {str(e)}")
        traceback.print_exc()
        return {"status": "Error", "error": str(e)}

def create_evaluation_agent():
    tools = [evaluate_technical_response, calculate_score]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an interview evaluator with these tools: {tools} {tool_names}"),
        ("human", "Evaluate this state concisely: {input}"),
        ("assistant", "I'll provide a brief evaluation."),
        ("assistant", "{agent_scratchpad}"),
    ])
    agent = create_react_agent(tools=tools, prompt=prompt, llm=llm)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

agent = create_evaluation_agent()

def start_evaluation_agent(state: State):
    print("Starting concise evaluation")
    try:
        return process_state(state)
    except Exception as e:
        print(f"Direct processing failed: {str(e)}")
        try:
            state = state.copy() if hasattr(state, 'copy') else dict(state)
            state.setdefault('scores', [])
            
            result = agent.invoke({"input": state})
            time.sleep(2)
            for step in result["intermediate_steps"]:
                if isinstance(step[1], dict):
                    state.update(step[1])
            
            return state
        except Exception as e:
            print(f"Agent processing also failed: {str(e)}")
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
    result = start_evaluation_agent(test_state)
    print("Final result:", result['scores'])
