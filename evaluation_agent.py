import ast
from functools import wraps
import time
import json
import traceback
from typing import Callable, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator

from langchain_core.tools import tool
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
    """Decorator to handle state conversion for tools.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function that handles state conversion
    """
    @wraps(func)
    def wrapper(state: Union[Dict[str, Any], str, StateParam], *args, **kwargs) -> Dict[str, Any]:
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
def evaluate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the candidate's response with a concise score and short feedback."""
    try:
        # Get conversation history and scores from state
        conversation_history = state.get('conversation_history', [])
        scores = state.get('scores', [])

        # Get job details with fallbacks if not present
        job_details = state.get('job_details', {})
        job_skills = job_details.get('skills', 'Not specified')
        job_description = job_details.get('description', 'Not specified')

        # Track which scores have been updated
        updated_scores = []

        # Process each score that doesn't have an evaluation yet
        for score in scores:
            # Skip scores that already have evaluations
            if score.get('evaluation'):
                updated_scores.append(score)
                continue

            # Create evaluation prompt
            prompt = f"""
            Evaluate this response concisely based on candidate's profile and these provided infos:
            Question: {score.get('question', 'Not provided')}
            Response: {score.get('response', 'Not provided')}
            Conversation History: {conversation_history}
            Job Requirements: {job_skills} {job_description}
            Give a score (0-10) and a short justification (1-2 sentences).
            """

            # Generate evaluation
            evaluation = rag.generate_response(query=prompt)
            time.sleep(1)  # Prevent rate limiting

            # Extract evaluation text
            evaluation_result = ""
            if hasattr(evaluation, 'content') and evaluation.content:
                evaluation_result = evaluation.content.strip()
            elif isinstance(evaluation, str):
                evaluation_result = evaluation.strip()
            else:
                evaluation_result = str(evaluation).strip()

            # Clean up the evaluation text
            evaluation_result = evaluation_result.replace('```', '').replace('json', '')

            # Update score with evaluation
            score['evaluation'] = evaluation_result
            updated_scores.append(score)

            # Add to conversation history
            evaluation_event = {
                'event_type': 'evaluate_response',
                'question': score.get('question', 'Not provided'),
                'response': score.get('response', 'Not provided'),
                'evaluation': evaluation_result
            }
            conversation_history.append(evaluation_event)

        # Update state
        state.update({
            'scores': updated_scores,
            'conversation_history': conversation_history,
            'status': 'Evaluation Complete'
        })
        return state
    except Exception as e:
        print(f"Error in evaluate_response: {str(e)}")
        traceback.print_exc()
        return {'status': 'Error', 'error': str(e), 'original_state': state}


def create_evaluation_agent():
    """Creates an agent to evaluate candidate interview responses.

    Returns:
        An AgentExecutor configured to evaluate responses
    """
    tools = [evaluate_response]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an evaluation agent responsible for scoring interview responses. Your task is to evaluate each response in the scores list.

        Available tools: {tools}{tool_names}

        Process:
        1. Examine the state to understand the candidate's profile and job requirements
        2. Use evaluate_response to evaluate all responses in the scores list
        3. If the status is 'Evaluation Complete', stop immediately

        Format your responses as:
        Thought: I will analyze the current state and determine next action
        Action: evaluate_response
        Action Input: {{complete_current_state}}
        Observation: [Result of evaluation]

        Important Rules:
        - Always pass the complete state dictionary in Action Input
        - Do not modify the status field directly
        - Ensure all responses are evaluated fairly based on job requirements
        - Consider both technical accuracy and communication skills
        """),
        ("human", "State: {input}"),
        ("assistant", "Thought:{agent_scratchpad}")
    ])


    agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True,max_iterations=1)

agent = create_evaluation_agent()

def start_evaluation_agent(state: Union[State, Dict[str, Any]]) -> Union[State, Dict[str, Any]]:
    """Starts the evaluation agent to evaluate candidate responses.

    Args:
        state: The current interview state containing scores to evaluate

    Returns:
        Updated state with evaluations added to scores
    """
    print("Starting response evaluation...")

    # Check if there are any scores to evaluate
    if not state.get('scores'):
        print("No scores to evaluate, skipping evaluation")
        return state

    # Create a proper copy of the state
    working_state = state.copy() if isinstance(state, dict) else dict(state)

    try:
        # Invoke the evaluation agent
        result = agent.invoke({
            "input": working_state
        })

        time.sleep(1)  # Prevent rate limiting

        # Update working state with all intermediate steps
        if result.get("intermediate_steps"):
            for step in result["intermediate_steps"]:
                if isinstance(step[1], dict) and step[1].get('status') != 'Error':
                    working_state.update(step[1])

        # Update the original state
        if isinstance(state, dict):
            # For dictionary state
            for key, value in working_state.items():
                state[key] = value
        else:
            # For object state
            for key, value in working_state.items():
                if hasattr(state, key):
                    setattr(state, key, value)

        print(f"Evaluation complete. Processed {len(working_state.get('scores', []))} responses.")
        return working_state
    except Exception as e:
        print(f"Evaluation agent processing failed: {str(e)}")
        traceback.print_exc()
        # Return original state to avoid data loss
        return state

if __name__ == "__main__":
    # Create a test state with sample data
    test_state = {
        'name': 'Test Candidate',
        'skills': ['Python', 'JavaScript', 'React'],
        'job_details': {
            'applied_role': 'Software Developer',
            'skills': 'Python, JavaScript, React, Node.js',
            'description': 'Looking for a skilled full-stack developer with experience in modern web technologies.'
        },
        'scores': [
            {
                "question": "What is your experience with Python?",
                "response": "I have 3 years of experience with Python, primarily working on data analysis projects and web applications using Django and Flask. I've also contributed to open-source Python libraries.",
                "evaluation": ""
            },
            {
                "question": "How would you handle a situation where project requirements change frequently?",
                "response": "I would implement an agile approach with regular check-ins with stakeholders. This helps manage changing requirements by breaking work into smaller iterations and getting feedback early.",
                "evaluation": ""
            }
        ],
        'conversation_history': []
    }

    # Run the evaluation agent
    print("\nRunning evaluation agent on test data...\n")
    result_state = start_evaluation_agent(test_state)

    # Display results
    print("\nEvaluation Results:")
    for i, score in enumerate(result_state.get('scores', [])):
        print(f"\nQuestion {i+1}: {score.get('question')}")
        print(f"Response: {score.get('response')}")
        print(f"Evaluation: {score.get('evaluation')}")
