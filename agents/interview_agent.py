# Standard library imports
import ast
import json
import time
import traceback
from typing import Callable, Dict, Any, Union
from functools import wraps

# Third-party imports
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# Local imports
from utils.stt_tts import text_to_speech_and_play
from utils.needs import llm, State
from utils.shared_state import shared_state


class StateParam(BaseModel):
    """Pydantic model for the state parameter.

    This model handles conversion between string and dictionary representations
    of the interview state, which is necessary for the LangChain tools interface.
    """
    state: Dict[str, Any] = Field(..., description="The current interview state")

    @field_validator("state", mode="before")
    def parse_state(cls, value):
        """Parse the state from various formats into a dictionary."""
        if isinstance(value, str):
            try:
                # First try to evaluate as a Python literal
                return ast.literal_eval(value)  # Convert string to dict
            except Exception:
                try:
                    # Then try to parse as JSON (replacing single quotes with double quotes)
                    return json.loads(value.replace("'", "\""))
                except Exception:
                    raise ValueError(f"Cannot parse state: {value}")
        return value  # If already a dict, return as is

def handle_state_conversion(func: Callable) -> Callable:
    """Decorator to handle state conversion for interview agent tools.

    This decorator ensures that regardless of how the state is passed to a tool
    (as a string, dict, or StateParam object), it's properly converted to a
    dictionary before being processed by the tool function.

    Args:
        func: The tool function that operates on the state

    Returns:
        A wrapped function that handles state conversion and error handling
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

            # Initialize internal flags if they don't exist
            if '_internal_flags' not in converted_state:
                converted_state['_internal_flags'] = {}

            return func(converted_state, *args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            traceback.print_exc()
            return {'status': 'Error', 'error': str(e)}
    return wrapper


@tool(args_schema=StateParam)
@handle_state_conversion
def check_interview_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """Checks the status of the interview plan.

    This tool examines the current plan to determine if all questions have been asked.
    If the plan is empty, it marks the interview as complete and handles the final
    messaging. If questions remain, it marks the status as incomplete.

    Args:
        state: The current interview state

    Returns:
        Updated interview state with status information
    """
    plan = state.get("plan", [])

    if not plan:
        # No more questions in the plan
        state.update({'status': 'Plan Complete'})
        state['_internal_flags']['interview_complete'] = True

        # If thank you message has been presented, mark this as the final check
        if state['_internal_flags'].get('thank_you_presented', False):
            state['_internal_flags']['final_check'] = True
            print("\nInterview completed successfully.")

            # Show completion message if not already shown
            if not state['_internal_flags'].get('completion_message_shown', False):
                print("\nAll questions have been asked and responses collected.")
                print("The interview process has ended.")
                state['_internal_flags']['completion_message_shown'] = True
    else:
        # There are still questions to ask
        state.update({'status': 'Plan Incomplete'})

    return state

@tool(args_schema=StateParam)
@handle_state_conversion
def present_question(state: Dict[str, Any]) -> Dict[str, Any]:
    #BUG: tts refined question are not played
    """Presents the next question in the interview plan.

    This tool either presents the next question from the plan or delivers a thank you
    message if the interview is complete. It handles tracking the conversation history
    and manages the state transitions between questions.

    Args:
        state: The current interview state

    Returns:
        Updated interview state with the current question information
    """
    # Initialize conversation history if it doesn't exist
    state['conversation_history'] = state.get('conversation_history', [])
    plan = state.get("plan", [])

    # Check if the interview is already complete
    if state['_internal_flags'].get('interview_complete', False):
        # Present a thank you message if this is the first time after completion
        if not state['_internal_flags'].get('thank_you_presented', False):
            thank_you_message = f"Thank you, {state.get('name', 'candidate')}, for participating in this interview. We appreciate your time and responses."

            # Add the thank you message to shared state messages
            shared_state.add_message("assistant", thank_you_message)

            # Mark the interview as complete in shared state
            shared_state.mark_interview_complete()

            text_to_speech_and_play(thank_you_message)

            # Mark that the thank you message has been presented
            state['_internal_flags']['thank_you_presented'] = True

            # Track the thank you message in conversation history
            end_event = {
                'event_type': 'interview_end',
                'message': thank_you_message
            }
            state['conversation_history'].append(end_event)
        return state

    # Check if plan is empty
    if not plan:
        state.update({'status': 'Plan Complete'})
        state['_internal_flags']['interview_complete'] = True
        return state

    # Get the current question from the plan
    current_question = plan[0]

    # Check if this is a new question (not a refined version of the current question)
    is_new_question = state.get('current_question', '') != current_question and not state['_internal_flags'].get('question_refined', False)
    
    if is_new_question:
        print("\nProcessing new question...")
        # Reset refinement flags for new questions
        state['_internal_flags']['question_refined'] = False
        state['_internal_flags']['needs_refinement'] = False
        
        # Add regular message for new questions
        shared_state.add_message("assistant", current_question)
        text_to_speech_and_play(current_question)
    else:
        # This is either a refined question or a repeat of current question
        if state['_internal_flags'].get('question_refined', False):
            print("\nPresenting refined question...")
            # For refined questions, always play TTS with prefix
            prefixed_question = f"Let me rephrase: {current_question}"
            # Note: UI message is already added by refine_question
            text_to_speech_and_play(prefixed_question)

    # Track the question presentation in conversation history
    question_event = {
        'event_type': 'present_question',
        'question': current_question,
        'is_refined': state['_internal_flags'].get('question_refined', False),
    }
    state['conversation_history'].append(question_event)
    
    # Add a short delay to ensure UI updates
    time.sleep(0.5)

    # Update the current question in the state regardless
    state.update({
        'current_question': current_question,
    })

    return state

# Modified collect_response function to work with shared state
@tool(args_schema=StateParam)
@handle_state_conversion
def collect_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Collects the candidate's response to the current question."""
    # Initialize required state fields
    state['conversation_history'] = state.get('conversation_history', [])
    state['plan'] = state.get('plan', [])
    state['scores'] = state.get('scores', [])

    # Check if the interview is already complete
    if state['_internal_flags'].get('interview_complete', False):
        return state  # Nothing to do if interview is complete

    # Only collect response if there's a current question
    if state.get("current_question"):
        # Wait for user input using shared state
        print("Waiting for user response...")
        response = shared_state.get_user_response(timeout=60)

        # Handle empty or very short responses
        if response is None or len(str(response).strip()) < 2:
            print("\nMoving to the next question due to empty or timeout response...")
            # Mark as answered and move to next question
            if state['plan']:
                state['plan'].pop(0)
            state['_internal_flags']['question_answered'] = True

            # Record the empty response in scores
            scores = {
                'question': state.get('current_question', ''),
                'response': '[No response provided]',
                'evaluation': 'Empty response'
            }
            state['scores'].append(scores)
            state.update({
                'current_question': '',
                'response': ''
            })
            return state

        # Record the response in conversation history
        conversation_event = {
            'event_type': 'collect_response',
            'response': response
        }
        state['conversation_history'].append(conversation_event)

        # Only check for refinement if the question hasn't been refined yet

        if not state['_internal_flags'].get('question_refined'):

            # Use LLM to evaluate if the candidate understood the question
            messages = [
                SystemMessage(content="You are evaluating whether a candidate understood an interview question based on their response. Answer with ONLY 'yes' or 'no'."),
                HumanMessage(content=f"Did the candidate understand this question: '{state['current_question']}' based on their answer: '{response}'?")
            ]

            # Add a small delay to prevent rate limiting
            time.sleep(0.5)

            try:
                # Get the evaluation with retry logic
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        check = llm.invoke(messages)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Error checking response understanding (attempt {attempt+1}): {str(e)}. Retrying...")
                            time.sleep(1.5 * (attempt + 1))  # Exponential backoff
                        else:
                            print(f"Failed to check response after {max_retries} attempts: {str(e)}")
                            # Default to accepting the response if we can't check it
                            check_content = "yes"
                            check = type('obj', (object,), {'content': check_content})
                            break

                # Add a small delay to prevent rate limiting
                time.sleep(0.5)
            except Exception as e:
                print(f"Unexpected error in collect_response: {str(e)}")
                # Default to accepting the response
                check_content = "yes"
                check = type('obj', (object,), {'content': check_content})

            if 'no' in check.content.lower():
                # Mark that the question needs refinement
                state['_internal_flags'].update({
                    'needs_refinement': True,
                    'question_answered': False
                })
                state.update({'response': response})
                return state
        else:
            # If the question has already been refined once, accept any response
            print("\nQuestion was already refined once. Accepting response and moving on.")

        # If no refinement is needed or we're accepting the response anyway
        if not state['_internal_flags'].get('needs_refinement', False):
            if state['plan']:
                state['plan'].pop(0)  # Remove the current question from the plan
                # Reset refinement flags for the next question
                state['_internal_flags']['question_refined'] = False
            state['_internal_flags']['question_answered'] = True

        # Record the response in scores
        scores = {
            'question': state.get('current_question', ''),
            'response': response,
            'facial_analysis': shared_state.get_facial_analysis(),
            'evaluation': ''  # TODO: Add evaluation of response quality
        }
        state['scores'].append(scores)

        # Clear current question and response
        state.update({
            'current_question': '',
            'response': ''
        })

        return state

    # No current question, just clear the response
    state.update({'response': ''})
    return state

@tool(args_schema=StateParam)
@handle_state_conversion
def refine_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """Refines the current question to make it clearer for the candidate.

    This tool uses the LLM to rewrite a question that the candidate didn't understand,
    making it clearer while maintaining the same topic and difficulty level. It handles
    the one-time refinement policy by tracking whether a question has already been refined.

    Args:
        state: The current interview state

    Returns:
        Updated interview state with the refined question
    """
    # Initialize required state fields
    state['conversation_history'] = state.get('conversation_history', [])
    state['plan'] = state.get('plan', [])

    # Only refine if refinement is needed and the question hasn't been refined yet
    if state['_internal_flags'].get('needs_refinement', False) and not state['_internal_flags'].get('question_refined', False):
        print("\nRefining question for better clarity...")

        # Get the current question to refine
        current_question = state.get('current_question', '')

        # Prepare the prompt for the LLM to refine the question
        messages = [
            SystemMessage(content="""
            You are helping to refine an interview question that the candidate didn't understand.
            Please rewrite the question to make it clearer and more specific.
            Keep the refined question concise but add context or examples if helpful.
            Maintain the same topic and difficulty level.
            """),
            HumanMessage(content=f"""
            Original question: {current_question}
            Candidate's response: {state.get('response', '')}

            Please provide only the refined question without any additional text.
            """)
        ]

        # Add a small delay to prevent rate limiting
        time.sleep(0.5)

        try:
            # Get the refined question from the LLM with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    refined = llm.invoke(messages)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Error refining question (attempt {attempt+1}): {str(e)}. Retrying...")
                        time.sleep(2 * (attempt + 1))  # Exponential backoff
                    else:
                        print(f"Failed to refine question after {max_retries} attempts: {str(e)}")
                        # Provide a simple refinement as fallback
                        refined_content = f"Let me rephrase: {current_question}. Could you please answer this question?"
                        refined = type('obj', (object,), {'content': refined_content})
                        break

            # Add a small delay to prevent rate limiting
            time.sleep(0.5)
        except Exception as e:
            print(f"Unexpected error in refine_question: {str(e)}")
            # Provide a simple refinement as fallback
            refined_content = f"Let me rephrase: {current_question}. Could you please answer this question?"
            refined = type('obj', (object,), {'content': refined_content})

        # Extract and record the refined question
        refined_question = refined.content.strip()
        
        # Create a refinement event for conversation history
        refinement_event = {
            'event_type': 'refine_question',
            'original_question': current_question,
            'refined_question': refined_question
        }
        state['conversation_history'].append(refinement_event)

        # Format the refined question with a prefix
        prefixed_refined_question = f"Let me rephrase: {refined_question}"
        
        # Add the refined question as a new message (not updating the old one)
        shared_state.add_refined_message("assistant", prefixed_refined_question, current_question)
        print(f"Added refined question to the chat: '{prefixed_refined_question}'")

        # Replace the current question in the plan with the refined version
        if state['plan']:
            # Store the refined question without prefix in the plan
            state['plan'][0] = refined_question
            print(f"\nUpdated plan with refined question: {refined_question}")

        # Update flags to indicate the question has been refined
        state['_internal_flags'].update({
            'needs_refinement': False,
            'question_refined': True
        })
        print("\nMarked question as refined in state")
        
        # Add a short pause to ensure the UI has time to refresh
        time.sleep(1)
        
    elif state['_internal_flags'].get('question_refined', False) and state['_internal_flags'].get('needs_refinement', False):
        # If the question has already been refined once, move to the next question
        print("\nQuestion was already refined once. Moving to the next question.")

        if state['plan']:
            state['plan'].pop(0)  # Remove the current question

        # Reset flags for the next question
        state['_internal_flags'].update({
            'needs_refinement': False,
            'question_refined': False,  # Reset for the next question
            'question_answered': True   # Mark as answered so we move on
        })

        # Clear current question and response
        state.update({
            'current_question': '',
            'response': ''
        })

    return state

def create_interview_agent(llm):
    """Creates an agent that conducts an interview.

    This function creates a LangChain agent with the necessary tools and prompt
    to conduct an interview. The agent follows a specific protocol for asking questions,
    collecting responses, and refining questions when needed.

    Args:
        llm: The language model to use for the agent

    Returns:
        An AgentExecutor that can conduct the interview
    """
    # Define the tools available to the agent
    tools = [check_interview_plan, present_question, collect_response, refine_question]

    # Create the prompt template with detailed instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are an interviewer conducting an interview with a candidate. Follow these steps precisely in this exact order:

        Steps:
        1. ALWAYS start by checking the interview plan status using check_interview_plan
        2. If the status is 'Plan Complete':
           - Use present_question ONCE to end the interview gracefully with a thank you message
           - Then use check_interview_plan as your FINAL action to end the interview
           - After the final check, STOP - do not take any more actions
        3. If the status is 'Plan Incomplete':
           - Use present_question to show the next question
           - Use collect_response to get the candidate's answer
           - If the response indicates the candidate didn't understand AND the question hasn't been refined yet,
             use refine_question ONCE and then repeat present_question and collect_response
           - If the question has already been refined once, accept any response and move to the next question
           - After collecting a response, ALWAYS use check_interview_plan to check if the plan is now complete

        Available tools:
        {tools}{tool_names}

        Format your responses exactly like this:
        Thought: I will [explain your next action]
        Action: [tool name]
        Action Input: {{[complete current state]}}
        Observation: [tool result]

        Always include the COMPLETE state in Action Input.

        IMPORTANT RULES:
        - ALWAYS start with check_interview_plan
        - After each response is collected, use check_interview_plan to verify the plan status
        - Each question can be refined AT MOST ONCE - if a question has already been refined, accept any response
        - When the plan is complete (empty plan array), use present_question ONCE to deliver the thank you message
        - After delivering the thank you message, use check_interview_plan as your FINAL action
        - When you see 'Interview completed successfully' in the output, the interview is over - do not take any more actions
        - Handle empty responses gracefully by moving to the next question
        - Do not try to use 'None' as a tool - always use one of the available tools
        """),
        ("human", "State: {input}"),
        ("assistant", "Thought:{agent_scratchpad}")
    ])

    # Create the agent and executor
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
        # max_iterations is set in start_interview_agent
    )

# Create the interview agent
agent = create_interview_agent(llm)

def start_interview_agent(state: Union[State, Dict[str, Any]]):
    """Starts the interview agent with the given state.

    This function initializes and runs the interview agent with the provided state.
    It calculates the appropriate number of iterations based on the number of questions,
    handles the agent execution, and updates the state with the results.

    Args:
        state: The initial interview state, either as a State object or dictionary

    Returns:
        The final interview state after completion
    """
    try:
        # Create a working copy of the state, only including essential fields
        # This reduces memory usage by not carrying around large conversation histories
        essential_fields = [
            'name', 'plan', 'status', 'nb_questions', 'current_question',
            'response', '_internal_flags', 'job_details'
        ]

        # Create a minimal working state with only essential fields
        working_state = {}
        source_state = state.copy() if hasattr(state, 'copy') else state.copy()

        for field in essential_fields:
            if field in source_state:
                working_state[field] = source_state[field]

        # Ensure we have the conversation_history and scores arrays, but don't copy all their contents
        working_state['conversation_history'] = source_state.get('conversation_history', [])[-5:] if 'conversation_history' in source_state else []
        working_state['scores'] = source_state.get('scores', [])

        # Print interview start message
        print(f"\nStarting interview for {working_state.get('name', 'candidate')} ...")

        # Calculate max_iterations based on the number of questions and expected tool calls
        nb_questions = working_state.get('nb_questions', 2)

        # Assuming at most one refinement per question:
        max_calls_per_question = 6  # 3 base + 3 for refinement

        # Add check_interview_plan calls at the beginning and end
        initial_and_final_checks = 2

        # Add extra calls for handling the end of the interview (1 for thank you message)
        end_handling = 1

        # Calculate the base number of iterations needed
        base_iterations = (nb_questions * max_calls_per_question) + initial_and_final_checks + end_handling

        # Set max_iterations with a small buffer to prevent premature termination
        max_iterations = base_iterations + 2

        # Add retry logic for agent invocation
        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Run the agent with the calculated max_iterations
                result = agent.invoke({"input": working_state}, max_iterations=max_iterations)

                # If we get here, the invocation succeeded
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"Error in agent invocation (attempt {attempt+1}): {str(e)}. Retrying...")
                    # Reduce state size further for retry
                    if 'conversation_history' in working_state:
                        working_state['conversation_history'] = working_state['conversation_history'][-2:] if working_state['conversation_history'] else []
                    time.sleep(2 * (attempt + 1))  # Exponential backoff

        # If all retries failed, raise the last error
        if last_error is not None:
            raise last_error

        # Brief pause to allow any pending output to complete
        time.sleep(0.5)

        # Update working state with results from all intermediate steps
        if result.get("intermediate_steps"):
            for step in result["intermediate_steps"]:
                if isinstance(step[1], dict):
                    working_state.update(step[1])

        # Ensure the status is consistent with the plan
        if 'plan' in working_state and not working_state.get('plan'):
            working_state['status'] = 'Plan Complete'

        # Make sure we preserve the scores from the working state
        if 'scores' in working_state:
            source_state['scores'] = working_state['scores']

        # Update the original state object with only the essential updated fields
        # This prevents overwriting fields we didn't include in the working state
        update_fields = [
            'status', 'current_question', 'response', '_internal_flags',
            'plan', 'scores'
        ]

        if isinstance(state, dict):
            for field in update_fields:
                if field in working_state:
                    state[field] = working_state[field]
        else:
            # If it's a State object, update its attributes
            for field in update_fields:
                if field in working_state:
                    setattr(state, field, working_state[field])

        return working_state

    except Exception as e:
        print(f"Error in start_interview_agent: {str(e)}")
        traceback.print_exc()

        # Create a minimal error state that won't cause further issues
        error_state = {
            "status": "Error",
            "error": str(e),
            "plan": [],  # Empty plan to signal completion
            "_internal_flags": {"interview_complete": True}
        }

        # Update the original state to prevent further processing
        if isinstance(state, dict):
            state["status"] = "Error"
            state["error"] = str(e)
            if "_internal_flags" in state:
                state["_internal_flags"]["interview_complete"] = True
            else:
                state["_internal_flags"] = {"interview_complete": True}

        return error_state

if __name__ == "__main__":
    # Test state for running the interview agent directly
    test_state = {
        # Candidate information
        'name': 'John Doe',
        'applied_role': 'Software Developer',

        # Interview plan and status
        'plan': ["What is your greatest strength?", "Why do you want this job?"],
        'status': 'Plan Incomplete',
        'nb_questions': 2,  # Explicitly set the number of questions

        # Current state tracking
        'current_question': '',
        'response': '',

        # Internal flags for controlling the interview flow
        '_internal_flags': {
            'needs_refinement': False,
            'question_answered': False,
            'question_refined': False
        },

        # History and scoring
        'conversation_history': [],
        'scores': []
    }

    # Run the interview
    result = start_interview_agent(test_state)

    # Print the final state (for debugging)
    print("\nFinal state:", result)
