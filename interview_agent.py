import ast
import json
import time
import traceback
from pydantic import BaseModel, Field, field_validator

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from needs import llm, State

class StateParam(BaseModel):
    """Pydantic model for the state parameter."""
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
        
        # Initialize internal flags if not present
        internal_flags = state.get('_internal_flags', {})
        
        conversation_history = state.get('conversation_history', [])
        plan = state.get("plan", [])
        
        if not plan:
            return {
                'status': 'Plan Complete', 
                'conversation_history': conversation_history,
                '_internal_flags': internal_flags
            }
            
        current_question = plan[0]
        
        # Track the question presentation
        question_event = {
            'event_type': 'present_question',
            'question': current_question,
            'is_refined': internal_flags.get('question_refined', False),
        }
        conversation_history.append(question_event)
        
        print(f"\nQ: {current_question}")
        
        # Update internal flags
        internal_flags['needs_refinement'] = False
        
        return {
            'current_question': current_question,
            'conversation_history': conversation_history,
            '_internal_flags': internal_flags
        }
    except Exception as e:
        print(f"Error in present_question: {str(e)}")
        traceback.print_exc()
        return {'status': 'Error'}

@tool(args_schema=StateParam)
def collect_response(state: dict) -> dict:
    """Collects the candidate's response to the current question."""
    try:
        # Ensure state is a dictionary
        if isinstance(state, str):
            state = ast.literal_eval(state)
        
        # Get internal flags
        internal_flags = state.get('_internal_flags', {})
            
        conversation_history = state.get('conversation_history', [])
        
        if state.get("current_question"):
            response = input("\nYour answer: ")
            
            # Track conversation history
            conversation_event = {
                'event_type': 'collect_response',
                'response': response
            }
            
            conversation_history.append(conversation_event)
            
            if not internal_flags.get('question_refined'):
                # Create proper message format for LLM
                messages = [
                    SystemMessage(content="You are evaluating whether a candidate understood an interview question based on their response. Answer with ONLY 'yes' or 'no'."),
                    HumanMessage(content=f"Did the candidate understand this question: '{state['current_question']}' based on their answer: '{response}'? The answer is very short.")
                ]
                
                check = llm.invoke(messages)
                time.sleep(1)
                
                if 'no' in check.content.lower():
                    internal_flags['needs_refinement'] = True
                    internal_flags['question_answered'] = False
                    return {
                        'response': response,
                        'conversation_history': conversation_history,
                        '_internal_flags': internal_flags,
                        'ready_for_eval': False  # Explicitly set to False
                    }
        
            # If we don't need refinement, proceed normally
            plan = state.get("plan", [])
            
            # Only remove the question from the plan if we're not going to refine it
            if not internal_flags.get('needs_refinement', False):
                if plan:  # Check if plan is not empty
                    plan.pop(0)
                internal_flags['question_answered'] = True
                
            # Create the response object
            result = {
                'response': response,
                'plan': plan,
                'conversation_history': conversation_history,
                '_internal_flags': internal_flags,
            }
            
            # Explicitly set ready_for_eval based on question_answered
            result['ready_for_eval'] = internal_flags['question_answered']
            
            # Debug print
            print(f"DEBUG: Setting ready_for_eval to {result['ready_for_eval']}")
            
            return result
        
        return {
            'response': '', 
            'conversation_history': conversation_history,
            '_internal_flags': internal_flags,
            'ready_for_eval': False  # Explicitly set to False
        }
    except Exception as e:
        print(f"Error in collect_response: {str(e)}")
        traceback.print_exc()
        return {'status': 'Error'}
    
@tool(args_schema=StateParam)
def refine_question(state: dict) -> dict:
    """Refines the current question to make it clearer for the candidate."""
    try:
        # Ensure state is a dictionary
        if isinstance(state, str):
            state = ast.literal_eval(state)
        
        # Get internal flags
        internal_flags = state.get('_internal_flags', {})
        
        conversation_history = state.get('conversation_history', [])
        
        if internal_flags.get('needs_refinement', False):
            # Track the refinement attempt
            refinement_event = {
                'event_type': 'refine_question',
                'original_question': state.get('current_question', '')
            }
            
            # Create proper message format for LLM
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
                
                Please provide a refined version of this question.
                """)
            ]
            
            refined = llm.invoke(messages)
            time.sleep(1)
            
            refined_question = refined.content.strip()
            refinement_event['refined_question'] = refined_question
            conversation_history.append(refinement_event)
            
            plan = state.get("plan", [])
            if plan:  # Check if plan is not empty
                plan[0] = refined_question
            
            # Update internal flags
            internal_flags['needs_refinement'] = False
            internal_flags['question_refined'] = True
            
            return {
                'plan': plan,
                'conversation_history': conversation_history,
                '_internal_flags': internal_flags
            }
        
        # If we don't need refinement, just return the current state
        return {
            'current_question': state.get('current_question', ''),
            'conversation_history': conversation_history,
            '_internal_flags': internal_flags
        }
    except Exception as e:
        print(f"Error in refine_question: {str(e)}")
        traceback.print_exc()
        return {'status': 'Error'}

def create_interview_agent(llm):
    """Creates an agent that conducts the interview."""
    tools = [check_interview_plan, present_question, collect_response, refine_question]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are an interviewer conducting an interview with a candidate. Use the following tools to manage the process:  

        {tools}  

        ### **Format:**  
        State: (current state as Python dictionary)  
        Thought: (analyze state and tool outputs to decide the next action)  
        Action: (the action to take, must be one of [{tool_names}])  
        Action Input: (the input to the action)  
        Observation: (result from the tool, updates the state)  
        Thought: (reevaluate the next step based on observations)
         ... (repeat until termination)   
        Final Answer: (final action when the interview process reaches completion)  

        ### **Critical Rules:**  
        I. **Always pass the COMPLETE state when using any tool.** This ensures all context is preserved.
        
        II. **Always start by using `check_interview_plan` tool** to decide termination conditions.
        
        III. **If the plan is complete or status is 'Plan Complete'**, STOP IMMEDIATELY and return the current state.
        
        IV. **Follow this exact sequence for conducting the interview:**
           1. Present question to the candidate
           2. Collect their response
           3. Check the _internal_flags['needs_refinement'] flag:
              a. If true, use refine_question tool to make the question clearer
              b. Present the refined question
              c. Collect their response again
           4. If the response is understood (_internal_flags['question_answered'] is true), check:
              a. If 'ready_for_eval' is true, STOP and return the state for evaluation
        
        V. **Check the internal flags in the _internal_flags dictionary**. These track the interview process.
        
        VI. **If you see 'ready_for_eval' is true in the state, STOP IMMEDIATELY and return the current state for evaluation.**
        
        VII. **Never modify the state directly.** Tools will update it automatically.
         
         
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
        )

agent = create_interview_agent(llm)

def start_interview_agent(state: State):
    """Manages the interview process."""
    # Create a proper deep copy of the state to ensure we preserve all fields
    working_state = state.copy()
    
    # Initialize the internal flags dictionary if not present
    if '_internal_flags' not in working_state:
        working_state['_internal_flags'] = {
            'needs_refinement': False,
            'question_refined': False,
            'question_answered': False,
        }
    
    # Check if coming back from evaluation
    if working_state.get('technical_score'):
        print("DEBUG: Coming back from evaluation, resetting flags")
        # Reset the ready_for_eval flag
        working_state['ready_for_eval'] = False
        # Reset the technical_score since we've now processed it
        working_state['technical_score'] = ''
        # Reset internal flags
        working_state['_internal_flags']['question_answered'] = False
        working_state['_internal_flags']['question_refined'] = False
    
    # Debug the state
    """ print(f"\nDEBUG: Current state keys: {working_state.keys()}")
    print(f"DEBUG: ready_for_eval = {working_state.get('ready_for_eval', False)}")
    print(f"DEBUG: status = {working_state.get('status', 'Unknown')}")
    print(f"DEBUG: plan = {working_state.get('plan', [])}") """
    
    try:
        print(f"\Starting interview for {working_state.get('name', 'candidate')} ({working_state.get('applied_role', 'unknown role')})...")
        
        # Debug internal flags
        internal_flags = working_state.get('_internal_flags', {})
        """ debug_flags = f"BEFORE: needs_refinement={internal_flags.get('needs_refinement', False)}, question_refined={internal_flags.get('question_refined', False)}, question_answered={internal_flags.get('question_answered', False)}"
        print(f"DEBUG: {debug_flags}")
         """
        # Run through the steps
        result = agent.invoke({
            "input": working_state
        })
        
        time.sleep(2)
        
        # Update working state with all intermediate steps
        for step in result["intermediate_steps"]:
            if isinstance(step[1], dict):
                working_state.update(step[1])
        
        # Debug internal flags after processing
        internal_flags = working_state.get('_internal_flags', {})
        """ debug_flags = f"AFTER: needs_refinement={internal_flags.get('needs_refinement', False)}, question_refined={internal_flags.get('question_refined', False)}, question_answered={internal_flags.get('question_answered', False)}"
        print(f"DEBUG: {debug_flags}")
        print(f"DEBUG: ready_for_eval = {working_state.get('ready_for_eval', False)}") """
        
        # Force the status update based on the plan
        if 'plan' in working_state and not working_state.get('plan'):
            working_state['status'] = 'Plan Complete'
        
        # Update the original state with all changes
        if hasattr(state, 'update'):
            state.update(working_state)
        elif isinstance(state, dict):
            state.clear()
            state.update(working_state)
        else:
            # Handle State object
            for key, value in working_state.items():
                setattr(state, key, value)
        
        return working_state
        
    except Exception as e:
        print(f"Error in start_interview_agent: {str(e)}")
        traceback.print_exc()
        return {"status": "Error", "error": str(e)}

if __name__ == "__main__":
    test_state = {
        'name': 'John Doe',
        'applied_role': 'Software Developer',
        'plan': ["What is your greatest strength?", "Why do you want this job?"],
        'current_question': '',
        'response': '',
        'status': 'Plan Incomplete',
        'needs_refinement': False,
        'question_refined': False,
        'question_answered': False,
        'conversation_history': []
    }
    result = start_interview_agent(test_state)
    print("Final state:", result)