from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from needs import llm , State
from typing import List, Dict, Union

@tool
def check_interview_plan(state: State):
    """Checks the status of the interview plan."""
    is_complete = not state['plan'] or len(state['plan']) == 0
    return {'status': 'Plan Complete' if is_complete else 'Plan Incomplete'}

@tool
def present_question(state: State):
    """Presents the next question in the interview plan."""
    if not state['plan']:
        return {'current_question': '', 'status': 'Plan Complete'}
    
    current_question = state['plan'][0]
    remaining_plan = state['plan'][1:]  # Remove the current question from the plan
    
    return {
        'current_question': current_question,
        'plan': remaining_plan  # Update the plan to remove the current question
    }
@tool
def collect_response(state: State):
    """Collects the candidate's response to the current question."""
    if state['current_question']:
        print(f"\nQ: {state['current_question']}")
        response = input("Your answer: ").strip()
        return {
            'response': response,
        }
    return {'response': ''}

def create_interview_agent(llm):
    """Creates an agent that conducts the interview."""
    tools = [present_question, collect_response, check_interview_plan]

    prompt = ChatPromptTemplate([
        ("system", """
        You are an interviewer conducting an interview. Use these tools to manage the interview:
        1. present_question - Get the next question from the plan
        2. collect_response - Mark the response as received
        3. check_plan - Check if there are more questions

        Rules:
         -Ask only one question at a time
        -If there's a user response collect it using the collect_response tool
        -Check the Interview Plan between each question and the next question
         
        """),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True  # Capture intermediate steps
    )

def start_interview_agent(plan: List[str]):
    """Generate initial plan and start the interview."""

    agent = create_interview_agent(llm)

    # Present the first question
    response = agent.invoke({
        "input": f"Here's the interview plan: {plan}"
    })

    # Extract the tool's output directly from intermediate steps
    if 'intermediate_steps' in response:
        for step in response['intermediate_steps']:
            if step[0].tool == "present_question":
                tool_output = step[1]
                if isinstance(tool_output, dict):
                    current_question = tool_output.get('current_question', '')
                    remaining_plan = tool_output.get('plan', [])
                    status = tool_output.get('status', '')

                    if status == 'Plan Complete':
                        print("Interview completed. No more questions in the plan.")
                        return  # Exit the function as the interview is complete

                    print(f"\nQ: {current_question}")
                    user_response = input('Your answer: ')

                    # Check the plan status
                    check_response = agent.invoke({
                        "input": f"Check if the plan is complete: {remaining_plan}"
                    })

                    if check_response.get('status', '') == 'Plan Complete':
                        print("Interview completed. No more questions in the plan.")
                        return  # Exit the function as the interview is complete
                    else:
                        # Update the plan and return to the workflow
                        return {'plan': remaining_plan, 'response': user_response}
                else:
                    print("Unexpected tool output format:", tool_output)
                    return  # Exit the function due to unexpected output
    else:
        print("Unexpected response format:", response)
        return  # Exit the function due to unexpected response


# Example usage
if __name__ == "__main__":
    start_interview_agent()
