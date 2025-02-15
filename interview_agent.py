from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from needs import llm, State
from pydantic import BaseModel, Field
from typing import List, Dict



# Define the schema for the `state` parameter
class StateSchema(BaseModel):
    messages: List[str] = Field(default_factory=list, description="List of messages in the conversation")
    applied_role: str = Field(..., description="The role the candidate has applied for")
    technical_skills: List[str] = Field(default_factory=list, description="List of technical skills")
    name: str = Field(..., description="Name of the candidate")
    plan: List[str] = Field(default_factory=list, description="List of questions in the interview plan")
    #scores: List[Dict[str, str]] = Field(default_factory=list, description="List of scores and evaluations")
    status: str = Field(..., description="Current status of the interview")
    current_question: str = Field(default="", description="Current question being asked")
    response: str = Field(default="", description="Candidate's response to the current question")
    technical_score: str = Field(default="", description="Technical score for the current response")
    report: str = Field(default="", description="Final interview report")

# Update the tools with the schema
@tool(args_schema=StateSchema)
def check_interview_plan(state: StateSchema):
    """Checks the status of the interview plan."""
    is_complete = not state.plan or len(state.plan) == 0
    return {'status': 'Plan Complete' if is_complete else 'Plan Incomplete'}

@tool(args_schema=StateSchema)
def present_question(state: StateSchema):
    """Presents the next question in the interview plan."""
    if not state.plan:
        return {'current_question': '', 'status': 'Plan Complete'}
    current_question = state.plan[0]
    remaining_plan = state.plan[1:]  # Remove the current question from the plan
    return {
        'current_question': current_question,
        'plan': remaining_plan  # Update the plan to remove the current question
    }

@tool(args_schema=StateSchema)
def collect_response(state: StateSchema):
    """Collects the candidate's response to the current question."""
    if state.current_question:
        print(f"\nQ: {state.current_question}")
        response = input("Your answer: ").strip()
        return {
            'response': response,
        }
    return {'response': ''}
def create_interview_agent(llm):
    """Creates an agent that conducts the interview."""
    tools = [check_interview_plan, present_question, collect_response]
    prompt = ChatPromptTemplate([
        ("system", """
         You are an interviewer conducting an interview. Use these tools to manage the interview:
         1. present_question - Get the next question from the plan
         2. collect_response - Mark the response as received
         3. check_interview_plan - Check if there are more questions
         Rules:
         - Ask only one question at a time
         - If there's a user response collect it using the collect_response tool
         - Check the plan status using the check_interview_plan tool
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

def start_interview_agent(state: State):
    """Manages the interview process."""    
    agent = create_interview_agent(llm)

    try:
        print(f"\nStarting interview for {state['name']} ({state['applied_role']})...")
        
        # Get first question
        response = agent.invoke({"input": f"Here is the plan: {state['plan']}"})

        if "current_question" in response:
            state["current_question"] = response["current_question"]
            state["plan"] = response["plan"]

            if response.get("status") == "Plan Complete" or not state["current_question"]:
                return {"status": "Plan Complete"}

            # Collect response
            collect_response = agent.invoke({"input": "collect the response"})
            if "response" in collect_response:
                state["response"] = collect_response["response"]

                return {
                    "current_question": state["current_question"],
                    "response": state["response"],
                    "plan": state["plan"]
                }

        # Check interview completion
        check_response = agent.invoke({"input": "check_interview_plan"})
        if check_response.get("status") == "Plan Complete":
            return {"status": "Plan Complete"}

        return {"status": "Error"}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "Error"}

# Run the test
if __name__ == "__main__":
    test_state = {
        'plan': ["What is your name?", "What is your greatest strength?", "Why do you want this job?"],
        'status': 'Plan Incomplete'
    }
    start_interview_agent(test_state)
