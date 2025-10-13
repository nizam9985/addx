import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_openai import OpenAI
os.environ["OPENAI_API_KEY"] = ''
#'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'
# Load the CSV file into a pandas DataFrame
df = pd.read_csv("/Users/nizam/Desktop/fraud_detection_agent/claims_dataset.csv")

# Create a Gemini language model instance

llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0)
def query_agent(agent, query: str):
    """
    Query the agent and return the response.
    """
    prompt = (
        """
        If it is just asking a question that requires a textual answer, reply as follows:
        {"answer": "answer"}
        For example:
        {"answer": "The title with the highest rating is 'Gilead'"}
        
        If you do not know the answer, reply as follows:
        {"answer": "I do not know."}
        
        Return all output as a string.
        
        All strings in "columns" list and data list, should be in double quotes,
    
        Lets think step by step.
        """
        + query
    )

    response = agent.run(prompt)
    return response.__str__()

# Create the LangChain agent
def agent_is_replying(df):
    system_prompt = f"""
    You are a fraud detection agent. Your task is to classify a new insurance claim as fraudulent (1) or not fraudulent (0).
    Analyze the provided new claim data and the existing dataset to make your decision.
    Pay close attention to inconsistencies, suspicious patterns, and red flags.

    Here are some examples of how to classify claims:
    - A claim with a history of similar claims, inconsistent stories, and contradictory witness statements is likely fraudulent (1).
    - A claim for a non-covered procedure, like cosmetic surgery listed as a medical necessity, is fraudulent (1).
    - A claim with a valid police report, receipts for stolen items, and a clear incident description is likely not fraudulent (0).
    - A claim for a minor incident with an exaggerated injury claim, like whiplash from a fender bender with no visible damage, is suspicious and could be fraudulent (1).

    Now, classify the following new claim:
    {{new_claim}}

    Based on your analysis, is this claim fraudulent? Answer with 0 for not fraudulent or 1 for fraudulent.
    """
    response = query_agent(agent, system_prompt)
    print("Agent's classification:")

from langchain.tools import Tool

def func1(prompt: str):
    """
    Executes a natural language prompt using the agent.
    """
    response = agent.run(prompt)
    return response


tool1 = Tool(
    name='calculate_fraud_percentage',
    func=func1,
    description="Executes a natural language prompt using the agent."
)

extra_tools = [tool1]

agent = create_pandas_dataframe_agent(llm, df, verbose=True, extra_tools=extra_tools, allow_dangerous_code=True)

# Example query
agent.run("What is the total number of claims that are related to home?")