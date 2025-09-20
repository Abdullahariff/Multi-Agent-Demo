import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import initialize_agent, Tool, AgentType

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize tools
wiki_api = WikipediaAPIWrapper()
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)
calc_tool = PythonREPLTool()

llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-3.5-turbo",
    temperature=0
)

# Wrap tools
tools = [
    Tool(
        name="Wikipedia Search",
        func=wiki_tool.run,
        description="Useful for answering questions about people, places, or things from Wikipedia."
    ),
    Tool(
        name="Calculator",
        func=calc_tool.run,
        description="Useful for solving math problems."
    )
]

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.title("Multi-Agent Demo")
user_query = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_query:
        with st.spinner("Thinking..."):
            try:
                response = agent.run(user_query)
                st.success(response)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
