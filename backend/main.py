import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool

# Load API Key
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Multi-Agent Backend")

# Initialize LLM (GPT-4o-mini via OpenRouter)
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Tools
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
python_tool = PythonREPLTool()

tools = [
    Tool(
        name="Wikipedia Search",
        func=wiki.run,
        description="Use this to find information from Wikipedia."
    ),
    Tool(
        name="Python REPL",
        func=python_tool.run,
        description="Use this to run Python code for calculations."
    )
]

# Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Request model
class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_agent(request: QueryRequest):
    response = agent.invoke(request.query)
    return {"answer": response["output"]}
