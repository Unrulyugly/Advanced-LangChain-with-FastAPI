import os
import math
import json
import requests
import dotenv
import logging
import uvicorn
import operator
from enum import Enum
from typing import List, Any
from pydantic import BaseModel
from langserve import add_routes
from langchain_core.runnables import chain
from langchain_groq import ChatGroq
from typing import Sequence, Literal
from langchain.schema import Document
from langchain.tools import StructuredTool
from fastapi import FastAPI, HTTPException
from store_factory import get_vector_store
from dotenv import find_dotenv, load_dotenv
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph, END
from langchain_nomic import NomicEmbeddings
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import FunctionMessage

from langchain_core.runnables import RunnableLambda
from langchain.indexes import SQLRecordManager, index
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ValidationError, validator
from langchain.tools.retriever import create_retriever_tool
from fastapi import FastAPI, HTTPException,UploadFile, File
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader , DirectoryLoader



load_dotenv(find_dotenv())
config = dotenv.dotenv_values(".env")
OPENAI_API_KEY = config['OPENAI_API_KEY']
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
groq_api_key = config['GROQ_API_KEY']
groq_api_key = os.environ["GROQ_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class SearchResult(BaseModel):
    title: str
    snippet: str

    @validator('snippet',)
    def must_contain_keywords(cls, value, field):
        keywords = ["CEO", "Chief Executive Officer", "Executive Director", "Managing Partner", "President"]
        if not any(keyword in value for keyword in keywords):
            raise ValueError(f"{field.name} must contain one of the following: {keywords}")
        return value

def get_ceo_info(email_url: str) -> List[dict]:
    search_query = f"CEO of {email_url}"
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": search_query})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        return []
    return response.json().get('organic', [])

def validate_and_filter_results(data: List[dict]) -> List[SearchResult]:
    valid_results = []
    for item in data:
        try:
            valid_result = SearchResult(**item)
            valid_results.append(valid_result)
        except ValidationError:
            continue
    return valid_results
def display_results_to_file(results, email_url, output_directory, file_number):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    output_file_path = os.path.join(output_directory, f"data_part_{file_number}.txt")
    
    with open(output_file_path, "a", encoding='utf-8') as text_file:
        text_file.write(f"Email URL: {email_url}\n")
        for result in results:
            text_file.write(f"Title: {result.title}\nSnippet: {result.snippet}\n")
        text_file.write("\n" + "-"*150 + "\n")

loader = DirectoryLoader('./FAQ', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1840, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=texts,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1"),
)
    
retriever = vectorstore.as_retriever()
retriever_tool=create_retriever_tool(
    retriever,
    "company-search",
    description="Use this tool when retrieving information about any company C.E.O or C.E.O equivalent from the retriever")

class EmailUrlInput(BaseModel):
    email_url: str = Field(description="The email URL to search for in the input.txt file")

def find_email_line(email_url: str) -> str:
    """Read the input file and find the line containing the specified email URL."""
    try:
        with open('input.txt', 'r') as file:
            for line in file:
                if email_url in line:
                    return line.strip()
    except FileNotFoundError:
        return "File not found."
    return "Email URL not found in the file."
# Create the structured tool
email_finder = StructuredTool.from_function(
    func=find_email_line,
    name="EmailFinder",
    description="Find a line with the specified email URL in the input.txt file",
    args_schema=EmailUrlInput,
    return_direct=True 
)
class AppendCEOInput(BaseModel):
    email_url: str = Field(description="The email URL to find in the file.")
    ceo_name: str = Field(description="The CEO's name to append above the email URL.")

def append_ceo_above_email(email_url: str, ceo_name: str) -> str:
    """Inserts the CEO's name above the associated email URL in 'input.txt'."""
    try:
        with open('input.txt', 'r+') as file:
            lines = file.readlines()
            file.seek(0) 
            file.truncate() 
            updated = False
            for line in lines:
                if email_url in line:
                    file.write(f"{ceo_name}\n")
                    updated = True
                file.write(line)

            if not updated:
                return "Email URL not found in the file."
                
        return "CEO name appended successfully."
    except Exception as e:
        return f"Failed to append CEO name: {str(e)}"

ceo_appender = StructuredTool.from_function(
    func=append_ceo_above_email,
    name="CEOAppender",
    description="Appends the CEO's name above the associated email URL in input.txt",
    args_schema=AppendCEOInput,
    return_direct=True 
)
tools = [retriever_tool,email_finder,ceo_appender]
tool_executor = ToolExecutor(tools)
class Response(BaseModel):
    """Final answer to the user"""
    explanation: str = Field(
        description="explanation of the steps taken to get the result"
    )

chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
llm_with_tools = chat.bind_tools([email_finder,ceo_appender, retriever_tool])
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
def should_continue(state):
    return "continue" if state["messages"][-1].tool_calls else "end"

def call_model(state, config):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
def _invoke_tool(tool_call):
    tool = {tool.name: tool for tool in tools}[tool_call["name"]]
    return ToolMessage(tool.invoke(tool_call["args"]), tool_call_id=tool_call["id"])

tool_executor = RunnableLambda(_invoke_tool)

def call_tools(state):
    last_message = state["messages"][-1]
    return {"messages": tool_executor.batch(last_message.tool_calls)}

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")
graph = workflow.compile()

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any

@chain
def custom_chain(text):
  inputs = {"messages": [HumanMessage(content=f"{text}")]}
  return graph.invoke(inputs)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search-ceo/", response_model=List[SearchResult])
async def search_ceo(email_url: str):
    api_response_data = get_ceo_info(email_url)
    if not api_response_data:
        raise HTTPException(status_code=400, detail="Failed to fetch data or no data available for this query.")
    validated_results = validate_and_filter_results(api_response_data)
    return validated_results

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    lines = content.decode("utf-8").splitlines()
    total_urls = len(lines)
    urls_per_file = math.ceil(total_urls / 10)
    output_directory = 'google_engine'

    for i, line in enumerate(lines):
        at_pos = line.find('@')
        if at_pos != -1:
            email_url = line[at_pos+1:].strip()
            api_response_data = get_ceo_info(email_url)
            validated_results = validate_and_filter_results(api_response_data)
            
            file_number = i // urls_per_file + 1
            display_results_to_file(validated_results, email_url, output_directory, file_number)

    return {"message": "File processed successfully"}


add_routes(
    app,
    custom_chain.with_types(input_type=str, output_type=Any),
    path="/chat"
)
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
