import json
import pandas as pd
import os
import operator
import uvicorn
import logging
import requests
from typing import Optional, Type
from enum import Enum
from typing import List
from fastapi.responses import StreamingResponse
from langchain_groq import ChatGroq
from langchain.schema import Document
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import BaseMessage,HumanMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph, END
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain.indexes import SQLRecordManager, index
from dotenv import load_dotenv, find_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, ValidationError, validator
from collections import defaultdict
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

# Load environment variables
load_dotenv(find_dotenv())
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

class SearchResult(BaseModel):
    title: str
    snippet: str

    @validator("snippet")
    def must_contain_keywords(cls, value):
        keywords = ["CEO", "Chief Executive Officer", "Executive Director", "Managing Partner", "President", "Founder"]
        if not any(keyword in value for keyword in keywords):
            raise ValueError("Snippet must contain a relevant keyword.")
        return value

def get_ceo_info(email_url: str) -> List[dict]:
    search_query = f"CEO of {email_url}"
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": search_query})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        return []
    return response.json().get("organic", [])

def validate_and_filter_results(data: List[dict]) -> List[SearchResult]:
    valid_results = []
    for item in data:
        try:
            valid_result = SearchResult(**item)
            valid_results.append(valid_result)
        except ValidationError:
            continue
    return valid_results

def format_results(all_results):
    # This function aggregates titles and snippets by 'email_url'
    organized_data = defaultdict(list)
    for result in all_results:
        organized_data[result['email_url']].append((result['title'], result['snippet']))
    # Now format the data for output
    formatted_data = []
    for email_url, results in organized_data.items():
        titles = " || ".join(f"{i+1}. {title}" for i, (title, _) in enumerate(results))
        snippets = " || ".join(f"{i+1}. {snippet}" for i, (_, snippet) in enumerate(results))
        formatted_data.append({"Email_URL": email_url, "Title": titles, "Snippet": snippets, "Source": "email_domain_result.csv"})
    return formatted_data

CONNECTION_STRING = "postgresql://langchain:langchain@localhost:6024/langchain"
COLLECTION_NAME= "langchain_pg_embeddding"
embeddings =NomicEmbeddings(model="nomic-embed-text-v1")

vectorstore = PGVector(
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    embeddings=embeddings,
    use_jsonb=True
)
namespace = f"pgvector/{COLLECTION_NAME}"

record_manager = SQLRecordManager(namespace, db_url=CONNECTION_STRING)
record_manager.create_schema()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRequest(BaseModel):
    page_content: str
    metadata: dict

class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"

retriever = vectorstore.as_retriever()
retriever_tool=create_retriever_tool(
    retriever,
    "ceo-name-search",
    description="Use this tool when retrieving information about the name of any company C.E.O or C.E.O equivalent from the retriever")
class FileInfoInput(BaseModel):
    domain: str = Field(description="Domain to search in the file")
    ceo_name: str = Field(description="CEO's name to append")

class FileManipulatorTool(BaseTool):
    name = "file_manipulator"
    description = "Appends CEO's name to the line just above the matching domain in input.txt"
    args_schema: Type[BaseModel] = FileInfoInput

    def _run(self, domain: str, ceo_name: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        file_path = 'input.txt'
        if not os.path.exists(file_path):
            return "File not found"

        new_content = []
        found = False
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                if domain in lines[i]:
                    new_content.append(ceo_name + "\n") 
                    found = True
                new_content.append(lines[i])

        if found:
            with open(file_path, 'w') as file:
                file.writelines(new_content)
            return "CEO's name appended successfully"
        else:
            return "Domain not found"

    async def _arun(self, domain: str, ceo_name: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._run(domain, ceo_name, run_manager)
manipulator_tool = FileManipulatorTool()
tools = [retriever_tool, manipulator_tool]
tool_executor = ToolExecutor(tools)

chat = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
model = chat.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"
# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    # We construct an ToolInvocation for each tool call
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        action = ToolInvocation(
                        tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        tool_invocations.append(action)

    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # We call the tool_executor and get back a response
    responses = tool_executor.batch(tool_invocations, return_exceptions=True)
    # We use the response to create tool messages
    tool_messages = [
        ToolMessage(
            content=str(response),
            name=tc["name"],
            tool_call_id=tc["id"],
        )
        for tc, response in zip(last_message.tool_calls, responses)
    ]

    # We return a list, because this will get added to the existing list
    return {"messages": tool_messages}
def first_model(state):
    human_input = state["messages"][-1].content
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ceo-name-search",
                        "args": {
                            "query": human_input,
                        },
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }
workflow = StateGraph(AgentState)

# Define the new entrypoint
workflow.add_node("first_agent", first_model)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("first_agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

workflow.add_edge("action", "agent")
workflow.add_edge("first_agent", "action")
graph = workflow.compile()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/search-ceo/", response_model=List[SearchResult])
async def search_ceo(email_url: str):
    api_response_data = get_ceo_info(email_url)
    if not api_response_data:
        raise HTTPException(status_code=400, detail="Failed to fetch data or no data available for this query.")
    validated_results = validate_and_filter_results(api_response_data)
    return validated_results

@app.post("/upload-list/")
async def upload_file(file: UploadFile = File(...)):
    all_results = []
    with open('input.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            at_pos = line.find('@')
            if at_pos != -1:
                email_url = line[at_pos+1:].strip()
                api_response_data = get_ceo_info(email_url)  # Moved inside the loop
                if api_response_data:
                    validated_results = validate_and_filter_results(api_response_data)
                    for result in validated_results:
                        all_results.append({"email_url": email_url, "title": result.title, "snippet": result.snippet, "Source": "email_domain_result.csv"})

    if not all_results:
        print("No results to write to CSV.")
    else:
        formatted_results = format_results(all_results)
        df = pd.DataFrame(formatted_results)
        output_directory = 'google_engine'
        output_file_path = os.path.join(output_directory, "email_domain_results.csv")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        df.to_csv(output_file_path, index=False)
        print(f"All results saved to {output_file_path}")

    return {"message": "File processed successfully"}

@app.post("/index")
async def index_documents(docs_request: list[DocumentRequest], cleanup: CleanupMethod = CleanupMethod.incremental) -> dict:
    try:
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in docs_request
        ]
        result = index(
            documents,
            record_manager,
            vectorstore,
            cleanup=cleanup.value,
            source_id_key="Source",
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        return {"status": "error", "message": str(e)}


async def generate_chat_events(message):
    inputs = [HumanMessage(content=message)]
    buffer = ""
    
    async for event in graph.invoke({"messages": inputs}, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                buffer += content
                # If the buffer contains a complete line, yield it
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    yield f"{line}\n\n"
        elif kind == "on_tool_start":
            yield f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}\n"
        elif kind == "on_tool_end":
            yield f"Done tool: {event['name']}\n"
            yield f"Tool output was: {event['data'].get('output')}\n"
        elif kind == "on_chat_model_end":
            print("Chat model has completed its response.")
            if buffer:
                # Yield any remaining content in the buffer
                yield f"{buffer}\n"

@app.get("/chat_stream/{message}")
async def chat_stream_events(message: str):
    return StreamingResponse(generate_chat_events(message), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
