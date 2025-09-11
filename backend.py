from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools.retriever import create_retriever_tool
import sqlite3
# Load environment variables
load_dotenv()
travly = os.getenv('travly')
os.environ["TAVILY_API_KEY"] = travly
api_key = os.getenv('groq_api')

# Initialize LLM and Embeddings
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vector store
try:
    db = FAISS.load_local('my_faiss_index', embeddings, allow_dangerous_deserialization=True)
except FileNotFoundError:
    print("Error: FAISS index not found. Please create the index first by running a script to generate it.")
    exit()

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "retriever_tool",
    "Information related to medical conditions, symptoms, treatments, medications, and health advice. Use this tool to answer specific questions about illnesses, drugs, and general health topics.",
)
search_tool = TavilySearch(max_results=2)
tools = [search_tool, retriever_tool]

# Agent Function
def agent(state: AgentState):
    """
    The agent's main function. It reasons and decides to act or provide a final answer.
    """
    messages = state['messages']
    tool_names = ", ".join([tool.name for tool in tools])
    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

    prompt = f"""
    You are DoctorBot, a highly knowledgeable and empathetic AI medical assistant. 
    Your role is to provide accurate and helpful medical information and possible explanations, 
    but you are NOT a licensed doctor. 

    You have access to the following tools:
    {tool_descriptions}

    Your primary goals are:
    - Be empathetic and reassuring, maintaining a professional and warm tone.
    - Provide accurate and concise information based on your knowledge and tools.
    - Use the retriever tool to look up possible causes of symptoms whenever symptoms are mentioned.

    You MUST follow these rules:

    1. If the user reports symptoms (e.g., fever, headache, cough), always query the retriever tool first 
    to check for possible related conditions.
    2. When giving the Final Answer, clearly explain:
    - The common possible causes of those symptoms (from retriever/database).
    - General self-care measures (hydration, rest, monitoring).
    3. Never present a single definitive diagnosis. Always present information as "possible causes" or "conditions that are sometimes associated".
    4. Never prescribe medications, doses, or treatments. Instead, recommend consulting a licensed healthcare professional.
    5. If symptoms indicate an emergency (e.g., severe chest pain, difficulty breathing, unconsciousness), instruct the user to seek immediate emergency care.
    6. The Final Answer must be conversational, empathetic, and merged from all relevant tool outputs.

    Response format (strictly follow ReAct pattern):

    If reasoning:
    Thought: your reasoning about the request.
    Action: the tool you want to use, must be one of [{tool_names}]
    Action Input: input for that tool.

    If answering:
    Final Answer: A complete, empathetic, and conversational response with possible causes 
    (based on tool output), self-care tips, and a clear disclaimer that only a doctor can provide a real diagnosis.

    Conversation history:
    {messages}
    """
    response_content = llm.invoke(prompt).content
    return {'messages': [AIMessage(content=response_content)]}

# Router Function
def should_continue(state: AgentState):
    """
    Router function to decide if the workflow should continue.
    """
    last_message = state['messages'][-1]
    
    if "Final Answer:" in last_message.content:
        return END
        
    if "Action:" in last_message.content and "Action Input:" in last_message.content:
        return "tools"

    return END

# Tool Calling Function
def call_tool(state: AgentState):
    """
    Parses the agent's action and calls the appropriate tool.
    """
    last_message = state['messages'][-1]
    
    action_match = re.search(r"Action:\s*([^\n]+)", last_message.content)
    action_input_match = re.search(r"Action Input:\s*([^\n]+)", last_message.content)

    if not action_match or not action_input_match:
        return {'messages': [ToolMessage(content="Error: Could not parse tool action from agent response.", tool_call_id="parse_error")]}

    tool_name = action_match.group(1).strip()
    tool_input = action_input_match.group(1).strip()

    tool_to_run = next((t for t in tools if t.name == tool_name), None)
    if tool_to_run:
        try:
            tool_output = tool_to_run.invoke(tool_input)
            return {'messages': [ToolMessage(content=tool_output, tool_call_id=tool_name)]} 
        except Exception as e:
            return {'messages': [ToolMessage(content=f"Error: Tool '{tool_name}' failed with: {str(e)}", tool_call_id=tool_name)]}
    else:
        return {'messages': [ToolMessage(content=f"Error: Unknown tool '{tool_name}'", tool_call_id="unknown_tool")]}

# Build the Graph
db=sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=db)
graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", call_tool)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=checkpointer)


def get_all_thread():

    all_thread=set()

    for checpoint in checkpointer.list(None):
        all_thread.add(checpoint.config['configurable']['thread_id'])
    return list(all_thread)

# Example usage
query = "I have a persistent cough and a headache. What could be the cause?"
conf = {"configurable": {"thread_id": 1}}

# The print loop is correct, but the prompt may not generate a tool call for "my name is sahil raj".
# A more relevant query for testing would be "I have a fever and a headache."
# print(f"User Query: {query}")
print("-" * 20)
for message_chunk in app.stream({"messages": [HumanMessage(content=query)]}, config=conf):
   print(message_chunk)

   print("-"*80)