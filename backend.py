import os
import requests
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# --- Tools ---
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic: add, sub, mul, div."""
    try:
        if operation == "add": result = first_num + second_num
        elif operation == "sub": result = first_num - second_num
        elif operation == "mul": result = first_num * second_num
        elif operation == "div":
            if second_num == 0: return {"error": "Division by zero"}
            result = first_num / second_num
        else: return {"error": "Unsupported operation"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> str:
    """Fetch latest stock price for a symbol (e.g. 'AAPL')."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY") # Load from .env
    if not api_key:
        return "Error: API Key not configured."
        
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        r = requests.get(url)
        data = r.json()
        quote = data.get("Global Quote", {})
        if not quote:
            return f"Could not find data for {symbol}."
        return f"Price of {symbol}: ${quote.get('05. price')} (Change: {quote.get('10. change percent')})"
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

tools = [get_stock_price, search_tool, calculator]

# --- Model & Graph ---
# Ensure you use the correct model version available to your API key
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # Simple system instruction to balance memory and tools
    sys_msg = SystemMessage(content="You are a helpful assistant. Use your tools for math, search, and stock info. If asked about user identity, check the conversation history.")
    
    # We invoke the model with the system message + current history
    response = llm_with_tools.invoke([sys_msg] + state['messages'])
    
    # Return the new message to update the state
    return {"messages": [response]}
    
    # Prepend the system instruction to the existing message history
    return {"messages": [llm_with_tools.invoke([sys_msg] + state['messages'])]}

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

checkpointer = InMemorySaver()
chatbot = builder.compile(checkpointer=checkpointer)
