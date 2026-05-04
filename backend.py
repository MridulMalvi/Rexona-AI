import os
import requests
import tempfile
from typing import TypedDict, Annotated, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# --- NEW IMPORTS FOR AUTOMATIC ID HANDLING ---
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))

# --- DOCKER NETWORKING ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

#  Global Storage 
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

#  SETUP LOCAL EMBEDDINGS 
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL,
)

#  Helper Functions 
def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = None
        batch_size = 20
        
        print(f"Embedding {len(chunks)} chunks locally...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            
        retriever = vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.5
            }
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

def retrieve_all_threads():
    return list(_THREAD_RETRIEVERS.keys())

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

#  Tools 
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
    """Fetch latest stock price for a symbol."""
    api_key = (os.getenv("ALPHAVANTAGE_API_KEY") or "").strip()
    normalized_symbol = (symbol or "").strip().upper()

    if not normalized_symbol:
        return "Error: Please provide a valid stock symbol (example: AAPL)."
    if not api_key:
        return "Error: ALPHAVANTAGE_API_KEY is missing. Add it to your .env file and restart the app."

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={normalized_symbol}&apikey={api_key}"
    )

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Alpha Vantage often returns these keys when the key/symbol/rate-limit is invalid.
        if "Note" in data:
            return f"Error: Alpha Vantage rate limit reached. {data['Note']}"
        if "Error Message" in data:
            return f"Error: Invalid symbol '{normalized_symbol}' or request rejected by Alpha Vantage."
        if "Information" in data:
            return f"Error: {data['Information']}"

        quote = data.get("Global Quote") or {}
        price = quote.get("05. price")
        change = quote.get("09. change")
        change_percent = quote.get("10. change percent")
        last_trading_day = quote.get("07. latest trading day")
        volume = quote.get("06. volume")

        if not price:
            return (
                f"Error: No live quote returned for '{normalized_symbol}'. "
                "Try another symbol or wait a moment and retry."
            )

        return (
            f"{normalized_symbol} latest price: ${price} | "
            f"change: {change or 'N/A'} ({change_percent or 'N/A'}) | "
            f"last trading day: {last_trading_day or 'N/A'} | "
            f"volume: {volume or 'N/A'}"
        )
    except requests.Timeout:
        return "Error: Alpha Vantage request timed out. Please try again."
    except requests.RequestException as e:
        return f"Error: Network/API request failed: {e}"
    except Exception as e:
        return f"Error: {e}"

@tool
def rag_tool(
    query: str, 
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """
    Retrieve information from the uploaded PDF.
    Only provide the query. The system handles the file access automatically.
    """
    # Auto-extract thread_id from the system config
    thread_id = config.get("configurable", {}).get("thread_id")
    
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return "Error: No document indexed. Please ask the user to upload a PDF first."
    
    results = retriever.invoke(query)
    
    formatted_context = "\n\n---\n\n".join(
        [f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(results)]
    )
    
    return f"Here is the relevant context retrieved from the document:\n\n{formatted_context}"

tools = [get_stock_price, search_tool, calculator, rag_tool]

# SETUP LOCAL LLM 
llm = ChatOllama(
    model="qwen2.5",
    temperature=0.2,
    base_url=OLLAMA_BASE_URL,
)

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config=None):
    # System Prompt Simplified: AI doesn't need to know about thread_ids anymore
    sys_msg = SystemMessage(content=(
        "You are a helpful AI assistant. Follow these rules strictly:\n"
        "1. **Greetings:** If the user chats casually, reply naturally. Do NOT use tools.\n"
        "2. **Tool Use:** ONLY call a tool if needed (Search, Stocks, Math, or PDF info).\n"
        "3. **Stocks:** If the user asks for stock price/quote/market value/ticker (example: AAPL, TSLA, MSFT), call `get_stock_price` with that symbol before replying.\n"
        "4. **Math:** For arithmetic requests, call `calculator`.\n"
        "5. **Web:** For current events or general web lookup, call DuckDuckGo search.\n"
        "6. **RAG Usage:** To answer questions about the uploaded document, simply call `rag_tool` with the user's question. Do not worry about IDs."
    ))
    
    response = llm_with_tools.invoke([sys_msg] + state['messages'])
    return {"messages": [response]}

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

checkpointer = InMemorySaver()
chatbot = builder.compile(checkpointer=checkpointer)
