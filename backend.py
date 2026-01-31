import os
import requests
import tempfile
import time
from typing import TypedDict, Annotated, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
# Import the specific error to catch it
from langchain_google_genai._common import GoogleGenerativeAIError

load_dotenv()

# --- Global Storage ---
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

# --- Embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Helper Functions ---

def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Ingests PDF with robust error handling and 'Slow Mode' to survive API limits.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # --- SLOW MODE BATCHING ---
        vector_store = None
        # Batch size 1 is the safest for free tier
        batch_size = 1
        
        print(f"Starting embedding for {len(chunks)} chunks (Slow Mode)...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            attempt = 0
            max_retries = 3
            success = False

            while not success and attempt < max_retries:
                try:
                    print(f"Embedding chunk {i+1}/{len(chunks)}...")
                    if vector_store is None:
                        vector_store = FAISS.from_documents(batch, embeddings)
                    else:
                        vector_store.add_documents(batch)
                    
                    success = True
                    # Standard pause: 5 seconds
                    time.sleep(5) 
                    
                except GoogleGenerativeAIError as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print("Rate limit hit. Waiting 30 seconds before retrying...")
                        time.sleep(30) # Long pause on error
                        attempt += 1
                    else:
                        raise e # If it's another error, crash intentionally
            
            if not success:
                raise Exception("Failed to embed document after multiple retries due to rate limits.")

        # --- END BATCHING ---

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
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
    """Fetch latest stock price for a symbol."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
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

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Ask user to upload a PDF.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]

    return {
        "context": context,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [get_stock_price, search_tool, calculator, rag_tool]

# --- Model & Graph ---

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    sys_msg = SystemMessage(content=(
        "You are a helpful assistant. Use your tools for math, search, stock info, and document analysis. "
        "If the user asks about a PDF or document, use `rag_tool` and explicitly pass the "
        f"current thread_id: `{thread_id}`. "
        "If asked about user identity, check the conversation history."
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