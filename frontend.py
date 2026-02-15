import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# IMPORTS: Ensure these exist in your backend.py file
from backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    # Retrieve state from the backend graph using the thread_id config
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title(text_alignment="center", body="**Rexona AI**")
st.sidebar.caption(text_alignment="center", body="**Your Multi-Tool Intelligence Engine.**") # ADDED CAPTION
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# --- Document Uploader Section ---
doc_meta = thread_document_metadata(thread_key)

if doc_meta:
    st.sidebar.success(
        f"Using `{doc_meta.get('filename')}` "
        f"({doc_meta.get('chunks')} chunks)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])

if uploaded_pdf:
    # Check if we already processed this exact file for this thread to avoid re-processing
    if doc_meta.get('filename') == uploaded_pdf.name:
        st.sidebar.info(f"`{uploaded_pdf.name}` is already active.")
    else:
        with st.sidebar.status("Indexing PDF locally...", expanded=True) as status_box:
            # Call backend to ingest PDF
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            # Update session state with the result
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
            st.rerun()

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title(text_alignment="center", body="🤖 Multi-Agent Chatbot")

# Optional: Add the Hero Section I mentioned earlier
if not st.session_state["message_history"]:
    st.markdown(
        """
        You are Rexona AI, a private AI assistant .
        
        * 🧠 **Multi-Tool Intelligence:** I can use various tools like calculators, search engines, and stock APIs to provide accurate answers.
        * 📄 **RAG:** Upload PDFs to chat with them.
        * 🛠️ **Tools:** I can use Calculator, Search, and Stock APIs.
        """
    )
    st.divider()

# Display Chat History
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    # Add user message to state and display
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        # Generator to stream only AI content while handling Tool status updates separately
        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # If the chunk is a ToolMessage (tool execution output), update status
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` ...", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` ...",
                            state="running",
                            expanded=True,
                        )

                # If the chunk is an AIMessage (text response), yield content
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        # Stream the response to the UI
        ai_message = st.write_stream(ai_only_stream())

        # Close the tool status box if it was used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message to history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )   

st.divider()

# Handle switching threads
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        # Reconstruct history for UI (filter out raw ToolMessages to keep it clean)
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
            temp_messages.append({"role": role, "content": msg.content})
            
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()