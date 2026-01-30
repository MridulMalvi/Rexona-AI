import streamlit as st
import uuid
from backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage

# Utility functions
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread_to_history(thread_id)
    st.session_state['message_history'] = []

def add_thread_to_history(thread_id):
    st.session_state['chat_threads'].append(thread_id)
    
def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        values = state.values
        return values.get('messages', [])  # Returns empty list if 'messages' not found
    except Exception:
        return []

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    add_thread_to_history(st.session_state['thread_id'])

# Sidebar UI
st.sidebar.title('LangGraph Chatbot')
if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('Previous Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    # Use a unique key for the button to avoid conflicts
    if st.sidebar.button(str(thread_id), key=f"thread_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        
        message_list = load_conversation(thread_id)
        temp_message = []

        for msg in message_list:
           if isinstance(msg, HumanMessage):
               role = 'user'
           else:
               role = 'assistant'
           
           # Handle content
           content = msg.content
           if isinstance(content, list):
               # Join text parts if content is list
               parts = []
               for item in content:
                   if isinstance(item, dict) and 'text' in item:
                       parts.append(item['text'])
               content = "".join(parts)
           
           if content: # Only append if there is content
                temp_message.append({'role': role, 'content': content})
           
        st.session_state['message_history'] = temp_message

# Display chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content']) 

# User input
if user_input := st.chat_input("Enter your message..."):
    # Save user msg
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state['thread_id']}}

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_placeholder = st.empty() 
        full_response = ""
        
        try:
            # Stream both messages (tokens) and updates (tool status)
            for chunk_type, payload in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode=["messages", "updates"]
            ):
                # 1. Handle Updates (like "tools")
                if chunk_type == "updates":
                    if "tools" in payload and payload['tools']:
                        status_placeholder.info("Running tools...")

                # 2. Handle Message Chunks
                if chunk_type == "messages":
                    msg, metadata = payload
                    if isinstance(msg, AIMessage):
                        
                        chunk_content = msg.content
                        
                        # Handle varied content types from Gemini
                        text_to_add = ""
                        if isinstance(chunk_content, str):
                            text_to_add = chunk_content
                        elif isinstance(chunk_content, list):
                            for part in chunk_content:
                                if isinstance(part, dict) and 'text' in part:
                                    text_to_add += part['text']
                        
                        full_response += text_to_add
                        
                        # Streaming UI update
                        if full_response:
                             # Once we have text, clear the status
                            status_placeholder.empty()
                            response_placeholder.markdown(full_response + "▌")
                            
        except Exception as e:
            st.error(f"Error: {e}")

        # Final update
        response_placeholder.markdown(full_response)
        st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
