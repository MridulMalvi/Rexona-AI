import streamlit as st
import uuid
from backend import chatbot
from langchain_core.messages import HumanMessage

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
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    values = state.values
    return values.get('messages', [])  # Returns empty list if 'messages' not found


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
st.sidebar.title('MY CHATBOT')
if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('Previous Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    # Use a unique key for the button to avoid conflicts
    if st.sidebar.button(str(thread_id), key=f"thread_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        
        # --- Start of Correction ---
        # Renamed 'message' to 'message_list' to avoid variable shadowing
        message_list = load_conversation(thread_id)

        temp_message = []

        # Renamed 'message' to 'msg' in the loop
        for msg in message_list:
           if isinstance(msg, HumanMessage):
               role = 'user'
           else:
               role = 'assistant'
           # Use 'msg.content'
           temp_message.append({'role': role, 'content': msg.content})
        # --- End of Correction ---
           
        st.session_state['message_history'] = temp_message

# Display chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content']) 

# User input
user_input = st.chat_input("Enter your message...")

if user_input:
    # Save user msg
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}}

    # Stream assistant response
    response_chunks = []

    def handle_stream(chunk):
        response_chunks.append(chunk)
        return chunk

    with st.chat_message("assistant"):
        st.write_stream(
            handle_stream(message_chunk.content)
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )

    final_response = "".join(response_chunks)
    st.session_state['message_history'].append({'role': 'assistant', 'content': final_response})