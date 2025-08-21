import streamlit as st
from langchain_core.messages import HumanMessage
from Medical_bot.backend import app,get_all_thread
import re
import uuid

LANGCHAIN_PROJECT = 'Chatbot Project'

# ************************ Utility functions *****************
def generate_thread_id():
    """Generates a new, unique thread ID."""
    return str(uuid.uuid4())

def add_thread(thread_id):
    """Adds a new thread ID to the session state if it doesn't exist."""
    if thread_id not in st.session_state.get('chat_thread',):
        st.session_state['chat_thread'].append(thread_id)

def reset_chat():
    """Resets the chat by generating a new thread ID."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] =[]

def load_conversation(thread_id):
    """Loads a conversation from the LangGraph checkpointer using the thread ID."""

    config = {'configurable': {'thread_id': thread_id}}
    state = app.get_state(config=config)

    messages = state.values.get('messages',)
    
    temp_message =[]
    for msg in messages:
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        temp_message.append({'role': role, 'content': msg.content})
    
    return temp_message

# **************************************** Session Setup *****************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] =[]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_thread' not in st.session_state:

    st.session_state['chat_thread'] =[]


add_thread(st.session_state['thread_id'])

#******************* Sidebar **************
st.sidebar.title('🤖  Past Consultations')
if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.title('My Conversations')
for thd_id in st.session_state['chat_thread'][::-1]:
    if st.sidebar.button(str(thd_id)):
        st.session_state['thread_id'] = thd_id
        messages = load_conversation(thd_id)
        st.session_state['message_history'] = messages
        # Rerun the app to update the main UI with the loaded messages.
        st.rerun()

# ************************* Main UI *************************
st.title("🩺 AI-DRIVEN MEDICAL ASSISTANT")

# Display past messages
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Type here...")

# A dedicated generator to parse the stream output.
def parse_langgraph_stream(result_stream):
    """
    Parses the complex, nested output of a LangGraph stream to yield
    only the meaningful AI message content.
    """
    for chunk in result_stream:

        for node_name, node_output in chunk.items():
            if 'messages' in node_output and node_output['messages']:
                last_message = node_output['messages'][-1]

                if hasattr(last_message, 'content') and last_message.content.strip():

                    yield last_message.content
            

if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.write(user_input)

    config = {
        'configurable': {
            'thread_id': st.session_state['thread_id'],
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn"
    }

    query = user_input

    result_stream = app.stream({"messages": [HumanMessage(content=query)]}, config=config)

    cleaned_stream = parse_langgraph_stream(result_stream)
    

    with st.chat_message("AI"):
        ai_message = st.write_stream(cleaned_stream)

    # Save the full, concatenated AI message to the chat history.
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})