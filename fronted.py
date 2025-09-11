import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import app,get_all_thread
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
    """
    Loads and cleans a conversation from the LangGraph checkpointer for display.
    """
    config = {'configurable': {'thread_id': thread_id}}
    state = app.get_state(config=config)
    if not state:
        return []
    messages = state.values.get('messages', [])
    
    cleaned_messages = []
    for msg in messages:
        if msg.content:
            if isinstance(msg, HumanMessage):
                cleaned_messages.append({'role': 'user', 'content': msg.content})

            elif isinstance(msg, AIMessage) and "Final Answer:" in msg.content:
                final_answer_start = msg.content.find("Final Answer:") + len("Final Answer:")
                cleaned_content = msg.content[final_answer_start:].strip()
        
                cleaned_messages.append({'role': 'assistant', 'content': cleaned_content})
                
    return cleaned_messages

# **************************************** Session Setup *****************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] =[]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_thread' not in st.session_state:

    st.session_state['chat_thread'] =get_all_thread()

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
        st.rerun()

# ************************* Main UI *************************
st.title("🩺 AI-DRIVEN MEDICAL ASSISTANT")

# Display past messages
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Type here...")


def parse_langgraph_stream(result_stream, final_node_name="agent"):
    """
    Parses the LangGraph stream to yield only the final answer
    from the specified node.
    """
    final_answer_started = False
    for chunk in result_stream:
        if final_node_name in chunk:
            node_output = chunk[final_node_name]
            if 'messages' in node_output and node_output['messages']:
                last_message = node_output['messages'][-1]
                content = last_message.content

                if "Final Answer:" in content:
                    answer_start_index = content.find("Final Answer:") + len("Final Answer:")
                    final_answer_chunk = content[answer_start_index:].strip()
                    if final_answer_chunk:
                        yield final_answer_chunk
                elif final_answer_started:
                     yield content

# --- Inside your main app logic ---
if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        config = {
            'configurable': {'thread_id': st.session_state['thread_id']},
        }
        
        result_stream = app.stream({"messages": [HumanMessage(content=user_input)]}, config=config)

        cleaned_stream = parse_langgraph_stream(result_stream, final_node_name="agent") 
    
        with st.chat_message("AI"):
            ai_message = st.write_stream(cleaned_stream)


        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
        
    except Exception as e:
        st.error("Sorry, an error occurred. Please try again.")
        