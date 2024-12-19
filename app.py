import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains.summarize import load_summarize_chain
import os

# Constants
DB_FAISS_PATH = 'db_faiss'

# Loading the Llama-2 model
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Function to limit the chat history
def get_limited_history(session: str, max_messages: int = 10) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    history = st.session_state.store[session]
    # Limit the number of messages to `max_messages`
    history.messages = history.messages[-max_messages:]
    return history

# Function to summarize chat history
def summarize_history(history: BaseChatMessageHistory, llm) -> str:
    if len(history.messages) > 10:  # Summarize if more than 10 messages
        summarizer = load_summarize_chain(llm)
        return summarizer.run({"input_documents": history.messages})
    return " ".join([msg.content for msg in history.messages])

# Function to truncate input
def truncate_input(input_text: str, max_tokens: int = 512) -> str:
    tokens = input_text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return input_text

# Streamlit app setup
st.title("Medical Bot With Chat History")
st.write("Welcome to the Medical Bot. Ask any medical-related question!")

# Session management
session_id = st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

# Load vector database
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    st.success("Vector database loaded successfully!")
except Exception as e:
    st.error(f"Failed to load vector database: {e}")

# Load the Llama-2 model
llm = load_llm()

# History-aware retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# QA system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Conversational RAG Chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_limited_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# User interaction
user_input = st.text_input("Your question:")
if user_input:
    session_history = get_limited_history(session_id, max_messages=10)
    summarized_history = summarize_history(session_history, llm)
    combined_input = f"{summarized_history}\n{user_input}"
    truncated_input = truncate_input(combined_input, max_tokens=512)

    response = conversational_rag_chain.invoke(
        {"input": truncated_input},
        config={"configurable": {"session_id": session_id}}
    )
    st.write("Assistant:", response['answer'])
