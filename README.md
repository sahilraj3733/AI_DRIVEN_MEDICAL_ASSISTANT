
# AI-DRIVEN MEDICAL ASSISTANT

The **AI-DRIVEN MEDICAL ASSISTANT** is a conversational AI designed to provide reliable medical information based on user queries. It is built on a robust architecture that uses a **Retrieval-Augmented Generation (RAG)** pipeline. The core of its logic is a **LangGraph** framework that manages a **ReAct agent** to handle complex conversational flows. This allows the bot to reason and act dynamically by selecting and using the best tools to answer user questions.

---

## 🤖 How the ReAct Agent Works

The **ReAct (Reasoning and Acting)** agent is the brain of your bot. It enables the bot to think and act sequentially by following a cyclical process:

1.  **Thought**: The agent first analyzes the user's query and formulates a plan. For example, if a user mentions symptoms like a cough, the agent's thought might be, "The user is reporting symptoms, so I need to check the retriever tool for possible causes."
2.  **Action**: Based on its thought, the agent decides on the **action** to take. This action must be one of the available tools, such as the `retriever_tool` for internal knowledge or the `tavily_search` tool for web searches.
3.  **Tool Input**: The agent then provides the necessary **input** for the selected tool (e.g., "persistent cough and headache").
4.  **Observation**: The tool executes the action and returns an **observation**, which is the output from the tool (e.g., medical information from the FAISS database).
5.  **Final Answer**: The agent then synthesizes all the information (its initial thought, the tool's action, and the observation) to formulate a comprehensive and safe **final answer** for the user.

This cyclical process is what allows your bot to be both intelligent and reliable.

---

## ✨ Features

* **Intelligent Agentic Behavior**: The bot uses a LangGraph agent to dynamically decide whether to answer directly, perform a knowledge base search, or use a web search tool.
* **Vector Database Integration**: It leverages a **FAISS** index to store and retrieve information on medical conditions, symptoms, and treatments.
* **Real-time Web Search**: The **Tavily Search** tool provides the bot with access to the latest online information, ensuring answers are current.
* **Empathetic and Safe Responses**: The bot is prompted to provide empathetic answers while including crucial disclaimers, reinforcing that it is a medical assistant, not a licensed doctor.
* **Interactive User Interface**: A **Streamlit** frontend allows users to interact with the bot in a clean, chat-based interface.

---

## 🛠️ Prerequisites

Before you can run the bot, you must have the following installed on your system:

* **Python 3.10+**
* A **Groq API Key**
* A **Tavily API Key**

---

## ⚙️ Installation

Follow these steps to set up the project locally.

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/your-username/AI_DRIVEN_MEDICAL_ASSISTANT.git](https://github.com/your-username/AI_DRIVEN_MEDICAL_ASSISTANT.git)
    cd AI_DRIVEN_MEDICAL_ASSISTANT
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**

    Create a file named `.env` in the project's root directory and add your API keys:

    ```ini
    groq_api=YOUR_GROQ_API_KEY
    travly=YOUR_TAVILY_API_KEY
    ```

5.  **Create the FAISS Index**

    Run the *vector embeddings.ipynb* notebook and download the my_faiss_index

---

## 🚀 Getting Started

To launch the chatbot, ensure your virtual environment is active and run the Streamlit application from your terminal:

```bash
streamlit run fronted.py