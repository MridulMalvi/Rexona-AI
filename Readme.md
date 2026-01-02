# 🤖 Gemini-Powered Stateful Chatbot

A modern, stateful chatbot application built with **Streamlit**, **LangChain**, and **Google Gemini**. This assistant features persistent chat history and a modular backend using **LangGraph** to manage conversation threads.

---

## 🌟 Features

- **Stateful Conversations:** Maintains context across multiple turns using LangChain's message history.
- **Thread Management:** Create "New Chats" or switch between "Previous Conversations" via the sidebar.
- **Real-time Streaming:** Watch the assistant's response generate word-by-word for a more interactive experience.
- **Google Gemini Integration:** Utilizes the high-performance Gemini API for intelligent and fast responses.
- **Session Persistence:** Automatically saves and loads chat history based on unique Thread IDs.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Orchestration:** LangChain & LangGraph
- **Model:** Google Gemini (Generative AI)
- **Environment Management:** Python-Dotenv

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or higher
- A Google AI Studio API Key. [Get one here](https://aistudio.google.com/).

### 2. Installation
Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Install requirements
pip install -r requirements.txt
```

3. Environment Setup
Create a .env file in the root directory and add your API key:
````
GOOGLE_API_KEY=your_gemini_api_key_here
````
4. Run the Application
````
streamlit run app.py
````
📁 Project Structure
->app.py: The main Streamlit interface, handling UI logic and session state.
->backend.py: Contains the chatbot logic, LLM initialization, and state graph.
->requirements.txt: Lists all Python libraries required for the project.
->.env: Stores sensitive API keys (do not commit to GitHub).
