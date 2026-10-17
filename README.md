# Personal-AI-with-Long-Term-Memory-RAG-Chatbot-

# Personal AI with Long-Term Memory — RAG Chatbot

## Overview

**Personal AI with Long-Term Memory** is an intelligent chatbot designed to maintain context across conversations by storing, retrieving, and utilizing relevant information from previous interactions. Built using **LangChain** and **Retrieval-Augmented Generation (RAG)**, the system enables more personalized and context-aware responses.

The chatbot uses **hybrid search with BM25** to improve memory retrieval by combining keyword-based search with semantic similarity search. This helps retrieve relevant information based on both exact keyword matches and contextual meaning, making conversations more coherent and personalized over time.

## Key Features

- **Long-Term Memory:** Stores useful information from past conversations for future reference.
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant memories to provide context-aware responses.
- **Hybrid Search:** Combines BM25 keyword retrieval with semantic search to improve retrieval relevance.
- **Conversation Context:** Uses retrieved memories to maintain continuity across multiple interactions.
- **LangChain Integration:** Uses LangChain components to build and manage the retrieval and response-generation pipeline.
- **Personalized Responses:** Generates responses based on relevant information from previous conversations.

## Tech Stack

- **Python** — Core programming language
- **LangChain** — LLM orchestration and RAG pipeline
- **BM25** — Keyword-based information retrieval
- **Vector Store** — Semantic memory retrieval
- **Large Language Model (LLM)** — Context-aware response generation

## How It Works

1. **User Interaction:** The user sends a message to the chatbot.
2. **Memory Extraction:** Useful information is identified from the conversation for long-term storage.
3. **Memory Storage:** Extracted memories are stored in a searchable memory store.
4. **Hybrid Retrieval:** BM25 keyword search and semantic similarity search retrieve relevant memories.
5. **Context Augmentation:** Retrieved memories are added to the current query as context.
6. **Response Generation:** The LLM generates a personalized response using the current message and relevant past information.

## Architecture

```text
User Query
    |
    v
Memory Retrieval
    |
    v
Hybrid Search
(BM25 + Semantic Search)
    |
    v
Relevant Memories
    |
    v
Context Augmentation
    |
    v
LLM via LangChain
    |
    v
Personalized Response
    |
    v
Memory Extraction & Storage
```

## Project Goal

The goal of this project is to explore how long-term memory can make AI assistants more useful by enabling them to remember relevant information, connect related conversations, and deliver personalized responses instead of treating every interaction as an isolated session.

## Future Improvements

- Implement memory summarization and deduplication.
- Add memory management, updating, and deletion capabilities.
- Improve retrieval through reranking and relevance scoring.
- Develop a web interface for interactive conversations.
- Add support for persistent, multi-session memory.

## Getting Started

Clone the repository:

```bash
git clone <your-repository-url>
cd Personal-AI-with-Long-Term-Memory-RAG-Chatbot
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Configure your API keys in a `.env` file and run the application using the project's entry-point script.

## License

This project is intended for learning and experimentation with LLMs, RAG, hybrid retrieval, and long-term conversational memory.
