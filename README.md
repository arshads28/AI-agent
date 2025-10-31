🧠 Production-Level Chat Agent with FastAPI and LangGraph

A production-ready, stateful, and extensible AI chat agent built with FastAPI for the backend and LangGraph for the conversational and reasoning core.
This project provides a solid foundation for developers looking to deploy intelligent, memory-aware chatbots capable of dynamic tool usage and multi-model inference.

🚀 Overview

This system combines FastAPI (for a high-performance web server) and LangGraph (for managing conversational state and reasoning flow) to deliver a stateful AI assistant that can:

Maintain context across interactions (via persistent conversation threads)

Dynamically select between OpenAI and Google Gemini models

Use external tools (e.g., Tavily search, Google Custom Search)

Stream responses and tokens to clients via Server-Sent Events (SSE)

✨ Features
🧩 Core Functionality

FastAPI Backend:
Built for high performance and scalability with async endpoints.

LangGraph Integration:
Manages agent state, reasoning, and memory flow for each user session.

Multi-Model Support:
Dynamically select between LLMs like:

gpt-4o (OpenAI)

gemini-1.5-flash-latest (Google Gemini)

Stateful Conversations:
Each thread_id maps to a persistent memory checkpoint in checkpoints.sqlite, allowing users to continue their chats seamlessly.

Tool-Enabled Reasoning:
The agent can autonomously decide to call external tools (like Tavily or Google Search) to enhance responses with real-time information.

Real-Time Streaming (SSE):
Responses and tool invocations are streamed to clients for a fluid, token-by-token chat experience.

Production-Ready Setup:
Environment configuration, structured logging, and dependency management included.

🧰 Tech Stack
Component	Technology
Web Server	FastAPI

Agent Framework	LangGraph

Models	OpenAI GPT, Google Gemini
Database	SQLite (Persistent thread memory)
Streaming	Server-Sent Events (SSE)
Search Tools	Tavily API, Google Custom Search
Logging	Structured JSON logging with loguru / logging

for running 
uvicorn main:app --reload --log-config logging.yaml