# AI Chatbot

A professional, fully-functional AI chatbot built with **Python**, **Streamlit**, and the **Groq API**.

## Features

- Modern, responsive, dark-themed UI
- Secure API key management (never hardcoded or logged)
- Dynamic model fetching from Groq API
- Latest Groq models: Llama 3.1/3.3, GPT-OSS 120B/20B, Qwen3.8, MiniMax M2.7, Muse Spark
- Predefined system roles: General Assistant, Python Tutor, ML Expert, AI & Robotics Trainer, Data Analyst
- Custom system prompt support
- Configurable parameters: temperature, max tokens, top-p, frequency & presence penalty
- Streaming responses with live cursor
- Conversation history with session state
- Export conversations as TXT or JSON
- New chat, clear history, reset settings
- Graceful API error handling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run chatbot.py
```

Get your free Groq API key at [https://console.groq.com/keys](https://console.groq.com/keys), paste it in the sidebar, click **Validate**, and start chatting.

## Requirements

- Python 3.10+
- Streamlit
- Groq Python SDK