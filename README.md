# 🎙️ PodSeek

This repo contains the frontend and LLM connection for our Podcast Search project. We are using **Streamlit** for the UI and **Groq** (hosting Llama 3) for RAG.

## Quick Start Guide

### Install Dependencies
Install all required Python libraries from the instruction manual:

```bash
pip install -r requirements.txt
```

### API Key Setup
We are using Groq to access the open-source Llama 3 model. You need your own API key to run the app.

- Go to console.groq.com and sign up for a free account.

- Go to the API Keys tab and generate a new key.

#### IMPORTANT: Before generating a key, go to the Playground tab on the left. Select the llama-3.1-8b-instant model and interact with it. My API key got activated after I did this.

- In the root of the project folder, find the file named .env.example

- Paste your key and change the file's name to .env


### Run the Application

```bash
streamlit run app.py
```