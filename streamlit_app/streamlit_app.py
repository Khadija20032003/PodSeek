import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import config and es_search
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ES_HOST
from es_search.search import search, format_time
from elasticsearch import Elasticsearch
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.getenv("GROQ_API_KEY").strip() 

llm = ChatGroq(
    temperature=0,
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant"
)

# --- ELASTICSEARCH CONNECTION ---
@st.cache_resource
def get_es_client():
    return Elasticsearch(ES_HOST)

es_client = get_es_client()

# --- THE LLM PROMPT TEMPLATE ---
rag_prompt = PromptTemplate.from_template("""
You are a helpful podcast assistant. Answer the user's question by synthesizing ALL relevant information from the provided podcast transcripts. 

CRITICAL INSTRUCTIONS:
1. Use ONLY the provided transcripts.
2. If the transcripts do not contain the answer, say "I cannot find the answer in the current podcast database."
3. You MUST cite EVERY podcast ID and timestamp that contributed to your answer. If multiple podcasts discuss the topic, you must include information from all of them.

User Question: {question}

Podcast Transcripts:
{context}

Answer:
""")

# --- STREAMLIT UI ---
st.set_page_config(page_title="PodSeek", page_icon="🎙️")
st.title("🎙️ PodSeek")
st.write("Search the database to find exactly where a topic was discussed.")

# The Search Bar
query = st.text_input("What topic are you looking for? (e.g., 'Higgs Boson')")

# The Search Button Action
if st.button("Search Podcasts"):
    if query:
        with st.spinner("Searching database and reading transcripts..."):
            
            # 1. Get results from the database
            search_response = search(es_client, query, top_k=5)
            search_results = search_response["hits"]
            
            # 2. Format the search results into a single string for the LLM
            context_string = ""
            for result in search_results:
                src = result["_source"]
                episode = src.get("episode_name", "Unknown Episode")
                show = src.get("show_name", "Unknown Show")
                start = format_time(src.get("start_time", 0))
                end = format_time(src.get("end_time", 0))
                text_content = src.get("parent_text", src.get("text", "")) # Parent text holds more context
                
                context_string += f"\n- [{show} - {episode} | {start}-{end}]: {text_content}"
            
            # 3. Create the final prompt and run the LLM
            formatted_prompt = rag_prompt.format(question=query, context=context_string)
            response = llm.invoke(formatted_prompt)
            
            # 4. Display the results
            st.subheader("Podcast Results:")
            st.write(response.content)
            
            st.subheader("Raw Podcast Chunks Found")
            # Suggestion UI if exists
            suggestion = search_response.get("suggestion")
            if suggestion and suggestion.lower() != query.lower():
                st.info(f"Did you mean: **{suggestion}**?")

            if not search_results:
                st.warning("No relevant podcast chunks found.")
            else:
                for result in search_results:
                    src = result["_source"]
                    score = result["_score"]
                    episode = src.get("episode_name", "Unknown Episode")
                    show = src.get("show_name", "Unknown Show")
                    start = format_time(src.get("start_time", 0))
                    end = format_time(src.get("end_time", 0))
                    
                    # Highlight logic
                    highlights = result.get("highlight", {}).get("text", [])
                    if highlights:
                        display_text = "... " + " ... ".join(highlights) + " ..."
                        # Markdown might break if not careful with **, but streamlit usually handles it
                    else:
                        display_text = src.get("text", "")[:300] + "..."
                    
                    st.info(f"**{show}**: {episode} ({start} - {end}) [Score: {score:.2f}]\n\n{display_text}")
    else:
        st.warning("Please enter a search query first.")