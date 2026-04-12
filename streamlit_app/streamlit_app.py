import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import sys
from pathlib import Path
from elasticsearch import Elasticsearch

load_dotenv()

api_key = os.getenv("GROQ_API_KEY").strip() 

llm = ChatGroq(
    temperature=0,
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant"
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ES_HOST
from es_search.search import search as hybrid_search
from es_search.search import format_time

def elasticsearch_search(user_query: str, top_k: int = 5):
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {ES_HOST}")
    result = hybrid_search(es, user_query, top_k=top_k)
    return result

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
            try:
                es_result = elasticsearch_search(query, top_k=5)
                hits = es_result.get("hits", [])
                search_results = []
                for hit in hits:
                    src = hit.get("_source", {})
                    start = format_time(float(src.get("start_time", 0.0)))
                    end = format_time(float(src.get("end_time", 0.0)))
                    show_name = src.get("show_name", "Unknown")
                    episode_name = src.get("episode_name", "Unknown")
                    podcast_id = f"{show_name} — {episode_name}"

                    highlight_texts = hit.get("highlight", {}).get("text", [])
                    if highlight_texts:
                        snippet = " ".join(str(h) for h in highlight_texts)
                    else:
                        snippet = str(src.get("text", ""))

                    search_results.append(
                        {
                            "podcast_id": podcast_id,
                            "start": start,
                            "end": end,
                            "text": snippet,
                        }
                    )
            except Exception as e:
                st.error(str(e))
                search_results = []
            
            # 2. Format the search results into a single string for the LLM
            context_string = ""
            for result in search_results:
                context_string += f"\n- [{result['podcast_id']} | {result['start']}-{result['end']}]: {result['text']}"
            
            # 3. Create the final prompt and run the LLM
            formatted_prompt = rag_prompt.format(question=query, context=context_string)
            response = llm.invoke(formatted_prompt)
            
            # 4. Display the results
            st.subheader("Podcast Results:")
            st.write(response.content)
            
            st.subheader("Raw Audio Chunks Found")
            for result in search_results:
                st.info(f"**{result['podcast_id']}** ({result['start']} - {result['end']})\n\n{result['text']}")
    else:
        st.warning("Please enter a search query first.")