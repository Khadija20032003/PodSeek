import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.getenv("GROQ_API_KEY").strip() 

llm = ChatGroq(
    temperature=0,
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant"
)

# --- THE MOCK DATABASE ---
# We will replace this with Elasticsearch call later
def mock_elasticsearch_search(user_query):
    return [
        {
            "podcast_id": "Lex Fridman #300", 
            "start": "45:10", 
            "end": "47:10", 
            "text": "The Higgs Boson is often called the God particle, but physicists actually dislike that term. It gives mass to other fundamental particles."
        },
        {
            "podcast_id": "Science Daily Ep 12", 
            "start": "12:00", 
            "end": "14:00", 
            "text": "When the Large Hadron Collider found the Higgs Boson in 2012, it completed the standard model of particle physics."
        }
    ]

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
            search_results = mock_elasticsearch_search(query)
            
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