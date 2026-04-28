# PodSeek — Podcast Transcript Search Engine

A RAG-powered search engine over the Spotify podcast transcript dataset. Combines Elasticsearch with BM25 keyword search and kNN semantic search over ~937k chunked podcast transcript segments, with LLM-powered answer generation via Groq (Llama 3).

## Project Structure

```
PodSeek/
├── config.py                          # Central configuration (all paths, settings)
├── requirements.txt                   # Python dependencies
├── data/                              # Data preprocessing pipeline
│   ├── transcript_extractor.py        # Raw JSON → cleaned segments + grouped episodes
│   ├── podcast_creator.py             # Grouped episodes → full transcripts + segment files
│   ├── transcript_segmenter.py        # Segments → hierarchical parent/child chunks (RAG)
│   ├── rss_enrichment.py              # Scrapes RSS feeds for categories + metadata
│   ├── elastic_data_creator.py        # Merges chunks + metadata → elastic_ready.jsonl
│   ├── embedding_generator.py         # Adds dense vectors → elastic_ready_with_embeddings.jsonl
│   └── podcasts-no-audio-13GB/        # Raw Spotify dataset (not in git)
├── es_search/                         # Elasticsearch indexing + search
│   ├── docker-compose.yml             # Elasticsearch + Kibana containers
│   ├── index_chunks.py                # Bulk-indexes chunks into Elasticsearch
│   └── search.py                      # CLI search with BM25
├── es_eval/                           # Evaluation scripts
│   ├── rag_eval.py                    # RAGAS evaluation (faithfulness + relevancy)
│   └── .env.example                   # Groq API key for LLM judges
├── streamlit_app/                     # Frontend + LLM answer generation
│   ├── streamlit_app.py               # Streamlit UI with RAG pipeline
│   ├── requirements.txt               # Streamlit-specific dependencies
│   ├── .env.example                   # Example environment file
│   └── .env                           # Groq API key (not in git — see step 4)
├── benchmark_latency.py               # Latency benchmarking
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- The Spotify podcast transcript dataset (`podcasts-no-audio-13GB/`)
- A Groq API key (free — see step 4 below)

## Quick Start

### 1. Set up the environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r streamlit_app/requirements.txt
```

### 2. Run the data pipeline

You can run each stage individually or all at once with `python data/pipeline.py`.

```bash
cd data

# Step 1: Extract transcript segments and group by episode
python transcript_extractor.py

# Step 2: Build full transcripts and segment files
python podcast_creator.py

# Step 3: Chunk segments with hierarchical parent/child chunking
python transcript_segmenter.py

# Step 4: Enrich metadata via RSS feeds (optional, takes a while)
python rss_enrichment.py

# Step 5: Merge chunks + metadata into final index file
python elastic_data_creator.py

# Step 6: Generate embeddings for vector search
python embedding_generator.py
```

**Alternatively**, you can download pre-generated embeddings from [this Google Drive folder](https://drive.google.com/drive/folders/1e-xhHxDCSd7QM_QfIU5SBLG_kLXbdyZC?usp=sharing). Download both `elastic_ready_with_embeddings.jsonl` and `elastic_ready_with_embeddings.jsonl.progress`, and place them in `data/final_output/`.

### 3. Start Elasticsearch and index the data

```bash
cd es_search
docker compose up -d
```

Wait ~30 seconds for startup, then verify:

```bash
curl http://localhost:9200
```

Index the chunks:

```bash
python index_chunks.py --recreate
```

### 4. Set up the Groq API key

The app uses Groq to access the Llama 3 model for answer generation.

1. Go to [console.groq.com](https://console.groq.com) and sign up for a free account.
2. Go to the **API Keys** tab and generate a new key.
3. **Important:** After generating your key, go to the **Playground** tab, select the `llama-3.1-8b-instant` model, and send a message. This activates the key.
4. Create a `.env` file in `streamlit_app/`:

```bash
# streamlit_app/.env
GROQ_API_KEY=your_key_here
```

5. Do the same for `es_eval/` (needed to run RAG evaluation):

```bash
# es_eval/.env
GROQ_API_KEY=your_key_here
```

### 5. Run the application

```bash
streamlit run streamlit_app/streamlit_app.py
```

### 6. Run evaluation (optional)

The RAG evaluation uses RAGAS with Groq-hosted LLMs as judges to measure faithfulness and answer relevancy:

```bash
cd es_eval
python rag_eval.py
```

Make sure Elasticsearch is running and the `.env` file with your Groq key is in the `es_eval/` folder.

### 7. CLI search (optional)

You can also search directly from the command line without the Streamlit UI:

```bash
cd es_search

python search.py "large language models"
python search.py "recipe" --category Food
python search.py "neural networks" --show "The AI Podcast"
python search.py "climate change" --top 20
python search.py "carbonara" --json
```

### 8. Kibana (optional)

Open [http://localhost:5601](http://localhost:5601) and go to **Dev Tools**:

```
GET podcast_chunks/_search
{
  "query": {
    "match": {
      "text": "artificial intelligence"
    }
  }
}
```

## Elasticsearch Index Schema

| Field | Type | Purpose |
|---|---|---|
| file_id | keyword | Episode identifier |
| chunk_id | keyword | Unique child chunk id |
| text | text | Child chunk text (embedded + searched) |
| start_time / end_time | float | Chunk boundaries in audio (seconds) |
| embedding | dense_vector (384) | Sentence-transformer vector for kNN search |
| parent_id | keyword | Parent chunk id (120s window) |
| parent_text | text | Parent context returned to the LLM |
| parent_start_time / parent_end_time | float | Parent boundaries in audio (seconds) |
| show_name | text/keyword | Podcast show name (searchable + filterable) |
| episode_name | text/keyword | Episode title |
| publisher | text/keyword | Publisher name |
| category | keyword | Show category (e.g. Technology, Food) |
| rss_link | keyword | RSS feed URL |

## Configuration

All paths and settings are in `config.py`. Edit this file to change dataset locations, chunk parameters (parent/child window sizes), or the Elasticsearch host.

## Stopping Elasticsearch

```bash
cd es_search
docker compose down       # Stop containers (keeps indexed data)
docker compose down -v    # Stop and delete all indexed data
```
