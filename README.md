# PodSeek — Podcast Transcript Search Engine

A search engine over the Spotify podcast transcript dataset. Uses Elasticsearch with BM25 keyword search over ~937k chunked podcast transcript segments.

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
│   ├── PodcastMetadata.py             # Pydantic metadata model (deprecated)
│   └── podcasts-no-audio-13GB/        # Raw Spotify dataset (not in git)
├── es_search/                         # Elasticsearch indexing + search
│   ├── docker-compose.yml             # Elasticsearch + Kibana containers
│   ├── index_chunks.py                # Bulk-indexes chunks into Elasticsearch
│   └── search.py                      # CLI search with BM25
└── README.md
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- The Spotify podcast transcript dataset (`podcasts-no-audio-13GB/`)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the data pipeline

From the `data/` directory:

```bash
cd data

# Step 1: Extract transcript segments and group them by episode
python transcript_extractor.py

# Step 2: Build full transcripts and segment files
python podcast_creator.py

# Step 3: Chunk segments with hierarchical parent/child chunking
# Output: one JSON per line in PREPROCESSING_FILES/chunked_podcast_segments/*_chunked.jsonl
python transcript_segmenter.py

# Step 4: Enrich metadata via RSS feeds (optional, takes a while)
python rss_enrichment.py

# Step 5: Merge chunks + metadata into final index file
python elastic_data_creator.py

# Step 6 (optional but recommended): Generate embeddings for vector search
# Note: A small .progress sidecar file is written next to the output JSONL to support resuming.
python embedding_generator.py
```

Alternatively, you can run the full preprocessing pipeline in one go:

```bash
cd data
python pipeline.py
```

### 3. Start Elasticsearch

```bash
cd es_search
docker compose up -d
```

Verify it's running (wait ~30 seconds for startup):

```bash
curl http://localhost:9200
```

### 4. Index the data

```bash
cd es_search
python index_chunks.py --recreate
```

### 5. Search

```bash
cd es_search

# BM25 keyword search
python search.py "large language models"

# Filter by category or show
python search.py "recipe" --category Food
python search.py "neural networks" --show "The AI Podcast"

# More results
python search.py "climate change" --top 20

# Raw JSON output
python search.py "carbonara" --json
```

### 6. Kibana

Open http://localhost:5601 in your browser. Go to **Dev Tools** and run:

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

| Field        | Type         | Purpose                                 |
| ------------ | ------------ | --------------------------------------- |
| file_id      | keyword      | Episode identifier                      |
| chunk_id     | keyword      | Unique child chunk id (parent+child)    |
| text         | text         | Child chunk text (embedded + searched)  |
| start_time   | float        | Chunk start in audio (seconds)          |
| end_time     | float        | Chunk end in audio (seconds)            |
| parent_id    | keyword      | Parent chunk id (120s window)           |
| parent_text  | text         | Parent text returned to the LLM         |
| parent_start_time | float   | Parent start in audio (seconds)         |
| parent_end_time   | float   | Parent end in audio (seconds)           |
| show_name    | text/keyword | Podcast show name (searchable + filter) |
| episode_name | text/keyword | Episode title (searchable + filter)     |
| publisher    | text/keyword | Publisher name                          |
| category     | keyword      | Show category (e.g. Technology, Food)   |
| rss_link     | keyword      | RSS feed URL                            |

## Configuration

All paths and settings are defined in `config.py` at the project root. Edit this file to change dataset locations, chunk parameters, or Elasticsearch host.

The transcript extraction step scans `TRANSCRIPTS_JSON_DIR` (defaults to `data/podcasts-no-audio-13GB/podcasts-transcripts-6to7`) so you can iterate on a smaller subset without touching the TSV metadata path.

## Stopping Elasticsearch

```bash
cd es_search

# Stop containers (keeps indexed data)
docker compose down

# Stop and delete all indexed data
docker compose down -v
```
