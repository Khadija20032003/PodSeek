# PodSeek — Podcast Transcript Search Engine

A search engine over the Spotify podcast transcript dataset. Uses Elasticsearch with BM25 keyword search over ~937k chunked podcast transcript segments.

## Project Structure

```
PodSeek/
├── config.py                          # Central configuration (all paths, settings)
├── requirements.txt                   # Python dependencies
├── data/                              # Data preprocessing pipeline
│   ├── preprocess_data.py             # Raw JSON → cleaned segments (parallelized)
│   ├── PodcastProcessor.py            # Grouped segments → full transcripts + segment files
│   ├── TranscriptSegmenter.py         # Segments → 120s chunks with 30s overlap
│   ├── RSSEnrichmentPipeline.py       # Scrapes RSS feeds for categories + metadata
│   ├── build_elastic_index.py         # Merges chunks + metadata → elastic_ready.jsonl
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

# Step 1: Extract segments from raw JSON files
python preprocess_data.py

# Step 2: Build full transcripts and segment files
python PodcastProcessor.py

# Step 3: Chunk segments with sliding window
python TranscriptSegmenter.py

# Step 4: Enrich metadata via RSS feeds (optional, takes a while)
python RSSEnrichmentPipeline.py

# Step 5: Merge chunks + metadata into final index file
python build_elastic_index.py
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
| segment_id   | integer      | Chunk number within episode             |
| text         | text         | Full-text searchable transcript chunk   |
| start_time   | float        | Chunk start in audio (seconds)          |
| end_time     | float        | Chunk end in audio (seconds)            |
| show_name    | text/keyword | Podcast show name (searchable + filter) |
| episode_name | text/keyword | Episode title (searchable + filter)     |
| publisher    | text/keyword | Publisher name                          |
| category     | keyword      | Show category (e.g. Technology, Food)   |
| rss_link     | keyword      | RSS feed URL                            |

## Configuration

All paths and settings are defined in `config.py` at the project root. Edit this file to change dataset locations, chunk parameters, or Elasticsearch host.

## Stopping Elasticsearch

```bash
cd es_search

# Stop containers (keeps indexed data)
docker compose down

# Stop and delete all indexed data
docker compose down -v
```
