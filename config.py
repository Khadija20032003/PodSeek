"""
config.py — Central configuration for the PodSeek project.

All paths, settings, and parameters in one place.
Every other script imports from here instead of defining its own paths.
"""

import os
from pathlib import Path

# --- PROJECT STRUCTURE ---
PROJECT_ROOT = Path(__file__).resolve().parent          # PodSeek/
DATA_DIR = PROJECT_ROOT / "data"

# --- RAW DATASET ---
RAW_DATASET_DIR = DATA_DIR / "podcasts-no-audio-13GB"
TSV_FILE = RAW_DATASET_DIR / "metadata.tsv"

# --- PREPROCESSING OUTPUTS ---
CLEANED_OUTPUT_DIR = DATA_DIR / "cleaned_output"
CLEANED_DATA_FILE = CLEANED_OUTPUT_DIR / "cleaned_data.jsonl"
GROUPED_DATA_FILE = CLEANED_OUTPUT_DIR / "extracted_podcasts.jsonl"
ELASTIC_READY_FILE = CLEANED_OUTPUT_DIR / "elastic_ready.jsonl"

# --- INTERMEDIATE DIRECTORIES ---
TRANSCRIPT_DIR = DATA_DIR / "cleaned_full_podcast_transcript"
SEGMENTS_DIR = DATA_DIR / "cleaned_podcast_segments"
CHUNKED_DIR = DATA_DIR / "chunked_podcast_segments"
ENRICHED_META_DIR = DATA_DIR / "enriched_metadata"

# --- CHUNKING PARAMETERS ---
TARGET_CHUNK_SECONDS = 120
OVERLAP_SECONDS = 30

# --- MULTIPROCESSING ---
MAX_WORKERS_CPU = 8         # for preprocess_data.py (CPU-bound)
MAX_WORKERS_NETWORK = 20    # for RSSEnrichmentPipeline.py (network-bound)

# --- ELASTICSEARCH ---
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "podcast_chunks")