"""
config.py — Central configuration for the PodSeek project.

All paths, settings, and parameters in one place.
Every other script imports from here instead of defining its own paths.
"""

import os
from pathlib import Path

# --- PROJECT STRUCTURE ---
PROJECT_ROOT = Path(__file__).resolve().parent  # PodSeek/
DATA_DIR = PROJECT_ROOT / "data"

# It's best practice to make these subdirectories Path objects anchored to DATA_DIR
PREPROCESSING_FILES = DATA_DIR / "preprocessing_files"
FINAL_OUTPUT = DATA_DIR / "final_output"
CONTEXTBUILDING_FILES = DATA_DIR / "context_building_files"

# --- RAW DATASET ---
RAW_DATASET_DIR = DATA_DIR / "podcasts-no-audio-13GB"
TSV_FILE = RAW_DATASET_DIR / "metadata.tsv"

# --- PREPROCESSING OUTPUTS ---
CLEANED_OUTPUT_DIR = DATA_DIR / "cleaned_output"

# Now that CONTEXTBUILDING_FILES, etc., are Path objects, the / operator works perfectly
CLEANED_DATA_FILE = CONTEXTBUILDING_FILES / "cleaned_data.jsonl"
GROUPED_DATA_FILE = PREPROCESSING_FILES / "extracted_podcasts.jsonl"
ELASTIC_READY_FILE = FINAL_OUTPUT / "elastic_ready.jsonl"

# --- INTERMEDIATE DIRECTORIES ---
# Note: I removed DATA_DIR / from here since CONTEXTBUILDING_FILES already includes it
TRANSCRIPT_DIR = CONTEXTBUILDING_FILES / "full_podcast_transcript"
SEGMENTS_DIR = CONTEXTBUILDING_FILES / "podcast_segments"
CHUNKED_DIR = PREPROCESSING_FILES / "chunked_podcast_segments"
ENRICHED_META_DIR = PREPROCESSING_FILES / "enriched_metadata"

# --- CHUNKING PARAMETERS ---
TARGET_CHUNK_SECONDS = 120
OVERLAP_SECONDS = 30

# --- MULTIPROCESSING ---
MAX_WORKERS_CPU = 8  # for preprocess_data.py (CPU-bound)
MAX_WORKERS_NETWORK = 20  # for RSSEnrichmentPipeline.py (network-bound)


# --- EMBEDDING PARAMETERS ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 128
EMBEDDING_INPUT_FILE = ELASTIC_READY_FILE
EMBEDDING_OUTPUT_FILE = FINAL_OUTPUT / "elastic_ready_with_embeddings.jsonl"


# --- ELASTICSEARCH ---
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "podcast_chunks")
