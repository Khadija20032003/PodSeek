import json
import time
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\helpf\Desktop\DD2477 Project\Github Repo\PodSeek")
CHUNKED_DIR = BASE_DIR / "chunked_podcast_segments"
ENRICHED_META_DIR = BASE_DIR / "enriched_metadata"
OUTPUT_FILE = BASE_DIR / "cleaned_output" / "elastic_ready.jsonl"

# Absolute path to the original TSV file
TSV_FILE = Path(r"C:\Users\helpf\Desktop\DD2477 Project\podcasts-no-audio-13GB\metadata.tsv")

def build_elasticsearch_dataset():
    print("Loading TSV metadata into memory...")
    start_time = time.time()
    
    # Load the base metadata from the TSV
    try:
        df = pd.read_csv(TSV_FILE, sep="\t")
        df.fillna("", inplace=True)
    except FileNotFoundError:
        print(f"ERROR: Could not find TSV file at {TSV_FILE}")
        return
        
    # Create a super-fast lookup dictionary
    tsv_lookup = {}
    for _, row in df.iterrows():
        prefix = str(row["episode_filename_prefix"])
        tsv_lookup[prefix] = {
            "show_name": str(row["show_name"]),
            "episode_name": str(row["episode_name"]),
            "publisher": str(row["publisher"]),
            "rss_link": str(row["rss_link"])
        }
    
    print("Denormalizing chunks with metadata...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    total_chunks = 0
    missing_enriched_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        
        for chunk_file in CHUNKED_DIR.glob("*.json"):
            with open(chunk_file, "r", encoding="utf-8") as cf:
                try:
                    chunk_data = json.load(cf)
                except json.JSONDecodeError:
                    continue
                    
            file_id = chunk_data.get("file_id", "")
            segments = chunk_data.get("segments", [])
            
            if not file_id or not segments:
                continue
                
            # Grab the guaranteed metadata from our TSV lookup
            base_meta = tsv_lookup.get(file_id, {})
            
            # Check folder purely for the scraped "category"
            category = "Uncategorized"
            meta_file = ENRICHED_META_DIR / f"{file_id}.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as mf:
                    enriched_dict = json.load(mf)
                    # Grab the category, default to "Uncategorized" if blank
                    category = enriched_dict.get("show_category", "Uncategorized")
                    if not category: 
                        category = "Uncategorized"
            else:
                missing_enriched_count += 1
            
            # Merge everything
            for segment in segments:
                elastic_doc = {
                    "elastic_id": f"{file_id}_{segment.get('segment_id')}", 
                    "file_id": file_id,
                    "segment_id": segment.get("segment_id"),
                    "text": segment.get("text", ""),
                    "start_time": segment.get("start"),
                    "end_time": segment.get("end"),
                    
                    # Real metadata injected here
                    "show_name": base_meta.get("show_name", "Unknown Show"),
                    "episode_name": base_meta.get("episode_name", "Unknown Episode"),
                    "publisher": base_meta.get("publisher", "Unknown Publisher"),
                    "category": category,
                    "rss_link": base_meta.get("rss_link", "")
                }
                
                outfile.write(json.dumps(elastic_doc) + "\n")
                total_chunks += 1

    elapsed = round(time.time() - start_time, 2)
    print("\n========================================")
    print(f"DATASET BUILD COMPLETE in {elapsed} seconds.")
    print(f"Total Elasticsearch-ready chunks generated: {total_chunks}")
    print(f"ℹNote: {missing_enriched_count} episodes didn't have scraped categories, but all TSV metadata was saved!")
    print(f"Final index file ready at: {OUTPUT_FILE}")
    print("========================================")

if __name__ == "__main__":
    build_elasticsearch_dataset()