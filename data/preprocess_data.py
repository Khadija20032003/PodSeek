"""
preprocess_data.py — Extracts clean transcript segments from raw podcast JSON files.

Uses multiprocessing to process files across all CPU cores for maximum speed.
"""

import os
import sys
import json
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Import config from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATASET_DIR,
    CLEANED_OUTPUT_DIR,
    CLEANED_DATA_FILE,
    GROUPED_DATA_FILE,
    MAX_WORKERS_CPU,
)


def process_single_file(filepath: str) -> list:
    """
    Process one JSON file and return a list of cleaned chunk dicts.
    This function runs in a worker process — no shared state, no file writes.
    """
    filename = os.path.basename(filepath)
    chunks = []

    try:
        with open(filepath, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        for result in data.get("results", []):
            try:
                alt = result["alternatives"][0]
                transcript = alt.get("transcript", "").strip()
                words_array = alt.get("words", [])

                if not transcript or not words_array:
                    continue

                start_sec = float(words_array[0]["startTime"].replace("s", ""))
                end_sec = float(words_array[-1]["endTime"].replace("s", ""))

                chunks.append(
                    json.dumps({
                        "file_id": filename,
                        "text": transcript,
                        "start_time": start_sec,
                        "end_time": end_sec,
                    })
                )

            except (KeyError, IndexError, ValueError):
                continue

    except Exception:
        pass

    return chunks


def collect_file_paths(input_folder: str) -> list:
    """Walk the directory tree once and collect all .json file paths."""
    paths = []
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".json"):
                paths.append(os.path.join(root, filename))
    return paths


def group_by_episode():
    """
    Reads cleaned_data.jsonl (one line per segment) and groups segments
    by episode into extracted_podcasts.jsonl (one line per episode with
    a segments array). This is the format PodcastProcessor and
    TranscriptSegmenter expect.
    """
    print("\nGrouping segments by episode...")
    start_time = time.time()

    episodes = {}
    with open(str(CLEANED_DATA_FILE), "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            file_id = data["file_id"]
            if file_id not in episodes:
                episodes[file_id] = []
            episodes[file_id].append({
                "text": data["text"],
                "start": data["start_time"],
                "end": data["end_time"],
            })

    with open(str(GROUPED_DATA_FILE), "w", encoding="utf-8") as f:
        for file_id, segments in episodes.items():
            record = {"file_id": file_id, "segments": segments}
            f.write(json.dumps(record) + "\n")

    elapsed = round(time.time() - start_time, 2)
    print(f"Grouped {sum(len(s) for s in episodes.values())} segments into {len(episodes)} episodes in {elapsed}s")
    print(f"Output: {GROUPED_DATA_FILE}")


def main():
    CLEANED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Collect all file paths ---
    print(f"Scanning '{RAW_DATASET_DIR}' for JSON files...")
    start_time = time.time()
    file_paths = collect_file_paths(str(RAW_DATASET_DIR))
    total_files = len(file_paths)
    print(f"Found {total_files} JSON files in {time.time() - start_time:.1f}s\n")

    # --- Phase 2: Process files in parallel ---
    num_workers = min(cpu_count(), MAX_WORKERS_CPU)
    print(f"Processing with {num_workers} workers...")
    start_time = time.time()

    total_chunks = 0
    batch_size = 500
    write_buffer = []

    with open(str(CLEANED_DATA_FILE), "w", encoding="utf-8") as outfile:
        with Pool(processes=num_workers) as pool:
            for i, chunk_lines in enumerate(
                pool.imap_unordered(process_single_file, file_paths, chunksize=batch_size)
            ):
                if chunk_lines:
                    write_buffer.extend(chunk_lines)
                    total_chunks += len(chunk_lines)

                if len(write_buffer) >= 10_000:
                    outfile.write("\n".join(write_buffer) + "\n")
                    write_buffer.clear()

                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (total_files - i - 1) / rate / 60
                    print(
                        f"  Processed {i+1}/{total_files} files "
                        f"({total_chunks} chunks) — "
                        f"~{remaining:.1f} min remaining"
                    )

        if write_buffer:
            outfile.write("\n".join(write_buffer) + "\n")

    elapsed = round((time.time() - start_time) / 60, 2)
    print("\n========================================")
    print(f"DONE! Processed {total_files} files in {elapsed} minutes.")
    print(f"Total clean chunks saved: {total_chunks}")
    print(f"Workers used: {num_workers}")
    print(f"Output: {CLEANED_DATA_FILE}")
    print("========================================")

    # --- Phase 3: Group segments by episode ---
    group_by_episode()


if __name__ == "__main__":
    main()