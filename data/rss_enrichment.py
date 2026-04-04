"""
RSSEnrichmentPipeline.py — Enriches podcast metadata by scraping RSS feeds.

Extracts show categories, publication dates, audio URLs, and clean summaries.
Uses concurrent requests to fetch multiple RSS feeds in parallel.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import feedparser
import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TSV_FILE, ENRICHED_META_DIR, MAX_WORKERS_NETWORK

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class RSSEnrichmentPipeline:
    """
    Downloads RSS feeds per show, extracts episode-level data,
    cleans HTML, and saves enriched metadata locally.
    Uses ThreadPoolExecutor to fetch multiple feeds concurrently.
    """

    def __init__(
        self, tsv_path: Path, output_dir: Path, max_workers: int = MAX_WORKERS_NETWORK
    ):
        self.tsv_path = tsv_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Podcast Research Project; contact@example.com)"
            }
        )

    def _clean_html(self, raw_html: str) -> str:
        """Removes HTML tags and normalizes whitespace."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text(separator=" ", strip=True)

    def process_dataset(self) -> None:
        """Reads the TSV, groups by show, and processes feeds concurrently."""
        logging.info("Loading TSV for grouping...")
        df = pd.read_csv(self.tsv_path, sep="\t")
        df.fillna("", inplace=True)

        grouped_shows = df.groupby("rss_link")

        tasks = [
            (rss_url, episodes_df)
            for rss_url, episodes_df in grouped_shows
            if rss_url and str(rss_url).startswith("http")
        ]

        total = len(tasks)
        logging.info(f"Found {total} unique RSS feeds to fetch")

        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_show, rss_url, episodes_df
                ): rss_url
                for rss_url, episodes_df in tasks
            }

            for future in as_completed(futures):
                rss_url = futures[future]
                completed += 1

                try:
                    future.result()
                except Exception as e:
                    failed += 1
                    # logging.error(f"Failed: {rss_url} — {e}")

                if completed % 1000 == 0:
                    logging.info(
                        f"  Progress: {completed}/{total} feeds ({failed} failed)"
                    )

        logging.info(f"\nDONE! Processed {completed} feeds ({failed} failed)")

    def _process_single_show(self, rss_url: str, episodes_df: pd.DataFrame) -> None:
        """Fetches one RSS feed and maps its items to the dataset episodes."""
        response = self.session.get(rss_url, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.content)

        show_category = ""
        if "tags" in feed.feed:
            show_category = feed.feed.tags[0].get("term", "")

        entry_lookup = {}
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            if title:
                entry_lookup[title] = entry

        for _, row in episodes_df.iterrows():
            episode_name = str(row["episode_name"]).strip()
            prefix_id = str(row["episode_filename_prefix"])

            output_path = self.output_dir / f"{prefix_id}.json"
            if output_path.exists():
                continue

            matched_item = entry_lookup.get(episode_name)

            enriched_data = row.to_dict()
            enriched_data["show_category"] = show_category

            if matched_item:
                enriched_data.update(self._extract_item_data(matched_item))

            self._save_enriched_data(prefix_id, enriched_data)

    def _extract_item_data(self, item: Any) -> Dict[str, str]:
        """Pulls and cleans the useful fields from a single RSS item."""
        audio_url = ""
        for link in item.get("links", []):
            if link.get("rel") == "enclosure":
                audio_url = link.get("href", "")
                break

        return {
            "rss_pub_date": item.get("published", ""),
            "rss_audio_url": audio_url,
            "rss_clean_summary": self._clean_html(item.get("summary", "")),
            "rss_duration": item.get("itunes_duration", ""),
        }

    def _save_enriched_data(self, prefix_id: str, data: Dict[str, Any]) -> None:
        """Saves the combined metadata to disk."""
        output_path = self.output_dir / f"{prefix_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    enricher = RSSEnrichmentPipeline(
        tsv_path=TSV_FILE,
        output_dir=ENRICHED_META_DIR,
    )
    enricher.process_dataset()
