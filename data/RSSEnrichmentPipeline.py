import pandas as pd
import feedparser
import requests
import json
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

## Enriches the metadata file with RSS links.
# This is an example  "show_category": "Technology",
#    "rss_pub_date": "Tue, 07 Jan 2020 11:45:45 GMT",
#   "rss_audio_url": "https://anchor.fm/s/124dc120/podcast/play/9537692/https%3A%2F%2Fd3ctxlq1ktw2nl.cloudfront.net%2Fstaging%2F2020-03-05%2Fe9e9fe41824bf80a23010759484918e9.m4a",
#   "rss_clean_summary": "A brief overview of WebXR in general and WebAR and WebVR projects that we have worked on in the past.",
#   "rss_duration": "00:04:50"


class RSSEnrichmentPipeline:
    """
    Downloads RSS feeds per show, extracts episode-level data,
    cleans HTML, and saves enriched metadata locally.
    """

    def __init__(self, tsv_path: str, output_dir: str):
        self.tsv_path = Path(tsv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # We use a session to pool connections, making requests much faster
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
        """Reads the TSV, groups by show, and processes each feed once."""
        logging.info("Loading TSV for grouping...")
        df = pd.read_csv(self.tsv_path, sep="\t")
        df.fillna("", inplace=True)

        # Group all episodes by their RSS link
        grouped_shows = df.groupby("rss_link")

        for rss_url, episodes_df in grouped_shows:
            if not rss_url or not str(rss_url).startswith("http"):
                continue

            logging.info(f"Fetching feed: {rss_url} ({len(episodes_df)} episodes)")
            self._process_single_show(rss_url, episodes_df)

    def _process_single_show(self, rss_url: str, episodes_df: pd.DataFrame) -> None:
        """Fetches one RSS feed and maps its items to the dataset episodes."""
        try:
            response = self.session.get(rss_url, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            # Extract Show-level info (e.g., category)
            show_category = ""
            if "tags" in feed.feed:
                show_category = feed.feed.tags[0].get("term", "")

            for _, row in episodes_df.iterrows():
                episode_name = str(row["episode_name"])
                prefix_id = str(row["episode_filename_prefix"])

                # Find the corresponding item in the RSS feed
                matched_item = self._find_item_by_title(feed.entries, episode_name)

                # Combine original TSV data with the newly scraped RSS data
                enriched_data = row.to_dict()
                enriched_data["show_category"] = show_category

                if matched_item:
                    enriched_data.update(self._extract_item_data(matched_item))
                else:
                    logging.warning(f"  -> Could not find RSS item for: {episode_name}")

                self._save_enriched_data(prefix_id, enriched_data)

        except Exception as e:
            logging.error(f"Failed to process {rss_url}: {e}")

    def _find_item_by_title(self, entries: list, target_title: str) -> Optional[Dict]:
        """Searches the RSS items for an exact title match."""
        for entry in entries:
            if entry.title.strip() == target_title.strip():
                return entry
        return None

    def _extract_item_data(self, item: Any) -> Dict[str, str]:
        """Pulls and cleans the useful fields from a single RSS item."""

        # Find the audio URL from the enclosures list
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


# --- Usage ---
if __name__ == "__main__":
    enricher = RSSEnrichmentPipeline(
        #tsv_path="./podcasts-no-audio-13GB/metadata.tsv",
        tsv_path=r".\podcasts-no-audio-13GB\metadata\spotify-podcasts-2020\metadata.tsv",
        output_dir="./enriched_metadata",
    )
    enricher.process_dataset() # Uncomment to run