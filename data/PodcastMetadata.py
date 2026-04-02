from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import ast


## DEPRICATED: Metadata, full episode transcript, full segment should be indexed. Not loaded to memory


class PodcastMetadata(BaseModel):
    show_name: str
    show_description: str
    publisher: str
    language: List[str]
    rss_link: str
    episode_name: str
    episode_description: str
    duration: float
    show_filename_prefix: str
    episode_filename_prefix: str


# Interface
class MetadataRepository(ABC):
    @abstractmethod
    def get_metadata_by_file_id(self, file_id: str) -> PodcastMetadata:
        pass


class TSVMetadataRepo(MetadataRepository):
    def __init__(self, tsv_path: str):
        self.metadata_df = pd.read_csv(tsv_path, sep="\t")
        self.metadata_df.fillna("", inplace=True)
        self.metadata_df.set_index("episode_filename_prefix", inplace=True)

    def get_metadata_by_file_id(self, file_id: str) -> PodcastMetadata:
        filename_prefix = file_id.removesuffix(".json")
        try:
            row = self.metadata_df.loc[filename_prefix]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            return PodcastMetadata(
                show_name=str(row.get("show_name", "")),
                show_description=str(row.get("show_description", "")),
                publisher=str(row.get("publisher", "")),
                language=ast.literal_eval(row.get("language", "[]")),
                rss_link=str(row.get("rss_link", "")),
                episode_name=str(row.get("episode_name", "")),
                episode_description=str(row.get("episode_description", "")),
                duration=float(row.get("duration", 0)),
                show_filename_prefix=str(row.get("show_filename_prefix", "")),
                episode_filename_prefix=str(row.get("episode_filename_prefix", "")),
            )
        except KeyError:
            raise ValueError(f"Metadata not found for episode: {file_id}")
