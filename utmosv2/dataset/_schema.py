from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetSchema:
    file_path: Path
    dataset: str
    mos: int | None = None
