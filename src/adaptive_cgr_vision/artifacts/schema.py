"""Run metadata schema and provenance tracking."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import datetime
import sys


@dataclass
class RunMetadata:
    """Run provenance metadata."""
    run_id: str
    experiment_name: str
    timestamp_utc: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    python_version: str = field(default_factory=lambda: sys.version)
    random_seed: int = 42
    dataset_manifest_checksum: str = ""
    config_dict: Dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    fusion_method: str = ""
    cv_folds: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
