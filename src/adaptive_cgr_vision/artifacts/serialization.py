"""Run artifact serialization and provenance saver."""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import json
import pandas as pd

from adaptive_cgr_vision.artifacts.schema import RunMetadata


def save_run_artifacts(
    run_dir: Union[str, Path],
    metadata: RunMetadata,
    metrics: Dict[str, Any],
    predictions_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Save all experiment run artifacts to a dedicated run directory.

    Saved files:
    - metadata.json (provenance, seeds, config)
    - metrics.json (macro-f1, accuracy, etc.)
    - predictions.parquet (optional per-sample predictions)
    """
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Save metadata.json
    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # Save metrics.json
    with open(path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions.parquet
    if predictions_df is not None:
        predictions_df.to_parquet(path / "predictions.parquet", index=False)

    return path
