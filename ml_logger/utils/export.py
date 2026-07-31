"""Export logger JSONL streams to flat CSV files."""

import json
import csv
from pathlib import Path
from ..runtime import get_logger

logger = get_logger(__name__)

def jsonl_to_csv(jsonl_path: Path, csv_path: Path):
    """Converts a JSONL file to a CSV file."""
    if not jsonl_path.exists():
        logger.warning("%s does not exist", jsonl_path)
        return

    data = []
    keys = set()
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                data.append(row)
                keys.update(row.keys())
            except json.JSONDecodeError:
                pass

    if not data:
        logger.warning("No valid data found in %s", jsonl_path)
        return

    # Ensure consistent column ordering (timestamp first if exists)
    fieldnames = list(keys)
    if "timestamp" in fieldnames:
        fieldnames.remove("timestamp")
        fieldnames.insert(0, "timestamp")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        
    logger.info("Exported %d rows to %s", len(data), csv_path)

def export_run_to_csv(save_dir: str):
    """Finds telemetry and metrics JSONL files and exports them to CSV."""
    save_path = Path(save_dir)
    
    metrics_jsonl = save_path / "metrics.jsonl"
    metrics_csv = save_path / "metrics.csv"
    if metrics_jsonl.exists():
        jsonl_to_csv(metrics_jsonl, metrics_csv)
        
    telemetry_jsonl = save_path / "telemetry.jsonl"
    telemetry_csv = save_path / "telemetry.csv"
    if telemetry_jsonl.exists():
        jsonl_to_csv(telemetry_jsonl, telemetry_csv)
