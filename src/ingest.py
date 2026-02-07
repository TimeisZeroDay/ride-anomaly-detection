from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


REQUIRED_KEYS = {"ts", "ride_id", "motor_current_amps"}


@dataclass
class IngestStats:
    total_lines: int
    parsed_rows: int
    dropped_bad_json: int
    dropped_missing_fields: int
    dropped_bad_timestamp: int
    dropped_bad_current: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "total_lines": self.total_lines,
            "parsed_rows": self.parsed_rows,
            "dropped_bad_json": self.dropped_bad_json,
            "dropped_missing_fields": self.dropped_missing_fields,
            "dropped_bad_timestamp": self.dropped_bad_timestamp,
            "dropped_bad_current": self.dropped_bad_current,
        }


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], IngestStats]:
    stats = IngestStats(
        total_lines=0,
        parsed_rows=0,
        dropped_bad_json=0,
        dropped_missing_fields=0,
        dropped_bad_timestamp=0,
        dropped_bad_current=0,
    )

    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stats.total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats.dropped_bad_json += 1
                continue

            if not isinstance(obj, dict) or not REQUIRED_KEYS.issubset(obj.keys()):
                stats.dropped_missing_fields += 1
                continue

            rows.append(obj)
            stats.parsed_rows += 1

    return rows, stats


def clean_dataframe(df: pd.DataFrame, stats: IngestStats) -> pd.DataFrame:
    # Timestamp parse
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    bad_ts = df["ts"].isna().sum()
    if bad_ts:
        stats.dropped_bad_timestamp += int(bad_ts)
        df = df.dropna(subset=["ts"])

    # Current parse + validation
    df["motor_current_amps"] = df["motor_current_amps"].apply(_safe_float)
    bad_current = df["motor_current_amps"].isna().sum()
    if bad_current:
        stats.dropped_bad_current += int(bad_current)
        df = df.dropna(subset=["motor_current_amps"])

    # Enforce non-negative amps (realistic)
    neg_current = (df["motor_current_amps"] < 0).sum()
    if neg_current:
        stats.dropped_bad_current += int(neg_current)
        df = df[df["motor_current_amps"] >= 0]

    # Normalize ride_id
    df["ride_id"] = df["ride_id"].astype(str).str.strip()
    df = df[df["ride_id"] != ""]

    # Sort + de-dup
    df = df.sort_values(["ride_id", "ts"]).drop_duplicates(subset=["ride_id", "ts"], keep="last")

    # Reset index for clean downstream work
    return df.reset_index(drop=True)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest ride motor current JSONL into clean processed dataset.")
    parser.add_argument("--input", type=str, default="data/sample/ride_log.jsonl", help="Path to input JSONL file")
    parser.add_argument("--out_parquet", type=str, default="data/processed/processed.parquet", help="Output parquet path")
    parser.add_argument("--out_csv", type=str, default="data/processed/processed.csv", help="Output csv path (fallback)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_parquet = Path(args.out_parquet)
    out_csv = Path(args.out_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows, stats = read_jsonl(input_path)
    df = pd.DataFrame(rows)

    if df.empty:
        print("No valid rows found after JSON/schema validation.")
        print("Stats:", stats.as_dict())
        return 1

    df = clean_dataframe(df, stats)

    print("Ingest complete.")
    print("Stats:", stats.as_dict())
    print(f"Rows after cleaning: {len(df)}")
    print(f"Time range: {df['ts'].min()} → {df['ts'].max()}")
    print(f"Ride IDs: {sorted(df['ride_id'].unique().tolist())}")

    ensure_parent_dir(out_csv)
    df.to_csv(out_csv, index=False)
    print(f"Wrote CSV: {out_csv}")

    # Parquet is preferred; if pyarrow isn't installed, this will error
    try:
        ensure_parent_dir(out_parquet)
        df.to_parquet(out_parquet, index=False)
        print(f"Wrote Parquet: {out_parquet}")
    except Exception as e:
        print("Parquet write failed (is pyarrow installed?). CSV was still written.")
        print("Parquet error:", repr(e))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
