from __future__ import annotations

from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import argparse
import pandas as pd


def assign_severity(row: pd.Series) -> str:
    """
    Severity based on both sustained load (mean) and extreme peaks (max).
    """
    max_a = float(row["max_current"])
    mean_a = float(row["mean_current"])

    # Extreme transient spike
    if max_a >= 30:
        return "CRITICAL"

    # Sustained high load is a bigger operational risk than a single moderate max
    if mean_a >= 24:
        return "HIGH"

    # Elevated load or notable transient
    if mean_a >= 20 or max_a >= 26:
        return "MEDIUM"

    return "LOW"


def suggested_checks(row: pd.Series) -> list[str]:
    """
    Recommend operator actions using anomaly context, not just string matching.
    """
    reasons = row["reasons"]
    max_a = float(row["max_current"])
    mean_a = float(row["mean_current"])
    std_a = float(row["std_current"])

    checks: list[str] = []

    # Sustained high load pattern
    if mean_a >= 24:
        checks.extend(
            [
                "Inspect mechanical load (binding/friction) and drivetrain alignment.",
                "Verify lubrication schedule and check for abnormal wear on moving components.",
                "Review recent maintenance changes that could increase resistance (belt tension, alignment).",
            ]
        )

    # Spike/transient pattern
    if max_a >= 30 or ("max_current" in " ".join(reasons) and std_a >= 1.0):
        checks.extend(
            [
                "Inspect for intermittent obstructions or momentary binding events.",
                "Check motor drive / power electronics for transient faults or current limiting behavior.",
                "Correlate this timestamp with ride stop/start events or operator logs (if available).",
            ]
        )

    # Variability/noise pattern
    if std_a >= 3.0:
        checks.extend(
            [
                "Check sensor integrity and wiring for noise or intermittent connection.",
                "Confirm sampling consistency (missing data can inflate variability in windows).",
            ]
        )

    # Always include a conservative operational action
    checks.append("Monitor motor temperature and current trend closely over the next 10–15 minutes.")

    # De-dup while preserving order
    deduped: list[str] = []
    seen = set()
    for c in checks:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    return deduped


def build_alerts(events: pd.DataFrame) -> pd.DataFrame:
    alerts = []

    for _, row in events.iterrows():
        severity = assign_severity(row)

        alert = {
            "alert_id": f"{row['ride_id']}-{row['window_start'].isoformat()}",
            "ride_id": row["ride_id"],
            "time": row["window_start"],
            "severity": severity,
            "why_flagged": row["reasons"],
            "suggested_next_check": suggested_checks(row),
            "mean_current": round(float(row["mean_current"]), 2),
            "max_current": round(float(row["max_current"]), 2),
            "std_current": round(float(row["std_current"]), 3),
            "score": row["score"],
        }

        alerts.append(alert)

    return pd.DataFrame(alerts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert anomaly events into actionable alerts.")
    parser.add_argument(
        "--input",
        type=str,
        default="alerts/anomaly_events.csv",
        help="Input anomaly events CSV",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="alerts/alerts.json",
        help="Output alerts JSON",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="alerts/alerts.csv",
        help="Output alerts CSV",
    )
    args = parser.parse_args()

    events_path = Path(args.input)
    if not events_path.exists():
        raise FileNotFoundError(f"Anomaly events file not found: {events_path}")

    events = pd.read_csv(events_path, parse_dates=["window_start", "window_end"])

    # Ensure reasons is a list[str] even when reading from CSV
    # CSV stores lists like "['a', 'b']" so we convert safely.
    def _parse_reasons(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            # naive safe parse for this controlled MVP format
            # strip brackets and split on "','" patterns
            inner = x[1:-1].strip()
            if not inner:
                return []
            # Remove surrounding quotes if present and split
            parts = []
            cur = inner
            # simplest robust approach for this MVP: split on "',"
            for p in cur.split("',"):
                p = p.strip()
                p = p.strip("'").strip('"')
                if p:
                    parts.append(p)
            return parts
        return [str(x)]

    events["reasons"] = events["reasons"].apply(_parse_reasons)

    alerts_df = build_alerts(events)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    alerts_df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV alerts: {args.out_csv}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(alerts_df.to_dict(orient="records"), f, indent=2, default=str)

    print(f"Wrote JSON alerts: {args.out_json}")
    print("\nFinal alerts:")
    print(alerts_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
