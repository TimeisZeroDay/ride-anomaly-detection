from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import argparse
import pandas as pd
import numpy as np

import config


def detect_anomalies(
    feats: pd.DataFrame,
    baseline_windows: int = 2,
    spike_sigma: float = 4.0,
    sustain_sigma: float = 2.5,
    sustain_windows: int = 2,
):
    """
    Detect anomalies using rolling statistical baselines.
    Returns a list of anomaly event dicts.
    """
    feats = feats.sort_values(["ride_id", "window_start"]).reset_index(drop=True)

    events = []

    for ride_id, g in feats.groupby("ride_id"):
        g = g.reset_index(drop=True)

        for i in range(len(g)):
            # Need enough history to build baseline
            if i < baseline_windows:
                continue

            baseline = g.iloc[i - baseline_windows : i]

            base_mean = baseline["mean_current"].mean()
            base_std = baseline["mean_current"].std(ddof=0)
            base_std = max(base_std, 0.01)  # avoid divide-by-zero

            row = g.iloc[i]
            reasons = []
            score = 0.0

            # --- Spike detection ---
            if row["max_current"] > base_mean + spike_sigma * base_std:
                reasons.append(
                    f"max_current {row['max_current']:.1f}A > {spike_sigma:.1f}σ above baseline"
                )
                score += 1.0

            # --- Sustained high load ---
            if row["mean_current"] > base_mean + sustain_sigma * base_std:
                recent = g.iloc[max(0, i - sustain_windows + 1) : i + 1]
                if all(
                    recent["mean_current"]
                    > base_mean + sustain_sigma * base_std
                ):
                    reasons.append(
                        f"sustained mean_current > {sustain_sigma:.1f}σ for {len(recent)} windows"
                    )
                    score += 1.0

            # --- Variability jump ---
            if row["std_current"] > baseline["std_current"].mean() * 3:
                reasons.append("sudden increase in current variability")
                score += 0.5

            if reasons:
                events.append(
                    {
                        "ride_id": ride_id,
                        "window_start": row["window_start"],
                        "window_end": row["window_end"],
                        "mean_current": row["mean_current"],
                        "max_current": row["max_current"],
                        "std_current": row["std_current"],
                        "reasons": reasons,
                        "score": round(score, 2),
                    }
                )

    return pd.DataFrame(events)


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline anomaly detection.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.FEATURES_PARQUET),
        help="Features parquet path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="alerts/anomaly_events.csv",
        help="Output anomaly events CSV",
    )
    args = parser.parse_args()

    feats = pd.read_parquet(args.input)

    events = detect_anomalies(feats)

    if events.empty:
        print("No anomalies detected.")
        return 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out, index=False)

    print("Anomaly detection complete.")
    print(f"Events detected: {len(events)}")
    print("\nDetected events:")
    print(events.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
