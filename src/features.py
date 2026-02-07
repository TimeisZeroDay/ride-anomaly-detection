from __future__ import annotations

# --- path setup (must come AFTER __future__) ---
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- standard imports ---
import argparse
import numpy as np
import pandas as pd

import config


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_window_features(
    df: pd.DataFrame,
    window_seconds: int,
    spike_k: float,
) -> pd.DataFrame:
    """
    Compute windowed features from motor_current_amps.

    Expected columns:
      - ts (datetime64[ns, UTC])
      - ride_id (str)
      - motor_current_amps (float)
    """
    df = df.copy()
    df = df.sort_values(["ride_id", "ts"])

    feature_rows = []

    for ride_id, g in df.groupby("ride_id"):
        g = g.set_index("ts")

        # Resample into fixed windows
        windows = g["motor_current_amps"].resample(f"{window_seconds}s")

        mean_ = windows.mean()
        max_ = windows.max()
        std_ = windows.std(ddof=0).fillna(0.0)
        count_ = windows.count()

        slope_vals = []
        spike_counts = []

        for win_start, series in windows:
            s = series.dropna()

            # Not enough points → no slope, no spikes
            if len(s) < 2:
                slope_vals.append(0.0)
                spike_counts.append(0)
                continue

            # Time (seconds) relative to window start
            t = (s.index - s.index[0]).total_seconds().astype(float)
            y = s.values.astype(float)

            # Linear slope (amps / second)
            slope = float(np.polyfit(t, y, 1)[0])
            slope_vals.append(slope)

            mu = float(y.mean())
            sigma = float(y.std(ddof=0))

            if sigma > 0:
                thresh = mu + spike_k * sigma
                spikes = int((y > thresh).sum())
            else:
                spikes = int((y > mu).sum())

            spike_counts.append(spikes)

        feat = pd.DataFrame(
            {
                "ride_id": ride_id,
                "window_start": mean_.index,
                "window_end": mean_.index + pd.Timedelta(seconds=window_seconds),
                "mean_current": mean_.values,
                "max_current": max_.values,
                "std_current": std_.values,
                "slope_current": slope_vals,
                "spike_count": spike_counts,
                "n_points": count_.values,
            }
        )

        # Drop empty windows
        feat = feat[feat["n_points"] > 0].reset_index(drop=True)
        feature_rows.append(feat)

    if not feature_rows:
        return pd.DataFrame()

    return pd.concat(feature_rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute windowed features from processed ride telemetry."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.PROCESSED_PARQUET),
        help="Processed parquet path",
    )
    parser.add_argument(
        "--out_parquet",
        type=str,
        default=str(config.FEATURES_PARQUET),
        help="Features parquet path",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=str(config.FEATURES_CSV),
        help="Features CSV path",
    )
    parser.add_argument(
        "--window_seconds",
        type=int,
        default=config.WINDOW_SECONDS,
        help="Window size in seconds",
    )
    parser.add_argument(
        "--spike_k",
        type=float,
        default=config.SPIKE_K,
        help="Spike threshold multiplier",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Processed input not found: {in_path}")

    df = pd.read_parquet(in_path)

    required_cols = {"ts", "ride_id", "motor_current_amps"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feats = compute_window_features(
        df,
        window_seconds=args.window_seconds,
        spike_k=args.spike_k,
    )

    if feats.empty:
        print("No features produced.")
        return 1

    ensure_parent_dir(Path(args.out_csv))
    feats.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv}")

    try:
        ensure_parent_dir(Path(args.out_parquet))
        feats.to_parquet(args.out_parquet, index=False)
        print(f"Wrote Parquet: {args.out_parquet}")
    except Exception as e:
        print("Parquet write failed; CSV still written.")
        print("Error:", repr(e))

    print("\nFeature extraction complete.")
    print(f"Rows: {len(feats)}")
    print("Columns:", list(feats.columns))
    print(
        "Windows:",
        feats["window_start"].min(),
        "→",
        feats["window_start"].max(),
    )

    print("\nSample rows:")
    print(feats.head(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
