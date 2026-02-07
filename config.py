from pathlib import Path

# Paths
SAMPLE_JSONL = Path("data/sample/ride_log.jsonl")
PROCESSED_PARQUET = Path("data/processed/processed.parquet")
FEATURES_PARQUET = Path("data/features/features.parquet")
FEATURES_CSV = Path("data/features/features.csv")

# Feature extraction
WINDOW_SECONDS = 10  # 10-second windows for MVP
SPIKE_K = 2.5        # spike threshold within a window: mean + K*std
