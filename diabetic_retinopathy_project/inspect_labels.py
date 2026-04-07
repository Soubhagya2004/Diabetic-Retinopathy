import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "dataset" / "train.csv"
TRAIN_DIR = ROOT / "dataset" / "train_images"

if not CSV_PATH.exists():
    print("CSV not found:", CSV_PATH)
    raise SystemExit(1)

df = pd.read_csv(CSV_PATH)
print(f"Total rows in CSV: {len(df)}")
print("Label distribution:")
print(df["diagnosis"].value_counts().sort_index())

print("\nSample files per class:")
for cls in sorted(df["diagnosis"].unique()):
    sample = df[df["diagnosis"] == cls].head(5)
    print(f"Class {cls} (count={len(df[df['diagnosis']==cls])}):")
    for _, row in sample.iterrows():
        print(" ", row["id"]) 
    print()
