import pandas as pd
import json
import glob
import os
import numpy as np
from tqdm import tqdm

def to_serializable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val

def convert(input_dir, output_path, max_samples=None):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    print(f"Found {len(files)} parquet files in {input_dir}")

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for file in tqdm(files):
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                record = {k: to_serializable(row[k]) for k in df.columns}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if max_samples and count >= max_samples:
                    print(f"Reached max_samples={max_samples}, stopping.")
                    print(f"Saved {count} records to {output_path}")
                    return
    print(f"Saved {count} records to {output_path}")

# stage1: 23 shards, ~14M rows total
convert(
    input_dir="../data/stage1",
    output_path="../data/stage1/stage1_train_data.jsonl",
)

# stage2: 7 shards, ~868K rows total
convert(
    input_dir="../data/stage2",
    output_path="../data/stage2/stage2_train_data.jsonl",
)
