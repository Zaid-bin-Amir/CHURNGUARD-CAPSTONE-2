import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from churnguard import config, data, train


def combine(new_data_dir):
    base = data.load_raw()
    frames = [base]
    files = sorted(Path(new_data_dir).glob("*.csv"))
    for path in files:
        extra = pd.read_csv(path)
        missing = [column for column in base.columns if column not in extra.columns]
        if missing:
            raise SystemExit(f"{path.name} is missing columns: {', '.join(missing)}")
        frames.append(extra[base.columns])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=[config.ID_COLUMN], keep="last").reset_index(drop=True)
    return combined, len(base), files


def main():
    parser = argparse.ArgumentParser(description="Retrain ChurnGuard on the base dataset plus newly appended CSV files.")
    parser.add_argument("--new-data-dir", default=str(config.NEW_DATA_DIR), help="Folder with new CSV files in the original schema")
    parser.add_argument("--search-iter", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    combined, base_rows, files = combine(arguments.new_data_dir)
    print(f"Base rows: {base_rows}. New files: {len(files)}. Combined unique customers: {len(combined)}.")
    tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_retrain"
    results = train.run(raw_df=combined, tag=tag, search_iter=arguments.search_iter, track=not arguments.no_mlflow)
    print(f"New model version: {results['tag']}")
    print(f"Hold-out recall {results['holdout']['recall']:.3f} | PR-AUC {results['holdout']['pr_auc']:.3f} | ROC-AUC {results['holdout']['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
