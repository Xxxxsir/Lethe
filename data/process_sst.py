# process_sst.py
import pandas as pd
import argparse
import json


def convert_parquet_to_json(parquet_path, json_path):
    df = pd.read_parquet(parquet_path)

    if "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text"})
    
    if "idx" in df.columns:
        df = df.drop(columns=["idx"])

    records = df.to_dict(orient="records")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(f"✅ Finish : {parquet_path} → {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert SST-2 parquet files to JSON format.")
    parser.add_argument("--parquet", type=str, required=True, help="parquet files path")
    parser.add_argument("--json", type=str, required=True, help="output path")
    args = parser.parse_args()

    convert_parquet_to_json(args.parquet, args.json)


if __name__ == "__main__":
    main()
