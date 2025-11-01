# backend/adapters.py
import requests, os, time, json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DATAGOV_API_KEY")

def fetch_resource_paginated(resource_id, out_dir="data/raw", limit=100):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base = f"https://api.data.gov.in/resource/{resource_id}"
    offset = 0
    all_records = []
    while True:
        params = {"api-key": API_KEY, "format": "json", "limit": limit, "offset": offset}
        r = requests.get(base, params=params, timeout=60)
        r.raise_for_status()
        resp = r.json()
        records = resp.get("records", [])
        if not records:
            break
        all_records.extend(records)
        offset += limit
        print(f"Fetched {len(all_records)} records (offset {offset})")
        time.sleep(0.2)
    # Save raw JSON
    out_path = Path(out_dir) / f"{resource_id}.json"
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(all_records, f, ensure_ascii=False)
    # Convert to DataFrame for convenience
    df = pd.DataFrame(all_records)
    return df
