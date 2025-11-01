# backend/normalizer.py
import pandas as pd
from datetime import datetime
import hashlib

def standardize_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def canonicalize_state(s):
    if pd.isna(s): return None
    s = s.strip().title()
    # You should use a state_lookup mapping to canonical names
    return s

def attach_provenance(df, dataset_id, resource_url):
    df = df.copy()
    df["_dataset_id"] = dataset_id
    df["_resource_url"] = resource_url
    # create a stable row_id
    def make_row_id(row):
        # Include more fields to ensure uniqueness
        key_fields = ["state_name", "district_name", "crop_year", "season", "crop", "area_", "production_"]
        key = "|".join([str(row.get(c,"")) for c in key_fields if c in row])
        return hashlib.sha1(key.encode("utf-8")).hexdigest()
    df["_row_id"] = df.apply(make_row_id, axis=1)
    df["_fetched_at"] = datetime.utcnow().isoformat() + "Z"
    return df

def normalize_crop_production(df, dataset_id, resource_url):
    df = standardize_cols(df)
    if "state" in df.columns:
        df["state"] = df["state"].apply(canonicalize_state)
    # convert numeric fields, handling 'NA' and other non-numeric values
    if "production" in df.columns:
        df["production_tonnes"] = pd.to_numeric(df["production"], errors="coerce")
    # Clean up any other numeric columns that might have 'NA' values
    for col in df.columns:
        if col.endswith('_tonnes') or col in ['area', 'yield', 'production_']:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = attach_provenance(df, dataset_id, resource_url)
    return df
