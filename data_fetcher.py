# backend/data_fetcher.py
"""
Enhanced Data Fetcher for data.gov.in
Expands RAG knowledge base with comprehensive agricultural and related datasets
"""

import requests
import json
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv

load_dotenv()

class DataGovFetcher:
    """Enhanced fetcher for data.gov.in with comprehensive dataset support"""
    
    def __init__(self):
        self.api_key = os.getenv("DATAGOV_API_KEY")
        self.base_url = "https://api.data.gov.in/resource"
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed")
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset catalog
        self.datasets = self._load_dataset_catalog()
    
    def _load_dataset_catalog(self) -> Dict:
        """Load dataset catalog from datasets.json"""
        try:
            with open("datasets.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ datasets.json not found")
            return {}
    
    def fetch_dataset(self, dataset_key: str, limit: int = 1000, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch a dataset from data.gov.in
        
        Args:
            dataset_key: Key from datasets.json
            limit: Maximum records to fetch per request
            force_refresh: Force re-download even if file exists
        """
        if dataset_key not in self.datasets:
            print(f"❌ Dataset '{dataset_key}' not found in catalog")
            return None
        
        dataset_info = self.datasets[dataset_key]
        resource_id = dataset_info["resource_id"]
        
        # Check if already downloaded
        raw_file = self.raw_data_dir / f"{resource_id}.json"
        if raw_file.exists() and not force_refresh:
            print(f"📁 Loading existing data for {dataset_key}")
            return self._load_existing_data(raw_file)
        
        print(f"🔄 Fetching {dataset_key} from data.gov.in...")
        
        try:
            # Fetch data with pagination
            all_records = []
            offset = 0
            
            while True:
                params = {
                    "api-key": self.api_key,
                    "format": "json",
                    "limit": limit,
                    "offset": offset
                }
                
                response = requests.get(
                    f"{self.base_url}/{resource_id}",
                    params=params,
                    timeout=60
                )
                response.raise_for_status()
                
                data = response.json()
                records = data.get("records", [])
                
                if not records:
                    break
                
                all_records.extend(records)
                offset += limit
                
                print(f"  📊 Fetched {len(all_records)} records (offset {offset})")
                
                # Rate limiting
                time.sleep(0.2)
                
                # Safety limit to prevent infinite loops
                if len(all_records) > 50000:
                    print(f"  ⚠️ Reached safety limit of 50k records")
                    break
            
            # Save raw data
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved {len(all_records)} records for {dataset_key}")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_records)
            return df
            
        except requests.RequestException as e:
            print(f"❌ Network error fetching {dataset_key}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error fetching {dataset_key}: {e}")
            return None
    
    def _load_existing_data(self, file_path: Path) -> pd.DataFrame:
        """Load existing JSON data file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def process_and_normalize_dataset(self, dataset_key: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process and normalize dataset based on its category
        """
        if df.empty:
            return None
        
        dataset_info = self.datasets[dataset_key]
        category = dataset_info.get("category", "general")
        
        print(f"🔧 Processing {dataset_key} (category: {category})")
        
        # Standardize column names
        df.columns = [col.strip().lower().replace(" ", "_").replace("-", "_") 
                     for col in df.columns]
        
        # Category-specific processing
        if category == "agriculture":
            df = self._process_agriculture_data(df, dataset_key)
        elif category == "climate":
            df = self._process_climate_data(df, dataset_key)
        elif category == "soil":
            df = self._process_soil_data(df, dataset_key)
        elif category == "irrigation":
            df = self._process_irrigation_data(df, dataset_key)
        elif category == "market":
            df = self._process_market_data(df, dataset_key)
        else:
            df = self._process_general_data(df, dataset_key)
        
        # Add metadata
        df["_dataset_key"] = dataset_key
        df["_category"] = category
        df["_processed_at"] = datetime.now().isoformat()
        
        # Save processed data
        processed_file = self.processed_data_dir / f"{dataset_key}.parquet"
        df.to_parquet(processed_file, index=False)
        print(f"💾 Saved processed data to {processed_file}")
        
        return df
    
    def _process_agriculture_data(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """Process agricultural datasets"""
        # Standardize state names
        if "state_name" in df.columns:
            df["state_name"] = df["state_name"].str.title().str.strip()
        
        # Standardize district names
        if "district_name" in df.columns:
            df["district_name"] = df["district_name"].str.title().str.strip()
        
        # Convert numeric columns
        numeric_cols = ["production", "area", "yield", "productivity"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def _process_climate_data(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """Process climate datasets"""
        # Convert numeric weather columns
        weather_cols = ["rainfall", "temperature", "humidity", "wind_speed"]
        for col in weather_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Standardize date columns
        date_cols = ["date", "year", "month"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        return df
    
    def _process_soil_data(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """Process soil health datasets"""
        # Convert soil parameter columns
        soil_cols = ["ph", "organic_carbon", "nitrogen", "phosphorus", "potassium"]
        for col in soil_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def _process_irrigation_data(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """Process irrigation datasets"""
        # Convert irrigation metrics
        irrigation_cols = ["irrigated_area", "water_consumption", "efficiency"]
        for col in irrigation_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def _process_market_data(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """Process market price datasets"""
        # Convert price columns
        price_cols = ["price", "min_price", "max_price", "modal_price"]
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Standardize commodity names
        if "commodity" in df.columns:
            df["commodity"] = df["commodity"].str.title().str.strip()
        
        return df
    
    def _process_general_data(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """Process general datasets"""
        # Basic numeric conversion for common columns
        numeric_cols = ["value", "amount", "quantity", "count"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def fetch_all_datasets(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch all datasets from the catalog
        
        Args:
            force_refresh: Force re-download of all datasets
        """
        results = {}
        
        print(f"🚀 Starting bulk fetch of {len(self.datasets)} datasets...")
        
        for i, dataset_key in enumerate(self.datasets.keys(), 1):
            print(f"\n📊 [{i}/{len(self.datasets)}] Processing {dataset_key}...")
            
            try:
                # Fetch raw data
                df = self.fetch_dataset(dataset_key, force_refresh=force_refresh)
                
                if df is not None and not df.empty:
                    # Process and normalize
                    processed_df = self.process_and_normalize_dataset(dataset_key, df)
                    
                    if processed_df is not None:
                        results[dataset_key] = processed_df
                        print(f"✅ Successfully processed {dataset_key}: {len(processed_df)} records")
                    else:
                        print(f"⚠️ Failed to process {dataset_key}")
                else:
                    print(f"⚠️ No data retrieved for {dataset_key}")
                
            except Exception as e:
                print(f"❌ Error processing {dataset_key}: {e}")
                continue
            
            # Rate limiting between datasets
            time.sleep(1)
        
        print(f"\n🎉 Bulk fetch completed! Successfully processed {len(results)} datasets")
        return results
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary of available and processed datasets"""
        summary = {
            "total_datasets": len(self.datasets),
            "categories": {},
            "processed_files": [],
            "raw_files": []
        }
        
        # Count by category
        for dataset_key, info in self.datasets.items():
            category = info.get("category", "general")
            summary["categories"][category] = summary["categories"].get(category, 0) + 1
        
        # Check processed files
        for file_path in self.processed_data_dir.glob("*.parquet"):
            summary["processed_files"].append(file_path.name)
        
        # Check raw files
        for file_path in self.raw_data_dir.glob("*.json"):
            summary["raw_files"].append(file_path.name)
        
        return summary

# Convenience functions
def fetch_single_dataset(dataset_key: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Fetch a single dataset"""
    fetcher = DataGovFetcher()
    df = fetcher.fetch_dataset(dataset_key, force_refresh=force_refresh)
    if df is not None:
        return fetcher.process_and_normalize_dataset(dataset_key, df)
    return None

def fetch_all_data(force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch all datasets from catalog"""
    fetcher = DataGovFetcher()
    return fetcher.fetch_all_datasets(force_refresh=force_refresh)

def get_data_summary() -> Dict[str, Any]:
    """Get summary of data status"""
    fetcher = DataGovFetcher()
    return fetcher.get_dataset_summary()