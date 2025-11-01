# backend/enhanced_rag.py
"""
Enhanced RAG System with Dynamic Data Loading
Supports multiple data sources from data.gov.in with intelligent document processing
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime

# Use existing dependencies
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class EnhancedRAG:
    """
    Enhanced RAG system with dynamic data loading and intelligent document processing
    """
    
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.llm = genai.GenerativeModel("gemini-2.5-flash")
        
        # Document storage
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # Dataset tracking
        self.loaded_datasets = {}
        self.dataset_stats = {}
        
        # Load datasets catalog
        self.datasets_catalog = self._load_datasets_catalog()
        
        # Load all available data
        self._load_all_data()
    
    def _load_datasets_catalog(self) -> Dict:
        """Load datasets catalog"""
        try:
            with open("datasets.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ datasets.json not found")
            return {}
    
    def _load_all_data(self):
        """Load all available processed datasets"""
        print("🔄 Loading enhanced RAG system with all available datasets...")
        
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            print("❌ No processed data directory found")
            return
        
        # Load all parquet files
        for parquet_file in processed_dir.glob("*.parquet"):
            dataset_key = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                self._process_dataset_for_rag(dataset_key, df)
                print(f"✅ Loaded {dataset_key}: {len(df)} records")
            except Exception as e:
                print(f"❌ Error loading {dataset_key}: {e}")
        
        # Create embeddings if we have documents
        if self.documents:
            print(f"🔧 Creating embeddings for {len(self.documents)} documents...")
            self.embeddings = self.embedding_model.encode(self.documents)
            print("✅ Enhanced RAG system initialized successfully")
        else:
            print("⚠️ No documents loaded")
    
    def _process_dataset_for_rag(self, dataset_key: str, df: pd.DataFrame):
        """Process a dataset into RAG documents"""
        if df.empty:
            return
        
        # Get dataset info
        dataset_info = self.datasets_catalog.get(dataset_key, {})
        category = dataset_info.get("category", "general")
        
        # Sample data to manage memory (adjust sampling based on dataset size)
        sample_size = min(len(df), 2000)  # Max 2000 records per dataset
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.copy()
        
        # Process based on category
        if category == "agriculture":
            self._process_agriculture_documents(dataset_key, df_sample, dataset_info)
        elif category == "climate":
            self._process_climate_documents(dataset_key, df_sample, dataset_info)
        elif category == "soil":
            self._process_soil_documents(dataset_key, df_sample, dataset_info)
        elif category == "irrigation":
            self._process_irrigation_documents(dataset_key, df_sample, dataset_info)
        elif category == "market":
            self._process_market_documents(dataset_key, df_sample, dataset_info)
        elif category == "water":
            self._process_water_documents(dataset_key, df_sample, dataset_info)
        elif category == "machinery":
            self._process_machinery_documents(dataset_key, df_sample, dataset_info)
        elif category == "insurance":
            self._process_insurance_documents(dataset_key, df_sample, dataset_info)
        else:
            self._process_general_documents(dataset_key, df_sample, dataset_info)
        
        # Track loaded dataset
        self.loaded_datasets[dataset_key] = len(df_sample)
        self.dataset_stats[dataset_key] = {
            "category": category,
            "total_records": len(df),
            "processed_records": len(df_sample),
            "title": dataset_info.get("title", dataset_key)
        }
    
    def _process_agriculture_documents(self, dataset_key: str, df: pd.DataFrame, dataset_info: Dict):
        """Process agricultural datasets into documents"""
        for _, row in df.iterrows():
            parts = []
            
            # Core agricultural information
            if pd.notna(row.get('state_name')):
                parts.append(f"State: {row['state_name']}")
            if pd.notna(row.get('district_name')):
                parts.append(f"District: {row['district_name']}")
            if pd.notna(row.get('crop')):
                parts.append(f"Crop: {row['crop']}")
            if pd.notna(row.get('crop_year')):
                parts.append(f"Year: {row['crop_year']}")
            if pd.notna(row.get('season')):
                parts.append(f"Season: {row['season']}")
            if pd.notna(row.get('production')):
                parts.append(f"Production: {row['production']} tonnes")
            if pd.notna(row.get('area')):
                parts.append(f"Area: {row['area']} hectares")
            if pd.notna(row.get('yield')):
                parts.append(f"Yield: {row['yield']} kg/hectare")
            
            # Additional fields for other agricultural datasets
            if pd.notna(row.get('commodity')):
                parts.append(f"Commodity: {row['commodity']}")
            if pd.notna(row.get('variety')):
                parts.append(f"Variety: {row['variety']}")
            
            if parts:
                doc_text = " | ".join(parts)
                
                metadata = {
                    "source": dataset_key,
                    "category": "agriculture",
                    "state_name": str(row.get('state_name', '')),
                    "district_name": str(row.get('district_name', '')),
                    "crop": str(row.get('crop', '')),
                    "year": str(row.get('crop_year', row.get('year', ''))),
                    "dataset_title": dataset_info.get("title", dataset_key)
                }
                
                self.documents.append(doc_text)
                self.metadata.append(metadata)
    
    def _process_climate_documents(self, dataset_key: str, df: pd.DataFrame, dataset_info: Dict):
        """Process climate datasets into documents"""
        for _, row in df.iterrows():
            parts = []
            
            if pd.notna(row.get('sd_name')):
                parts.append(f"Region: {row['sd_name']}")
            if pd.notna(row.get('state_name')):
                parts.append(f"State: {row['state_name']}")
            if pd.notna(row.get('year')):
                parts.append(f"Year: {row['year']}")
            if pd.notna(row.get('month')):
                parts.append(f"Month: {row['month']}")
            if pd.notna(row.get('annual')):
                parts.append(f"Annual Rainfall: {row['annual']} mm")
            if pd.notna(row.get('rainfall')):
                parts.append(f"Rainfall: {row['rainfall']} mm")
            if pd.notna(row.get('temperature')):
                parts.append(f"Temperature: {row['temperature']}°C")
            if pd.notna(row.get('humidity')):
                parts.append(f"Humidity: {row['humidity']}%")
            
            if parts:
                doc_text = " | ".join(parts)
                
                metadata = {
                    "source": dataset_key,
                    "category": "climate",
                    "region": str(row.get('sd_name', row.get('region', ''))),
                    "state_name": str(row.get('state_name', '')),
                    "year": str(row.get('year', '')),
                    "dataset_title": dataset_info.get("title", dataset_key)
                }
                
                self.documents.append(doc_text)
                self.metadata.append(metadata)
    
    def _process_soil_documents(self, dataset_key: str, df: pd.DataFrame, dataset_info: Dict):
        """Process soil health datasets into documents"""
        for _, row in df.iterrows():
            parts = []
            
            if pd.notna(row.get('state_name')):
                parts.append(f"State: {row['state_name']}")
            if pd.notna(row.get('district_name')):
                parts.append(f"District: {row['district_name']}")
            if pd.notna(row.get('ph')):
                parts.append(f"Soil pH: {row['ph']}")
            if pd.notna(row.get('organic_carbon')):
                parts.append(f"Organic Carbon: {row['organic_carbon']}%")
            if pd.notna(row.get('nitrogen')):
                parts.append(f"Nitrogen: {row['nitrogen']} kg/ha")
            if pd.notna(row.get('phosphorus')):
                parts.append(f"Phosphorus: {row['phosphorus']} kg/ha")
            if pd.notna(row.get('potassium')):
                parts.append(f"Potassium: {row['potassium']} kg/ha")
            
            if parts:
                doc_text = " | ".join(parts)
                
                metadata = {
                    "source": dataset_key,
                    "category": "soil",
                    "state_name": str(row.get('state_name', '')),
                    "district_name": str(row.get('district_name', '')),
                    "dataset_title": dataset_info.get("title", dataset_key)
                }
                
                self.documents.append(doc_text)
                self.metadata.append(metadata)
    
    def _process_market_documents(self, dataset_key: str, df: pd.DataFrame, dataset_info: Dict):
        """Process market price datasets into documents"""
        for _, row in df.iterrows():
            parts = []
            
            if pd.notna(row.get('state_name')):
                parts.append(f"State: {row['state_name']}")
            if pd.notna(row.get('district_name')):
                parts.append(f"District: {row['district_name']}")
            if pd.notna(row.get('market_name')):
                parts.append(f"Market: {row['market_name']}")
            if pd.notna(row.get('commodity')):
                parts.append(f"Commodity: {row['commodity']}")
            if pd.notna(row.get('variety')):
                parts.append(f"Variety: {row['variety']}")
            if pd.notna(row.get('price')):
                parts.append(f"Price: ₹{row['price']}")
            if pd.notna(row.get('modal_price')):
                parts.append(f"Modal Price: ₹{row['modal_price']}")
            if pd.notna(row.get('date')):
                parts.append(f"Date: {row['date']}")
            
            if parts:
                doc_text = " | ".join(parts)
                
                metadata = {
                    "source": dataset_key,
                    "category": "market",
                    "state_name": str(row.get('state_name', '')),
                    "commodity": str(row.get('commodity', '')),
                    "dataset_title": dataset_info.get("title", dataset_key)
                }
                
                self.documents.append(doc_text)
                self.metadata.append(metadata)
    
    def _process_irrigation_documents(self, dataset_key: str, df: pd.DataFrame, dataset_info: Dict):
        """Process irrigation datasets into documents"""
        for _, row in df.iterrows():
            parts = []
            
            if pd.notna(row.get('state_name')):
                parts.append(f"State: {row['state_name']}")
            if pd.notna(row.get('district_name')):
                parts.append(f"District: {row['district_name']}")
            if pd.notna(row.get('irrigated_area')):
                parts.append(f"Irrigated Area: {row['irrigated_area']} hectares")
            if pd.notna(row.get('irrigation_source')):
                parts.append(f"Irrigation Source: {row['irrigation_source']}")
            if pd.notna(row.get('water_consumption')):
                parts.append(f"Water Consumption: {row['water_consumption']} cubic meters")
            
            if parts:
                doc_text = " | ".join(parts)
                
                metadata = {
                    "source": dataset_key,
                    "category": "irrigation",
                    "state_name": str(row.get('state_name', '')),
                    "district_name": str(row.get('district_name', '')),
                    "dataset_title": dataset_info.get("title", dataset_key)
                }
                
                self.documents.append(doc_text)
                self.metadata.append(metadata)
    
    def _process_water_documents(self, dataset_key: str, df: pd.DataFrame, dataset_info: Dict):
        """Process water resources datasets into documents"""
        for _, row in df.iterrows():
            parts = []
            
            if pd.notna(row.get('state_name')):
                parts.append(f"State: {row['state_name']}")
            if pd.notna(row.get('reservoir_name')):
                parts.append(f"Reservoir: {row['reservoir_name']}")
            if pd.notna(row.get('water_level')):
                parts.append(f"Water Level: {row['water_level']} meters")
            if pd.notna(row.get('capacity')):
                parts.append(f"Capacity: {row['capacity']} MCM")
            
            if parts:
                doc_text = " | ".join(parts)
                
                metadata = {
                    "source": dataset_key,
                    "category": "water",
                    "state_name": str(row.get('state_name', '')),
                    "dataset_title": dataset_info.get("title", dataset_key)
                }
                
                self.documents.append(doc_text)
                self.metadata.append(metadata)
    
    def _process_machinery_documents(self, dataset_key: str, df: pd.DataFrame, dataset_info: Dict):
        """Process agricultural machinery datasets into documents"""
        for _, row in df.iterrows():
            parts = []
            
            if pd.notna(row.get('state_name')):
                parts.append(f"State: {row['state_name']}")
            if pd.notna(row.get('machinery_type')):
                parts.append(f"Machinery: {row['machinery_type']}")
            if pd.notna(row.get('count')):
                parts.append(f"Count: {row['count']} units")
            
            if parts:
                doc_text = " | ".join(parts)
                
                metadata = {
                    "source": dataset_key,
                    "category": "machinery",
                    "state_name": str(row.get('state_name', '')),
                    "dataset_title": dataset_info.get("title", dataset_key)
                }
                
                self.documents.append(doc_text)
                self.metadata.append(metadata)
    
    def _process_insurance_documents(self, dataset_key: str, df: pd.DataFrame, dataset_info: Dict):
        """Process crop insurance datasets into documents"""
        for _, row in df.iterrows():
            parts = []
            
            if pd.notna(row.get('state_name')):
                parts.append(f"State: {row['state_name']}")
            if pd.notna(row.get('district_name')):
                parts.append(f"District: {row['district_name']}")
            if pd.notna(row.get('crop')):
                parts.append(f"Crop: {row['crop']}")
            if pd.notna(row.get('insured_area')):
                parts.append(f"Insured Area: {row['insured_area']} hectares")
            if pd.notna(row.get('claims')):
                parts.append(f"Claims: ₹{row['claims']}")
            
            if parts:
                doc_text = " | ".join(parts)
                
                metadata = {
                    "source": dataset_key,
                    "category": "insurance",
                    "state_name": str(row.get('state_name', '')),
                    "district_name": str(row.get('district_name', '')),
                    "crop": str(row.get('crop', '')),
                    "dataset_title": dataset_info.get("title", dataset_key)
                }
                
                self.documents.append(doc_text)
                self.metadata.append(metadata)
    
    def _process_general_documents(self, dataset_key: str, df: pd.DataFrame, dataset_info: Dict):
        """Process general datasets into documents"""
        for _, row in df.iterrows():
            parts = []
            
            # Include all non-null columns
            for col, value in row.items():
                if pd.notna(value) and not col.startswith('_'):
                    parts.append(f"{col.replace('_', ' ').title()}: {value}")
            
            if parts:
                doc_text = " | ".join(parts[:10])  # Limit to first 10 fields
                
                metadata = {
                    "source": dataset_key,
                    "category": "general",
                    "dataset_title": dataset_info.get("title", dataset_key)
                }
                
                self.documents.append(doc_text)
                self.metadata.append(metadata)
    
    def _similarity_search(self, query: str, k: int = 10, category_filter: Optional[str] = None) -> List[Dict]:
        """Perform similarity search with optional category filtering"""
        if self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Apply category filter if specified
        if category_filter:
            filtered_indices = [i for i, meta in enumerate(self.metadata) 
                              if meta.get("category") == category_filter]
            if filtered_indices:
                filtered_similarities = similarities[filtered_indices]
                top_filtered_indices = np.argsort(filtered_similarities)[-k:][::-1]
                top_indices = [filtered_indices[i] for i in top_filtered_indices]
            else:
                top_indices = []
        else:
            top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx])
            })
        
        return results
    
    def _create_enhanced_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Create enhanced prompt for LLM with better context organization"""
        # Organize context by category
        context_by_category = {}
        for doc in context_docs:
            category = doc["metadata"]["category"]
            if category not in context_by_category:
                context_by_category[category] = []
            context_by_category[category].append(doc["document"])
        
        # Build organized context
        context_parts = []
        for category, docs in context_by_category.items():
            context_parts.append(f"\n{category.upper()} DATA:")
            context_parts.extend(docs[:5])  # Limit per category
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert agricultural data analyst for the Government of India with access to comprehensive datasets from data.gov.in. Use the following data to answer the user's question accurately and comprehensively.

AVAILABLE DATA SOURCES:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Provide specific, accurate information based only on the data provided
2. Include exact numbers (production in tonnes, area in hectares, rainfall in mm, prices in ₹)
3. Mention specific districts, states, and years when available
4. For comparisons, clearly state the differences between regions/crops/time periods
5. If asking about "highest" or "lowest", identify the specific location and provide exact values
6. Cross-reference multiple data sources when relevant (e.g., correlate rainfall with crop production)
7. Cite data sources as "Government of India data from data.gov.in"
8. If the data is insufficient to answer fully, clearly state what information is missing

ANSWER:"""
        
        return prompt
    
    def query(self, question: str, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """Process a natural language query with enhanced capabilities"""
        try:
            if not self.documents:
                return {
                    "answer": "Sorry, the enhanced RAG system is not properly initialized with data.",
                    "sources": [],
                    "retrieved_docs": 0,
                    "categories_used": []
                }
            
            # Determine query characteristics
            question_lower = question.lower()
            
            # Auto-detect category if not specified
            if not category_filter:
                if any(word in question_lower for word in ['rainfall', 'rain', 'weather', 'climate', 'temperature']):
                    category_filter = "climate"
                elif any(word in question_lower for word in ['price', 'market', 'cost', 'rate']):
                    category_filter = "market"
                elif any(word in question_lower for word in ['soil', 'ph', 'nutrient', 'organic']):
                    category_filter = "soil"
                elif any(word in question_lower for word in ['irrigation', 'water']):
                    category_filter = "irrigation"
            
            # Perform similarity search
            k = 15 if category_filter else 20
            results = self._similarity_search(question, k=k, category_filter=category_filter)
            
            if not results:
                return {
                    "answer": "I couldn't find relevant information for your query in the available datasets.",
                    "sources": [],
                    "retrieved_docs": 0,
                    "categories_used": []
                }
            
            # Create enhanced prompt and get response
            prompt = self._create_enhanced_prompt(question, results)
            response = self.llm.generate_content(prompt)
            
            # Extract sources and categories
            sources = []
            categories_used = set()
            
            for result in results:
                metadata = result["metadata"]
                categories_used.add(metadata["category"])
                
                source_info = {
                    "dataset": metadata["source"],
                    "category": metadata["category"],
                    "title": metadata.get("dataset_title", metadata["source"]),
                    "similarity": result["similarity"]
                }
                
                # Add category-specific fields
                if metadata["category"] == "agriculture":
                    source_info.update({
                        "state": metadata.get("state_name", ""),
                        "district": metadata.get("district_name", ""),
                        "crop": metadata.get("crop", ""),
                        "year": metadata.get("year", "")
                    })
                elif metadata["category"] == "climate":
                    source_info.update({
                        "region": metadata.get("region", ""),
                        "state": metadata.get("state_name", ""),
                        "year": metadata.get("year", "")
                    })
                elif metadata["category"] == "market":
                    source_info.update({
                        "state": metadata.get("state_name", ""),
                        "commodity": metadata.get("commodity", "")
                    })
                
                sources.append(source_info)
            
            return {
                "answer": response.text,
                "sources": sources,
                "retrieved_docs": len(results),
                "categories_used": list(categories_used),
                "category_filter": category_filter,
                "system": "enhanced_rag_v2"
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "answer": f"Sorry, I encountered an error processing your question: {str(e)}",
                "sources": [],
                "retrieved_docs": 0,
                "categories_used": [],
                "system": "enhanced_rag_v2"
            }
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        category_counts = {}
        for meta in self.metadata:
            category = meta.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "status": "ready" if self.documents else "not_ready",
            "total_documents": len(self.documents),
            "total_datasets": len(self.loaded_datasets),
            "datasets_loaded": self.loaded_datasets,
            "dataset_details": self.dataset_stats,
            "documents_by_category": category_counts,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gemini-2.5-flash",
            "system_version": "enhanced_rag_v2"
        }

# Global instance
enhanced_rag_system = None

def get_enhanced_rag_system() -> EnhancedRAG:
    """Get or create global enhanced RAG system instance"""
    global enhanced_rag_system
    if enhanced_rag_system is None:
        enhanced_rag_system = EnhancedRAG()
    return enhanced_rag_system

def initialize_enhanced_rag_system():
    """Initialize the enhanced RAG system"""
    global enhanced_rag_system
    print("Initializing Enhanced RAG system...")
    enhanced_rag_system = EnhancedRAG()
    return enhanced_rag_system