"""
Project Samarth - Simplified RAG System
A simplified RAG implementation that works with current dependencies
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json

# Use existing dependencies
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class SimplifiedRAG:
    """
    Simplified RAG system for agricultural intelligence
    Uses existing sentence-transformers and direct similarity search
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
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and process agricultural data"""
        try:
            print("Loading agricultural data...")
            
            # Load crop production data
            crop_path = "data/processed/crop_production_district.parquet"
            if os.path.exists(crop_path):
                df_crop = pd.read_parquet(crop_path)
                print(f"Loaded {len(df_crop)} crop production records")
                self._process_crop_data(df_crop)
            
            # Load climate data
            climate_path = "data/processed/imd_rainfall_monthly.parquet"
            if os.path.exists(climate_path):
                df_climate = pd.read_parquet(climate_path)
                print(f"Loaded {len(df_climate)} climate records")
                self._process_climate_data(df_climate)
            
            # Create embeddings
            if self.documents:
                print(f"Creating embeddings for {len(self.documents)} documents...")
                self.embeddings = self.embedding_model.encode(self.documents)
                print("✅ RAG system initialized successfully")
            else:
                print("❌ No documents loaded")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def _process_crop_data(self, df: pd.DataFrame):
        """Process crop production data into documents"""
        # Sample data to avoid memory issues (take every 10th record)
        df_sample = df.iloc[::10].copy()
        
        for _, row in df_sample.iterrows():
            # Create document text
            parts = []
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
            
            doc_text = " | ".join(parts)
            
            # Create metadata
            metadata = {
                "source": "crop_production",
                "state_name": str(row.get('state_name', '')),
                "district_name": str(row.get('district_name', '')),
                "crop": str(row.get('crop', '')),
                "crop_year": str(row.get('crop_year', '')),
                "season": str(row.get('season', '')),
                "production": str(row.get('production', '')),
                "area": str(row.get('area', ''))
            }
            
            self.documents.append(doc_text)
            self.metadata.append(metadata)
    
    def _process_climate_data(self, df: pd.DataFrame):
        """Process climate data into documents"""
        # Sample data (take every 5th record)
        df_sample = df.iloc[::5].copy()
        
        for _, row in df_sample.iterrows():
            parts = []
            if pd.notna(row.get('sd_name')):
                parts.append(f"Subdivision: {row['sd_name']}")
            if pd.notna(row.get('year')):
                parts.append(f"Year: {row['year']}")
            if pd.notna(row.get('annual')):
                parts.append(f"Annual Rainfall: {row['annual']} mm")
            
            doc_text = " | ".join(parts)
            
            metadata = {
                "source": "climate",
                "subdivision": str(row.get('sd_name', '')),
                "year": str(row.get('year', '')),
                "annual_rainfall": str(row.get('annual', ''))
            }
            
            self.documents.append(doc_text)
            self.metadata.append(metadata)
    
    def _similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform similarity search"""
        if self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx])
            })
        
        return results
    
    def _create_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Create prompt for LLM"""
        context = "\n".join([doc["document"] for doc in context_docs])
        
        prompt = f"""You are an expert agricultural data analyst for the Government of India. Use the following data to answer the user's question accurately.

Context Data:
{context}

Question: {query}

Instructions:
1. **Primary Mode — Data-Grounded Reasoning**
   - Use all relevant numerical and factual information from the provided {context}.
   - Include exact figures (production in tonnes, rainfall in mm, area in hectares, yield in tonnes/ha, etc.).
   - Always prioritize real, verified data when present in the context.

2. **Secondary Mode — Intelligent Data Generation**
   - If a value or relationship is missing or incomplete:
     - Generate **synthetic but realistic** estimates based on statistical patterns or logical reasoning.
     - Ensure all generated values are **consistent with known domain trends** (e.g., rainfall range per state, crop yield limits, etc.).
     - Clearly mark generated values as *"AI-estimated"* or *"synthetic data (approximation)"*.
   - Use generative reasoning to **fill context gaps**, **simulate missing years**, or **project recent trends** when the dataset stops early.

3. **Analytical Precision**
   - For **comparisons**, show both absolute and relative differences (e.g., “Maharashtra received 120 mm or 15% more rainfall than Tamil Nadu”).
   - For **trends**, indicate direction (increasing/decreasing/stable), rate, and key turning points.
   - For **correlation**, describe observed relationships clearly (e.g., “a 10% rainfall drop corresponded to a 7% decline in rice yield”).
   - When analyzing **policy or sustainability**, combine quantitative evidence with qualitative reasoning.

4. **Advanced Question Handling**
   - **Highest/Lowest:** Identify top/bottom districts or states with values.
   - **Ranking:** List top M crops or regions, sorted by production or rainfall.
   - **Trends:** Describe year-wise or decade-level changes and possible causes.
   - **Policy Analysis:** Provide 2–3 concise, data-supported arguments combining climate and crop metrics.
   - **Cross-domain Questions:** Integrate agricultural, climatic, and economic data coherently.

5. **Traceability and Transparency**
   - Each numeric claim should be followed by a note on its source:
     - *(Source: Government of India dataset)* → for real data  
     - *(AI-estimated based on recent patterns)* → for generated data  
   - Do **not** mix real and generated values without indicating which is which.

6. **Formatting Rules**
   - Use concise bullet points and labeled sections:
     - **Rainfall Comparison**
     - **Crop Production Summary**
     - **Trend Analysis**
     - **Policy Insight**
   - Maintain consistent units:
     - Rainfall → mm  
     - Area → hectares  
     - Production → tonnes  
     - Yield → tonnes/hectare  
   - Provide numeric values wherever possible, even if approximate.

7. **Integrity & Realism**
   - Synthetic data must remain within realistic agricultural or climatic ranges.
   - Never produce exaggerated or implausible figures.
   - If uncertainty exists, include confidence language (e.g., “likely range: 900–1,000 mm”).

8. **Output Structure**
   - Begin with a 1-line summary of the overall finding.
   - Present detailed, structured data insights next.
   - End with a short interpretive insight or recommendation.
Answer:"""
        
        return prompt
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a natural language query"""
        try:
            if not self.documents:
                return {
                    "answer": "Sorry, the system is not properly initialized with data.",
                    "sources": [],
                    "retrieved_docs": 0
                }
            
            # Determine query type
            question_lower = question.lower()
            is_climate_query = any(word in question_lower for word in 
                                 ['rainfall', 'rain', 'precipitation', 'monsoon', 'weather', 'climate'])
            
            # Perform similarity search
            k = 10 if is_climate_query else 15  # More docs for crop queries
            results = self._similarity_search(question, k=k)
            
            if not results:
                return {
                    "answer": "I couldn't find relevant information for your query.",
                    "sources": [],
                    "retrieved_docs": 0
                }
            
            # Filter by source type if needed
            if is_climate_query:
                results = [r for r in results if r["metadata"]["source"] == "climate"][:5]
            else:
                # For crop queries, prefer crop production data
                crop_results = [r for r in results if r["metadata"]["source"] == "crop_production"]
                if crop_results:
                    results = crop_results[:10]
            
            # Create prompt and get response
            prompt = self._create_prompt(question, results)
            
            response = self.llm.generate_content(prompt)
            
            # Extract sources
            sources = []
            for result in results:
                source_info = {
                    "source_type": result["metadata"]["source"],
                    "similarity": result["similarity"]
                }
                
                if result["metadata"]["source"] == "crop_production":
                    source_info.update({
                        "state": result["metadata"]["state_name"],
                        "district": result["metadata"]["district_name"],
                        "crop": result["metadata"]["crop"],
                        "year": result["metadata"]["crop_year"]
                    })
                else:
                    source_info.update({
                        "subdivision": result["metadata"]["subdivision"],
                        "year": result["metadata"]["year"]
                    })
                
                sources.append(source_info)
            
            return {
                "answer": response.text,
                "sources": sources,
                "query_type": "climate" if is_climate_query else "crop",
                "retrieved_docs": len(results)
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "answer": f"Sorry, I encountered an error processing your question: {str(e)}",
                "sources": [],
                "query_type": "error",
                "retrieved_docs": 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "status": "ready" if self.documents else "not_ready",
            "total_documents": len(self.documents),
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gemini-2.5-flash",
            "crop_docs": len([d for d in self.metadata if d["source"] == "crop_production"]),
            "climate_docs": len([d for d in self.metadata if d["source"] == "climate"])
        }

# Global instance
simple_rag_system = None

def get_simple_rag_system() -> SimplifiedRAG:
    """Get or create global simplified RAG system instance"""
    global simple_rag_system
    if simple_rag_system is None:
        simple_rag_system = SimplifiedRAG()
    return simple_rag_system

def initialize_simple_rag_system():
    """Initialize the simplified RAG system"""
    global simple_rag_system
    print("Initializing Simplified RAG system...")
    simple_rag_system = SimplifiedRAG()
    return simple_rag_system