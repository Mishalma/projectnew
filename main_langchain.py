# backend/main_langchain.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.enhanced_rag import get_enhanced_rag_system, initialize_enhanced_rag_system
from backend.simple_rag import get_simple_rag_system, initialize_simple_rag_system
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Project Samarth API v3.0",
    description="Agricultural Intelligence Platform with Enhanced RAG and data.gov.in Integration",
    version="3.0.0"
)

# Initialize RAG system on startup
rag_system = None
use_enhanced_rag = True  # Toggle between enhanced and simple RAG

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system when the API starts"""
    global rag_system
    try:
        if use_enhanced_rag:
            print("🚀 Starting Project Samarth with Enhanced RAG...")
            rag_system = initialize_enhanced_rag_system()
        else:
            print("🚀 Starting Project Samarth with Simple RAG...")
            rag_system = initialize_simple_rag_system()
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        # Don't fail startup, but log the error
        rag_system = None

@app.get("/")
async def root():
    """API health check and information"""
    rag_stats = {}
    if rag_system:
        if use_enhanced_rag:
            rag_stats = rag_system.get_enhanced_stats()
        else:
            rag_stats = rag_system.get_stats()
    
    return {
        "message": "Project Samarth API v3.0 - Enhanced RAG with data.gov.in Integration",
        "status": "online",
        "endpoints": ["/ask", "/health", "/stats", "/datasets"],
        "rag_system": rag_stats,
        "enhanced_rag": use_enhanced_rag
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if not rag_system:
        return {"status": "error", "message": "RAG system not initialized"}
    
    if use_enhanced_rag:
        stats = rag_system.get_enhanced_stats()
    else:
        stats = rag_system.get_stats()
    
    return {
        "status": "healthy" if stats.get("status") == "ready" else "degraded",
        "rag_system": stats,
        "api_version": "3.0.0",
        "enhanced_rag": use_enhanced_rag
    }

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    if not rag_system:
        return {"error": "RAG system not available"}
    
    if use_enhanced_rag:
        return rag_system.get_enhanced_stats()
    else:
        return rag_system.get_stats()

@app.get("/datasets")
async def get_datasets():
    """Get information about available datasets"""
    if not rag_system or not use_enhanced_rag:
        return {"error": "Enhanced RAG system not available"}
    
    stats = rag_system.get_enhanced_stats()
    return {
        "datasets": stats.get("dataset_details", {}),
        "categories": stats.get("documents_by_category", {}),
        "total_datasets": stats.get("total_datasets", 0),
        "total_documents": stats.get("total_documents", 0)
    }

class AskPayload(BaseModel):
    question: str

@app.post("/ask")
async def ask(payload: AskPayload):
    """
    Process natural language queries using LangChain RAG system
    """
    try:
        if not rag_system:
            raise HTTPException(
                status_code=503, 
                detail="RAG system not available. Please check system initialization."
            )
        
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"🔍 Processing query: {question}")
        
        # Process query using LangChain RAG
        result = rag_system.query(question)
        
        print(f"✅ Query processed successfully. Retrieved {result.get('retrieved_docs', 0)} documents")
        
        # Format response for compatibility with existing frontend
        response = {
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_documents": result["retrieved_docs"],
            "system": "enhanced_rag_v3" if use_enhanced_rag else "simple_rag_v2"
        }
        
        # Add enhanced RAG specific fields
        if use_enhanced_rag:
            response.update({
                "categories_used": result.get("categories_used", []),
                "category_filter": result.get("category_filter"),
            })
        else:
            response["query_type"] = result.get("query_type", "general")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in /ask endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing your query or contact support if the issue persists.",
            "sources": [],
            "retrieved_documents": 0,
            "system": "enhanced_rag_v3" if use_enhanced_rag else "simple_rag_v2",
            "categories_used": [],
            "query_type": "error"
        }