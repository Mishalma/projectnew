#!/usr/bin/env python3
"""
Project Samarth - LangChain RAG Launcher
Starts both LangChain backend and dashboard services
"""

import subprocess
import sys
import time
import os

def start_backend():
    """Start the LangChain FastAPI backend server"""
    print("🚀 Starting LangChain RAG backend server...")
    backend_process = subprocess.Popen([
        sys.executable, '-m', 'uvicorn', 
        'backend.main_langchain:app', 
        '--reload', 
        '--port', '8000',
        '--host', 'localhost'
    ])
    return backend_process

def start_frontend():
    """Start the Streamlit dashboard"""
    print("🌐 Starting dashboard...")
    
    frontend_process = subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run',
        'frontend/dashboard.py',
        '--server.port', '8501',
        '--server.address', 'localhost',
        '--theme.primaryColor', '#2E8B57',
        '--theme.backgroundColor', '#ffffff',
        '--theme.secondaryBackgroundColor', '#f0f2f6'
    ])
    return frontend_process

def main():
    """Main launcher function"""
    print("🌾 Project Samarth - LangChain RAG System")
    print("=" * 60)
    
    processes = []
    
    try:
        # Start backend
        backend_proc = start_backend()
        processes.append(backend_proc)
        
        # Wait for backend to start
        print("⏳ Waiting for backend to initialize...")
        time.sleep(5)
        
        # Start frontend
        frontend_proc = start_frontend()
        processes.append(frontend_proc)
        
        print("\n" + "=" * 60)
        print("🎉 Project Samarth LangChain RAG is now running!")
        print("=" * 60)
        print(f"🔧 Backend API: http://localhost:8000")
        print(f"🌐 Dashboard: http://localhost:8501")
        print("\n💡 Open http://localhost:8501 in your browser")
        print("⏹️  Press Ctrl+C to stop all services")
        print("=" * 60)
        
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            for proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  Process {proc.pid} has stopped")
                    return
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Project Samarth...")
        
        # Terminate all processes
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        
        print("✅ All services stopped successfully")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Clean up processes
        for proc in processes:
            try:
                proc.terminate()
            except:
                pass

if __name__ == "__main__":
    main()