import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import time

# Try to import plotly, fallback to basic charts if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Project Samarth - Agricultural Intelligence Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 3rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header h3 {
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.8;
        margin: 0;
    }
    
    .query-section {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 2rem 0;
        border: 1px solid #e1e8ed;
    }
    
    .results-section {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 2rem 0;
        border: 1px solid #e1e8ed;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        font-size: 1.1rem;
        padding: 1.2rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea:focus {
        border-color: #2E8B57;
        box-shadow: 0 0 0 3px rgba(46, 139, 87, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(46, 139, 87, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 139, 87, 0.4);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .answer-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #2E8B57;
        margin: 1.5rem 0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .source-citation {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-online {
        background: #d4edda;
        color: #155724;
    }
    
    .status-offline {
        background: #f8d7da;
        color: #721c24;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2E8B57;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e1e8ed;
    }
    
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌾 Project Samarth</h1>
    <h3>Agricultural Intelligence Platform</h3>
    <p>Intelligent Q&A System over Government Agricultural & Climate Data</p>
</div>
""", unsafe_allow_html=True)

# Check backend status
try:
    response = requests.get("http://localhost:8000/", timeout=5)
    backend_status = response.status_code == 200
    if backend_status:
        backend_info = response.json()
    else:
        backend_info = None
except Exception as e:
    backend_status = False
    backend_info = None

# Status indicator
status_class = "status-online" if backend_status else "status-offline"
status_text = "System Online" if backend_status else "System Offline"
st.markdown(f'<div class="status-badge {status_class}">{status_text}</div>', unsafe_allow_html=True)

# Main query section
st.markdown('<div class="query-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Ask Your Question</div>', unsafe_allow_html=True)

# Initialize session state and clean up corrupted widgets
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""

# Clean up any corrupted selectbox states
corrupted_keys = []
for key in st.session_state.keys():
    if "selectbox" in key and key in st.session_state:
        try:
            value = st.session_state[key]
            if isinstance(value, str) and value.isdigit():
                # This indicates a corrupted state where string index is stored
                corrupted_keys.append(key)
        except:
            corrupted_keys.append(key)

for key in corrupted_keys:
    del st.session_state[key]

# Query input
default_query = st.session_state.get('selected_query', '')
query = st.text_area(
    "Enter your agricultural query",
    value=default_query,
    height=120,
    placeholder="Enter your question about crops, climate, or agricultural data...\n\nExample: Compare average rainfall in Karnataka and Maharashtra for the last 5 years",
    label_visibility="collapsed",
    key="query_input"
)

# Action buttons
col1, col2 = st.columns([3, 1])

with col1:
    ask_button = st.button("🚀 Analyze Question", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)
    if clear_button:
        st.session_state.selected_query = ""
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Debug info (remove in production)
if st.checkbox("Show Debug Info", value=False):
    st.write(f"Backend Status: {backend_status}")
    st.write(f"Backend Info: {backend_info}")
    st.write(f"Current Query: '{query}'")
    st.write(f"Query Length: {len(query.strip()) if query else 0}")
    
    # Session state reset button for debugging
    if st.button("🔄 Reset Session State", key="reset_session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session state cleared! Please refresh the page.")
        st.rerun()

# Process query
if ask_button and query.strip() and backend_status:
    with st.spinner("🔄 Processing your query... This may take up to 60 seconds for complex analysis."):
        try:
            # Make API request
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": query},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Results section
                st.markdown('<div class="results-section">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
                
                # Main answer
                st.markdown("### 💬 AI Analysis")
                st.markdown(f"""
                <div class="answer-box">
                    {result.get('answer', 'No answer provided')}
                </div>
                """, unsafe_allow_html=True)
                
                # Numeric results
                if result.get('numeric'):
                    st.markdown("### 📊 Key Metrics")
                    
                    numeric_data = result['numeric']
                    
                    # Display metrics in cards
                    cols = st.columns(min(len(numeric_data), 4))
                    for i, (key, value) in enumerate(numeric_data.items()):
                        with cols[i % 4]:
                            display_key = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)):
                                st.metric(display_key, f"{value:,.1f}")
                            else:
                                st.metric(display_key, str(value))
                    
                    # Visualization
                    numeric_values = {k: v for k, v in numeric_data.items() if isinstance(v, (int, float))}
                    if len(numeric_values) > 1:
                        st.markdown("### 📈 Data Visualization")
                        
                        if PLOTLY_AVAILABLE:
                            df_viz = pd.DataFrame(list(numeric_values.items()), columns=['Metric', 'Value'])
                            
                            fig = px.bar(
                                df_viz, 
                                x='Metric', 
                                y='Value',
                                title="Comparative Analysis",
                                color='Value',
                                color_continuous_scale='Greens'
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_family="Inter",
                                title_font_size=18,
                                showlegend=False
                            )
                            fig.update_traces(
                                texttemplate='%{y:.1f}',
                                textposition='outside'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback to Streamlit's built-in charts
                            df_viz = pd.DataFrame(list(numeric_values.items()), columns=['Metric', 'Value'])
                            st.bar_chart(df_viz.set_index('Metric'))
                
                # Data sources
                if result.get('provenance'):
                    st.markdown("### 🔗 Data Sources")
                    
                    # Group by dataset
                    sources = {}
                    for prov in result['provenance']:
                        dataset = prov.get('_dataset_id', 'Unknown')
                        if dataset not in sources:
                            sources[dataset] = {
                                'url': prov.get('_resource_url', ''),
                                'count': 0
                            }
                        sources[dataset]['count'] += 1
                    
                    for dataset, info in sources.items():
                        st.markdown(f"""
                        <div class="source-citation">
                            <strong>📊 {dataset.replace('_', ' ').title()}</strong><br>
                            <small>🔗 <a href="{info['url']}" target="_blank">{info['url']}</a></small><br>
                            <small>📈 {info['count']} records analyzed</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Query metadata
                
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to backend. Please ensure the server is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif ask_button and not backend_status:
    st.error("❌ Backend service is not available. Please start the backend server first.")

elif ask_button and not query.strip():
    st.warning("⚠️ Please enter a question before submitting.")

# Footer
st.markdown("""
<div class="footer">
    <h4>🌾 Project Samarth</h4>
    <p><strong>Agricultural Intelligence Platform</strong></p>
    <p>Powered by data.gov.in • Ministry of Agriculture & Farmers Welfare • India Meteorological Department</p>
    <p><small>Built with Streamlit, FastAPI, ChromaDB, and Google Gemini AI</small></p>
</div>
""", unsafe_allow_html=True)