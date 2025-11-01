"""
Project Samarth - Frontend Configuration
Centralized configuration for professional UI components
"""

# Application Settings
APP_CONFIG = {
    "title": "Project Samarth",
    "subtitle": "Agricultural Intelligence Platform",
    "description": "Intelligent Q&A System over Government Agricultural & Climate Data",
    "icon": "🌾",
    "version": "1.0.0"
}

# API Configuration
API_CONFIG = {
    "backend_url": "http://localhost:8000",
    "timeout": 60,
    "retry_attempts": 3
}

# UI Theme Configuration
THEME_CONFIG = {
    "primary_color": "#667eea",
    "secondary_color": "#764ba2",
    "success_color": "#28a745",
    "warning_color": "#ffc107",
    "error_color": "#dc3545",
    "background_gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "card_shadow": "0 5px 15px rgba(0,0,0,0.08)",
    "border_radius": "15px"
}

# Data Source Information
DATA_SOURCES = {
    "crop_production": {
        "name": "Ministry of Agriculture & Farmers Welfare",
        "description": "District-wise crop production data",
        "records": "246,091",
        "years": "Multiple years",
        "url": "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de"
    },
    "rainfall": {
        "name": "India Meteorological Department",
        "description": "State-wise rainfall data",
        "records": "2,300+",
        "years": "1951-2014",
        "url": "https://api.data.gov.in/resource/294a162a-92fb-4939-af88-e69bd84049f1"
    }
}

# Example Queries by Category
EXAMPLE_QUERIES = {
    "Climate Analysis": [
        "Compare average rainfall in Karnataka and Maharashtra for the last 5 years",
        "Which state has the highest annual rainfall in India?",
        "Show rainfall trends in Gujarat over the last decade",
        "What is the average monsoon rainfall in Kerala?",
        "Compare drought patterns between Rajasthan and Maharashtra"
    ],
    "Crop Production": [
        "What are the top rice producing districts in Tamil Nadu?",
        "Compare wheat production between Punjab and Haryana",
        "Which crops are most commonly grown in Kerala?",
        "Show cotton production trends in Gujarat",
        "List top 5 sugarcane producing states in India"
    ],
    "Cross-domain Analysis": [
        "Correlate rainfall patterns with rice production in West Bengal",
        "How does monsoon affect crop yields in Maharashtra?",
        "Analyze climate-agriculture relationship in drought-prone regions",
        "Impact of rainfall on wheat production in northern states",
        "Relationship between annual precipitation and crop diversity"
    ],
    "Policy & Planning": [
        "Which regions should focus on drought-resistant crops?",
        "Identify states with water-intensive crop production",
        "Compare agricultural productivity across climate zones",
        "Suggest crop diversification strategies for low-rainfall areas",
        "Analyze food security implications of climate patterns"
    ]
}

# Dashboard Metrics Configuration
DASHBOARD_METRICS = {
    "total_records": {
        "value": "246,091",
        "label": "Total Records",
        "icon": "📊",
        "color": "#667eea"
    },
    "states_covered": {
        "value": "36",
        "label": "States Covered",
        "icon": "🗺️",
        "color": "#28a745"
    },
    "data_years": {
        "value": "64",
        "label": "Years of Data",
        "icon": "📅",
        "color": "#ffc107"
    },
    "accuracy": {
        "value": "99.7%",
        "label": "Data Accuracy",
        "icon": "✅",
        "color": "#17a2b8"
    }
}

# Chart Configuration
CHART_CONFIG = {
    "default_theme": "plotly_white",
    "color_palette": [
        "#667eea", "#764ba2", "#f093fb", "#f5576c",
        "#4facfe", "#00f2fe", "#43e97b", "#38f9d7"
    ],
    "font_family": "Inter, sans-serif",
    "title_font_size": 16,
    "axis_font_size": 12
}

# Status Messages
STATUS_MESSAGES = {
    "backend_online": "✅ Backend Service Online",
    "backend_offline": "❌ Backend Service Offline", 
    "backend_error": "⚠️ Backend Service Error",
    "processing": "🔄 Processing your query...",
    "completed": "✅ Analysis completed successfully",
    "error": "❌ Query processing failed",
    "no_data": "📭 No data available for this query"
}

# Help Text and Tooltips
HELP_TEXT = {
    "query_input": "Ask questions about crops, climate, production trends, or correlations between agricultural and meteorological data",
    "example_queries": "Click on any example query to use it as a starting point",
    "data_sources": "All data is sourced directly from official government portals",
    "response_time": "Average response time for complex queries",
    "accuracy": "Data accuracy based on government source validation"
}

# Feature Flags
FEATURES = {
    "enable_history": True,
    "enable_analytics": True,
    "enable_export": True,
    "enable_visualizations": True,
    "enable_real_time_stats": True,
    "enable_query_suggestions": True
}