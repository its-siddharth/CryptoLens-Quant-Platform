import streamlit as st

def apply_custom_css():
    """Apply premium trading terminal dark theme, custom fonts, glassmorphism, and modern visual states."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Apply fonts */
        html, body, [class*="css"], .stWidgetFormValue {
            font-family: 'Outfit', sans-serif !important;
        }
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* Background & container overrides */
        .stApp {
            background: linear-gradient(135deg, #090e17 0%, #0d1527 100%) !important;
            color: #e2e8f0 !important;
        }
        
        /* Clean sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0b1120 !important;
            border-right: 1px solid #1e293b !important;
        }
        
        /* Sidebar headers & labels */
        [data-testid="stSidebar"] .stMarkdown p {
            color: #94a3b8 !important;
        }
        
        /* Card-like aesthetic for metrics with modern glassmorphism */
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.6) !important;
            backdrop-filter: blur(12px) !important;
            border-radius: 16px !important;
            padding: 20px 24px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-4px) !important;
            border-color: rgba(56, 189, 248, 0.4) !important;
            box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.15) !important;
        }
        
        /* Custom card wrapper for non-metric blocks */
        .premium-card {
            background: rgba(15, 23, 42, 0.5) !important;
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.06);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .premium-card:hover {
            border-color: rgba(56, 189, 248, 0.2);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        }
        
        /* Gradient H1 Header Styling */
        h1 {
            background: linear-gradient(135deg, #38bdf8 0%, #0369a1 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-weight: 800 !important;
            letter-spacing: -0.025em !important;
        }
        
        /* Styled sub-headers */
        h2, h3 {
            color: #f8fafc !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }
        
        /* Custom tabs styling */
        button[data-baseweb="tab"] {
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            color: #94a3b8 !important;
            border-bottom: 2px solid transparent !important;
            transition: all 0.2s ease !important;
            padding: 10px 16px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8 !important;
            border-bottom: 2px solid #38bdf8 !important;
            font-weight: 600 !important;
        }
        
        /* Streamlit divider customization */
        hr {
            border-color: rgba(255, 255, 255, 0.08) !important;
            margin: 24px 0 !important;
        }
        
        /* Form inputs customization */
        div[data-baseweb="select"] > div, input, textarea {
            background-color: #0f172a !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: #f1f5f9 !important;
            border-radius: 8px !important;
            transition: all 0.25s ease !important;
        }
        
        /* Premium button styles */
        div.stButton > button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            padding: 8px 22px !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.025em !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.15) !important;
            width: auto !important;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.25) !important;
            border-color: rgba(56, 189, 248, 0.4) !important;
            color: #ffffff !important;
        }
        div.stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Focus outlines */
        input:focus, textarea:focus, div[data-baseweb="select"] > div:focus {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
        }
        
        /* Interactive Live Status Indicator */
        .pulse-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse 1.6s infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        @keyframes pulse {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }
        
        /* Glassmorphic custom notification container */
        .custom-alert {
            padding: 16px 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border-left: 4px solid;
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(8px);
        }
        .alert-info { border-left-color: #38bdf8; color: #bae6fd; }
        .alert-success { border-left-color: #10b981; color: #a7f3d0; }
        .alert-warning { border-left-color: #f59e0b; color: #fef3c7; }
        .alert-error { border-left-color: #ef4444; color: #fee2e2; }
        
        /* Modern scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #090e17;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #334155;
        }
        </style>
    """, unsafe_allow_html=True)
