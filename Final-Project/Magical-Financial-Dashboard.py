import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="NeoFinance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Background images and styling
def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Futuristic neon styling
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-family: 'Arial', sans-serif;
        color: #00f2ff;
        text-align: center;
        text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff, 0 0 30px #00f2ff;
        font-size: 3em;
        padding: 20px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .sub-header {
        font-family: 'Arial', sans-serif;
        color: #ff00dd;
        text-align: center;
        text-shadow: 0 0 5px #ff00dd, 0 0 10px #ff00dd;
        font-size: 1.5em;
        margin-bottom: 30px;
    }
    
    .dashboard-card {
        background: rgba(31, 31, 46, 0.7);
        border: 1px solid #00f2ff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 15px #00f2ff;
    }
    
    .neon-button {
        background: #111111;
        color: #00f2ff;
        border: 1px solid #00f2ff;
        border-radius: 5px;
        padding: 10px 20px;
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
        text-shadow: 0 0 5px #00f2ff;
        box-shadow: 0 0 10px #00f2ff;
    }
    
    .neon-button:hover {
        background: #00f2ff;
        color: #111111;
        box-shadow: 0 0 20px #00f2ff;
    }
    
    .metric-container {
        background: rgba(17, 17, 17, 0.8);
        border: 1px solid #ff00dd;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 0 10px #ff00dd;
    }
    
    .metric-value {
        color: #ff00dd;
        font-size: 1.8em;
        font-weight: bold;
        text-shadow: 0 0 5px #ff00dd;
    }
    
    .metric-label {
        color: #ffffff;
        font-size: 0.9em;
    }
    
    @keyframes glow {
        from {
            text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff;
        }
        to {
            text-shadow: 0 0 15px #00f2ff, 0 0 30px #00f2ff, 0 0 40px #00f2ff;
        }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(17, 17, 17, 0.7);
        border: 1px solid #00f2ff;
        border-radius: 5px;
        color: #00f2ff;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 242, 255, 0.2);
        border: 1px solid #00f2ff;
        border-radius: 5px;
        color: #ffffff;
        box-shadow: 0 0 10px #00f2ff;
    }
    
    /* Loading animation */
    .loader {
        width: 100%;
        height: 5px;
        background: linear-gradient(to right, #00f2ff, #ff00dd);
        position: relative;
        overflow: hidden;
        border-radius: 5px;
        animation: loading 2s infinite ease-in-out;
    }
    
    @keyframes loading {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(100%);
        }
    }
    
    .stDataFrame {
        background: rgba(31, 31, 46, 0.8);
        border: 1px solid #00f2ff;
        border-radius: 10px;
        box-shadow: 0 0 10px #00f2ff;
    }
    
    .custom-info-box {
        background: rgba(0, 242, 255, 0.1);
        border: 1px solid #00f2ff;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    
    .custom-warning-box {
        background: rgba(255, 0, 221, 0.1);
        border: 1px solid #ff00dd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    
    /* Logo spinning animation */
    .logo-spin {
        animation: spin 10s linear infinite;
    }
    
    @keyframes spin {
        from {transform: rotate(0deg);}
        to {transform: rotate(360deg);}
    }
    
    /* Neon datepicker */
    .stDateInput > div > div {
        border: 1px solid #00f2ff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 5px #00f2ff !important;
    }
    
    /* Neon selectbox */
    .stSelectbox > div > div {
        border: 1px solid #00f2ff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 5px #00f2ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply styling
local_css()

# Background image URL - using the first futuristic background
bg_url = "https://pixabay.com/get/g5eea72b5c63cd4c7722161db8babae53565d276312a9ee43e44be256eb29277b2783063ab0d492f4e821f0450bd3de2864ec17936ad248300ee634041fe9f1c4_1280.jpg"
add_bg_from_url(bg_url)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'data' not in st.session_state:
    st.session_state.data = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = {}
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None

# SVG logo for futuristic theme
def get_neon_logo_svg():
    svg_code = '''
    <svg width="150" height="150" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <filter id="neon1" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2" result="blur"/>
                <feFlood flood-color="#00f2ff" flood-opacity="1" result="neon"/>
                <feComposite in="neon" in2="blur" operator="in" result="comp"/>
                <feMerge>
                    <feMergeNode in="comp"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <filter id="neon2" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="5" result="blur"/>
                <feFlood flood-color="#00f2ff" flood-opacity="0.7" result="neon"/>
                <feComposite in="neon" in2="blur" operator="in" result="comp"/>
                <feMerge>
                    <feMergeNode in="comp"/>
                </feMerge>
            </filter>
        </defs>
        <circle cx="75" cy="75" r="60" fill="none" stroke="#00f2ff" stroke-width="2" filter="url(#neon1)"/>
        <circle cx="75" cy="75" r="45" fill="none" stroke="#00f2ff" stroke-width="3" filter="url(#neon1)"/>
        <path d="M55,55 L95,95 M55,95 L95,55" stroke="#00f2ff" stroke-width="4" filter="url(#neon1)"/>
        <circle cx="75" cy="75" r="30" fill="none" stroke="#ff00dd" stroke-width="2" filter="url(#neon1)"/>
        <text x="75" y="80" text-anchor="middle" fill="#00f2ff" font-family="Arial" font-size="12" filter="url(#neon1)">NEOFIN</text>
    </svg>
    '''
    return svg_code

# Navigation functions
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# Sidebar navigation
with st.sidebar:
    st.markdown(f'<div class="logo-spin">{get_neon_logo_svg()}</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">NeoFinance Analytics</h2>', unsafe_allow_html=True)
    st.markdown('<div style="background: rgba(31, 31, 46, 0.7); padding: 10px; border-radius: 5px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)
    
    # Define navigation button functions
    def nav_welcome():
        navigate_to('welcome')
        
    def nav_dashboard():
        navigate_to('dashboard')
        
    def nav_stocks():
        navigate_to('stocks')
        
    def nav_ml():
        navigate_to('ml')
        
    def nav_comparison():
        navigate_to('comparison')
    
    # Navigation buttons
    st.button("🏠 Welcome", key="nav_welcome", on_click=nav_welcome)
    st.button("📊 Dashboard", key="nav_dashboard", on_click=nav_dashboard)
    st.button("📈 Stock Analysis", key="nav_stocks", on_click=nav_stocks)
    st.button("🤖 ML Analytics", key="nav_ml", on_click=nav_ml)
    st.button("📊 Stock Comparison", key="nav_comparison", on_click=nav_comparison)
    
    st.markdown('<div style="background: rgba(31, 31, 46, 0.7); padding: 10px; border-radius: 5px; margin-top: 20px;"></div>', unsafe_allow_html=True)
    
    # Data upload section
    st.markdown('<h3 style="color:#ff00dd; text-shadow: 0 0 3px #ff00dd;">Upload Data</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload financial dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error uploading file: {e}")

# Function to display welcome animation
def show_welcome_animation():
    st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
    
    # Welcome animation messages
    welcome_messages = [
        "Initializing NeoFinance systems...",
        "Connecting to quantum finance network...",
        "Calibrating predictive algorithms...",
        "Establishing secure data channels...",
        "Loading futuristic interface..."
    ]
    
    message_placeholder = st.empty()
    for message in welcome_messages:
        message_placeholder.markdown(f"<p style='color:#00f2ff; text-shadow: 0 0 5px #00f2ff;'>{message}</p>", unsafe_allow_html=True)
        time.sleep(0.5)
    
    message_placeholder.empty()
    
    # Final welcome message with animation
    st.markdown(
        """
        <div style="text-align: center; animation: fadeIn 2s;">
            <h1 class="main-header">WELCOME TO NEOFINANCE</h1>
            <p class="sub-header">The Future of Financial Analytics</p>
        </div>
        <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

# Function to get stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None

# Function to create ML model
def create_ml_model(data, model_type, target_col, feature_cols, test_size=0.2):
    # Check if data and necessary columns exist
    if data is None or target_col not in data.columns:
        return None, None, None, "No data available or target column not found."
    
    # Check if all feature columns exist
    for col in feature_cols:
        if col not in data.columns:
            return None, None, None, f"Feature column '{col}' not found in data."
    
    # Prepare data
    X = data[feature_cols]
    y = data[target_col]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features for clustering
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if model_type == 'kmeans':
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = None
    performance = None
    prediction = None
    error_msg = None
    
    try:
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            performance = {
                'r2_score': model.score(X_test, y_test),
                'mse': mean_squared_error(y_test, prediction),
                'rmse': np.sqrt(mean_squared_error(y_test, prediction)),
                'coefficients': dict(zip(feature_cols, model.coef_))
            }
            
        elif model_type == 'logistic':
            # For logistic regression, we need binary target
            if len(np.unique(y)) > 2:
                # Convert to binary if not already (for demo purposes)
                y_binary = (y > y.median()).astype(int)
                y_train = (y_train > y_train.median()).astype(int)
                y_test = (y_test > y_test.median()).astype(int)
            else:
                y_binary = y
                
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            performance = {
                'accuracy': accuracy_score(y_test, prediction),
                'coefficients': dict(zip(feature_cols, model.coef_[0]))
            }
            
        elif model_type == 'kmeans':
            # Determine optimal number of clusters using elbow method
            inertias = []
            K_range = range(1, min(10, len(X_train)))
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_train_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point (simplified)
            optimal_k = 3  # Default
            for i in range(1, len(inertias)-1):
                if (inertias[i-1] - inertias[i]) / (inertias[i] - inertias[i+1]) > 2:
                    optimal_k = i + 1
                    break
            
            model = KMeans(n_clusters=optimal_k, random_state=42)
            model.fit(X_train_scaled)
            prediction = model.predict(X_test_scaled)
            performance = {
                'inertia': model.inertia_,
                'optimal_clusters': optimal_k,
                'cluster_centers': model.cluster_centers_.tolist()
            }
        
        return model, prediction, performance, error_msg
    
    except Exception as e:
        error_msg = f"Error creating model: {str(e)}"
        return None, None, None, error_msg

# ========================= WELCOME PAGE =========================
if st.session_state.page == 'welcome':
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        show_welcome_animation()
        
        st.markdown(
            """
            <div class="dashboard-card">
                <h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">About NeoFinance Analytics</h2>
                <p>Welcome to the future of financial analysis. NeoFinance combines cutting-edge machine learning with real-time market data to bring you insights that were once thought impossible.</p>
                <p>This advanced platform leverages the power of artificial intelligence to predict market trends, identify investment opportunities, and optimize your financial strategies.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Feature overview
        st.markdown(
            """
            <div class="dashboard-card">
                <h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">Key Features</h2>
                <ul>
                    <li><span style="color:#ff00dd;">Real-time Stock Analysis</span> - Access live market data with interactive visualizations</li>
                    <li><span style="color:#ff00dd;">Predictive ML Models</span> - Forecast market trends using advanced algorithms</li>
                    <li><span style="color:#ff00dd;">Interactive Dashboards</span> - Customize your financial insights</li>
                    <li><span style="color:#ff00dd;">Data Integration</span> - Upload and analyze your own financial datasets</li>
                    <li><span style="color:#ff00dd;">Stock Comparison</span> - Compare multiple stocks with advanced metrics</li>
                </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Get started button
        st.markdown(
            """
            <div style="text-align: center; margin-top: 30px;">
                <button class="neon-button" onclick="parent.document.querySelector('[key=nav_dashboard]').click();">
                    Launch Dashboard
                </button>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Images for visual effect
        cols = st.columns(3)
        with cols[0]:
            st.markdown(
                f"""
                <div style="border: 1px solid #00f2ff; border-radius: 10px; overflow: hidden; box-shadow: 0 0 15px #00f2ff;">
                    <img src="https://pixabay.com/get/g27544873e4cb16c3213a7cc9342c6d91be2af74428e0c866a1b0a76b99d95c3bd8b87cc133e44b314ab65cea60f8573a699827b4c6fd80043d3479bbe1f23242_1280.jpg" style="width: 100%; height: auto;">
                </div>
                """, 
                unsafe_allow_html=True
            )
        with cols[1]:
            st.markdown(
                f"""
                <div style="border: 1px solid #ff00dd; border-radius: 10px; overflow: hidden; box-shadow: 0 0 15px #ff00dd; margin-top: 20px;">
                    <img src="https://pixabay.com/get/g723aa3b1d1244312889602756ebb5317d3afd626c7edbf34464de6f13bdeaf09eaf91edb006de4dc7b5b290bc7aaa1e975ae00797a1674dbc4e71d309c02786e_1280.jpg" style="width: 100%; height: auto;">
                </div>
                """, 
                unsafe_allow_html=True
            )
        with cols[2]:
            st.markdown(
                f"""
                <div style="border: 1px solid #00f2ff; border-radius: 10px; overflow: hidden; box-shadow: 0 0 15px #00f2ff;">
                    <img src="https://pixabay.com/get/g18279ab9e857076e84e0645c44348bafffed92722add60843739f805adb7260ac719c455d409848f7ba834aff08ebd3d89e33e4f68048c975dec77e0c0958692_1280.jpg" style="width: 100%; height: auto;">
                </div>
                """, 
                unsafe_allow_html=True
            )

# ========================= DASHBOARD PAGE =========================
elif st.session_state.page == 'dashboard':
    st.markdown('<h1 class="main-header">NEOFINANCE DASHBOARD</h1>', unsafe_allow_html=True)
    
    # Create tabs for different dashboard sections
    tab1, tab2, tab3 = st.tabs(["Market Overview", "Portfolio Analysis", "Economic Indicators"])
    
    with tab1:
        st.markdown('<h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">Market Overview</h2>', unsafe_allow_html=True)
        
        # Market overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                """
                <div class="metric-container">
                    <div class="metric-value">32,845.02</div>
                    <div class="metric-label">DJIA</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div class="metric-container">
                    <div class="metric-value">4,186.73</div>
                    <div class="metric-label">S&P 500</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with col3:
            st.markdown(
                """
                <div class="metric-container">
                    <div class="metric-value">13,562.81</div>
                    <div class="metric-label">NASDAQ</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with col4:
            st.markdown(
                """
                <div class="metric-container">
                    <div class="metric-value">1,892.21</div>
                    <div class="metric-label">Russell 2000</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Market indices chart
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Major Market Indices - Last 12 Months")
        
        # Sample data for demonstration
        dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
        np.random.seed(42)  # For reproducibility
        
        # Generate some random but somewhat realistic market data
        djia_base = 30000 + np.cumsum(np.random.normal(0, 300, len(dates)))
        sp500_base = 4000 + np.cumsum(np.random.normal(0, 40, len(dates)))
        nasdaq_base = 12000 + np.cumsum(np.random.normal(0, 150, len(dates)))
        
        # Add some correlation between indices
        correlation_factor = 0.7
        nasdaq_base = nasdaq_base * (1-correlation_factor) + correlation_factor * (djia_base / 30000 * 12000)
        sp500_base = sp500_base * (1-correlation_factor) + correlation_factor * (djia_base / 30000 * 4000)
        
        # Create DataFrame
        indices_df = pd.DataFrame({
            'Date': dates,
            'DJIA': djia_base,
            'S&P 500': sp500_base,
            'NASDAQ': nasdaq_base
        })
        
        # Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=indices_df['Date'], y=indices_df['DJIA'], name='DJIA', line=dict(color='#00f2ff', width=2)))
        fig.add_trace(go.Scatter(x=indices_df['Date'], y=indices_df['S&P 500'], name='S&P 500', line=dict(color='#ff00dd', width=2)))
        fig.add_trace(go.Scatter(x=indices_df['Date'], y=indices_df['NASDAQ'], name='NASDAQ', line=dict(color='#ffff00', width=2)))
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(17, 17, 17, 0.9)',
            paper_bgcolor='rgba(17, 17, 17, 0)',
            font=dict(family="Arial, sans-serif", size=12, color="#ffffff"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                showgrid=False,
                linecolor='rgba(255, 255, 255, 0.2)',
                title="Date"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(255, 255, 255, 0.2)',
                title="Price"
            ),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Top gainers and losers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Top Gainers")
            gainers_data = {
                'Symbol': ['TSLA', 'NVDA', 'AMZN', 'MSFT', 'AAPL'],
                'Price': ['$245.29', '$850.45', '$178.92', '$350.12', '$189.43'],
                'Change %': ['+4.2%', '+3.8%', '+3.5%', '+2.9%', '+2.4%']
            }
            gainers_df = pd.DataFrame(gainers_data)
            st.dataframe(gainers_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Top Losers")
            losers_data = {
                'Symbol': ['META', 'NFLX', 'GME', 'INTC', 'IBM'],
                'Price': ['$320.18', '$410.75', '$34.29', '$45.63', '$128.91'],
                'Change %': ['-3.7%', '-2.9%', '-2.8%', '-2.6%', '-2.3%']
            }
            losers_df = pd.DataFrame(losers_data)
            st.dataframe(losers_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">Portfolio Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            # Use uploaded data
            df = st.session_state.data
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Portfolio Overview")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Portfolio metrics if data contains these columns
            if all(col in df.columns for col in ['Symbol', 'Quantity', 'Purchase Price']):
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.subheader("Portfolio Metrics")
                
                # Calculate portfolio metrics
                portfolio_value = (df['Quantity'] * df['Purchase Price']).sum()
                portfolio_items = len(df)
                average_investment = portfolio_value / portfolio_items if portfolio_items > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
                with col2:
                    st.metric("Portfolio Items", portfolio_items)
                with col3:
                    st.metric("Avg. Investment", f"${average_investment:,.2f}")
                
                # Portfolio composition chart
                st.subheader("Portfolio Composition")
                fig = px.pie(df, values='Quantity', names='Symbol', hole=0.3,
                             color_discrete_sequence=px.colors.sequential.Plasma_r)
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                st.markdown(
                    """
                    <div class="custom-info-box">
                        <p>For portfolio analysis, upload a CSV file with columns: Symbol, Quantity, and Purchase Price.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            # Show sample portfolio data
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Sample Portfolio Overview")
            
            sample_portfolio = {
                'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA'],
                'Quantity': [10, 5, 2, 3, 15, 8, 6, 12],
                'Purchase Price': [150.25, 290.45, 2750.80, 3300.15, 190.75, 310.20, 550.35, 750.90],
                'Current Price': [189.43, 350.12, 2830.45, 3340.20, 245.29, 320.18, 410.75, 850.45],
                'Profit/Loss': ['+$391.80', '+$298.35', '+$159.30', '+$120.15', '+$817.10', '+$79.84', '-$837.60', '+$1194.60']
            }
            sample_df = pd.DataFrame(sample_portfolio)
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
            
            # Portfolio metrics
            col1, col2, col3 = st.columns(3)
            
            total_value = sum(sample_portfolio['Quantity'][i] * sample_portfolio['Current Price'][i] for i in range(len(sample_portfolio['Symbol'])))
            initial_value = sum(sample_portfolio['Quantity'][i] * sample_portfolio['Purchase Price'][i] for i in range(len(sample_portfolio['Symbol'])))
            profit_loss = total_value - initial_value
            profit_loss_pct = (profit_loss / initial_value) * 100 if initial_value > 0 else 0
            
            with col1:
                st.metric("Portfolio Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Profit/Loss", f"${profit_loss:,.2f}", f"{profit_loss_pct:.2f}%")
            with col3:
                st.metric("Number of Assets", len(sample_portfolio['Symbol']))
            
            # Portfolio composition chart
            st.subheader("Portfolio Composition")
            composition_data = pd.DataFrame({
                'Symbol': sample_portfolio['Symbol'],
                'Value': [sample_portfolio['Quantity'][i] * sample_portfolio['Current Price'][i] for i in range(len(sample_portfolio['Symbol']))]
            })
            
            fig = px.pie(composition_data, values='Value', names='Symbol', hole=0.3,
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)',
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Upload instructions
            st.markdown(
                """
                <div class="custom-info-box">
                    <p>This is sample data. Upload your own portfolio for personalized analysis.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    with tab3:
        st.markdown('<h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">Economic Indicators</h2>', unsafe_allow_html=True)
        
        # Economic indicators metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                """
                <div class="metric-container">
                    <div class="metric-value">4.5%</div>
                    <div class="metric-label">Unemployment Rate</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div class="metric-container">
                    <div class="metric-value">3.2%</div>
                    <div class="metric-label">Inflation Rate</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with col3:
            st.markdown(
                """
                <div class="metric-container">
                    <div class="metric-value">5.25%</div>
                    <div class="metric-label">Fed Interest Rate</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with col4:
            st.markdown(
                """
                <div class="metric-container">
                    <div class="metric-value">$24.3T</div>
                    <div class="metric-label">GDP (Annual)</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Economic indicators charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Inflation Rate Trend")
            
            # Generate sample inflation data
            months = pd.date_range(start=datetime.now() - timedelta(days=365*2), end=datetime.now(), freq='M')
            np.random.seed(43)
            inflation_base = 2.0 + np.cumsum(np.random.normal(0, 0.2, len(months))) % 5
            
            # Create DataFrame
            inflation_df = pd.DataFrame({
                'Date': months,
                'Inflation Rate': inflation_base
            })
            
            # Plotly figure
            fig = px.line(inflation_df, x='Date', y='Inflation Rate', markers=True,
                         line_shape='spline', color_discrete_sequence=['#00f2ff'])
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)',
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(title='Inflation Rate (%)', ticksuffix='%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Unemployment Rate vs GDP Growth")
            
            # Generate sample data
            quarters = pd.date_range(start=datetime.now() - timedelta(days=365*5), end=datetime.now(), freq='Q')
            np.random.seed(44)
            
            # Create somewhat realistic data with negative correlation
            unemployment = 5.0 + np.cumsum(np.random.normal(0, 0.3, len(quarters))) % 4 + 3
            base_gdp_growth = 2.5 + np.random.normal(0, 1.0, len(quarters))
            # Add negative correlation
            gdp_growth = base_gdp_growth - (unemployment - unemployment.mean()) * 0.3
            
            # Create DataFrame
            eco_df = pd.DataFrame({
                'Date': quarters,
                'Unemployment': unemployment,
                'GDP Growth': gdp_growth
            })
            
            # Plotly figure with dual y-axis
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=eco_df['Date'],
                y=eco_df['Unemployment'],
                name='Unemployment (%)',
                line=dict(color='#ff00dd', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=eco_df['Date'],
                y=eco_df['GDP Growth'],
                name='GDP Growth (%)',
                line=dict(color='#00f2ff', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)',
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(
                    title='Unemployment Rate (%)',
                    ticksuffix='%',
                    side='left',
                    showgrid=False
                ),
                yaxis2=dict(
                    title='GDP Growth (%)',
                    ticksuffix='%',
                    side='right',
                    overlaying='y',
                    showgrid=False
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Interest rates chart
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Interest Rates - Historical Trend")
        
        # Generate sample interest rate data
        years = pd.date_range(start=datetime.now() - timedelta(days=365*10), end=datetime.now(), freq='M')
        np.random.seed(45)
        
        # Create realistic interest rate patterns
        base_pattern = np.concatenate([
            np.linspace(3, 5, 40),  # Rising
            np.linspace(5, 0.5, 30),  # Falling (financial crisis)
            np.linspace(0.5, 0.25, 20),  # Low rates
            np.linspace(0.25, 0.25, 15),  # Steady low
            np.linspace(0.25, 2.5, 25),  # Rising again
            np.linspace(2.5, 5.25, 20),  # Recent inflation fight
        ])
        
        # Add noise to pattern
        interest_rates = base_pattern[:len(years)] + np.random.normal(0, 0.1, len(years))
        
        # Create DataFrame
        rates_df = pd.DataFrame({
            'Date': years,
            'Fed Funds Rate': interest_rates
        })
        
        # Plotly figure
        fig = px.line(rates_df, x='Date', y='Fed Funds Rate', 
                     line_shape='spline', color_discrete_sequence=['#00f2ff'])
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(17, 17, 17, 0.9)',
            paper_bgcolor='rgba(17, 17, 17, 0)',
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title='Interest Rate (%)', ticksuffix='%')
        )
        
        # Add recession shading (example)
        fig.add_vrect(
            x0="2008-09-01", x1="2009-06-01",
            fillcolor="#ff00dd", opacity=0.2,
            layer="below", line_width=0,
            annotation_text="Recession",
            annotation_position="top left"
        )
        
        fig.add_vrect(
            x0="2020-03-01", x1="2020-07-01",
            fillcolor="#ff00dd", opacity=0.2,
            layer="below", line_width=0,
            annotation_text="COVID",
            annotation_position="top left"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ========================= STOCK ANALYSIS PAGE =========================
elif st.session_state.page == 'stocks':
    st.markdown('<h1 class="main-header">STOCK ANALYSIS</h1>', unsafe_allow_html=True)
    
    # Stock symbol input
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Enter Stock Symbol")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_symbol = st.text_input("Stock Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL")
    with col2:
        period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
        selected_period = st.selectbox("Time Period", list(period_options.keys()))
        period = period_options[selected_period]
    
    def analyze_stock_handler():
        with st.spinner("Fetching stock data..."):
            hist, info = get_stock_data(stock_symbol, period)
            
            if hist is not None and not hist.empty:
                st.session_state.stock_data[stock_symbol] = {'hist': hist, 'info': info}
                st.success(f"Data fetched for {stock_symbol}")
                
                # Add to selected stocks if not already there
                if stock_symbol not in st.session_state.selected_stocks:
                    st.session_state.selected_stocks.append(stock_symbol)
            else:
                st.error(f"Failed to fetch data for {stock_symbol}. Please check the symbol and try again.")
    
    st.button("Analyze Stock", key="analyze_stock", on_click=analyze_stock_handler)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display stock information if available
    if stock_symbol in st.session_state.stock_data:
        hist = st.session_state.stock_data[stock_symbol]['hist']
        info = st.session_state.stock_data[stock_symbol]['info']
        
        # Company info
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Company logo and quick stats
            company_name = info.get('shortName', stock_symbol)
            st.markdown(f"<h2 style='color:#00f2ff; text-shadow: 0 0 5px #00f2ff;'>{company_name}</h2>", unsafe_allow_html=True)
            
            # Current price and change
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2]
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            
            price_color = "#00ff00" if price_change >= 0 else "#ff0000"
            change_symbol = "+" if price_change >= 0 else ""
            
            st.markdown(
                f"""
                <div style="margin-bottom: 20px;">
                    <span style="font-size: 2em; color: {price_color};">${current_price:.2f}</span><br>
                    <span style="color: {price_color};">{change_symbol}{price_change:.2f} ({change_symbol}{price_change_pct:.2f}%)</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Key statistics
            st.markdown("<h4 style='color:#ff00dd;'>Key Statistics</h4>", unsafe_allow_html=True)
            market_cap = info.get('marketCap', 0) / 1_000_000_000  # Convert to billions
            pe_ratio = info.get('trailingPE', 0)
            div_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            
            st.markdown(
                f"""
                <div style="margin-bottom: 5px;"><b>Market Cap:</b> ${market_cap:.2f}B</div>
                <div style="margin-bottom: 5px;"><b>P/E Ratio:</b> {pe_ratio:.2f}</div>
                <div style="margin-bottom: 5px;"><b>Dividend Yield:</b> {div_yield:.2f}%</div>
                <div style="margin-bottom: 5px;"><b>52W Range:</b> ${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}</div>
                <div style="margin-bottom: 5px;"><b>Volume:</b> {info.get('volume', 0):,}</div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            # Stock price chart
            st.subheader("Stock Price History")
            
            # Prepare data for plotting
            price_data = hist.reset_index()
            price_data['MA50'] = price_data['Close'].rolling(window=50).mean()
            price_data['MA200'] = price_data['Close'].rolling(window=200).mean()
            
            # Create figure with candlestick chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=price_data['Date'],
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price',
                increasing_line_color='#00f2ff',
                decreasing_line_color='#ff00dd'
            ))
            
            # Add moving averages
            if len(price_data) >= 50:
                fig.add_trace(go.Scatter(
                    x=price_data['Date'],
                    y=price_data['MA50'],
                    mode='lines',
                    name='50-Day MA',
                    line=dict(color='yellow', width=1)
                ))
            
            if len(price_data) >= 200:
                fig.add_trace(go.Scatter(
                    x=price_data['Date'],
                    y=price_data['MA200'],
                    mode='lines',
                    name='200-Day MA',
                    line=dict(color='white', width=1)
                ))
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)',
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Technical indicators
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Technical Indicators")
        
        # Calculate technical indicators
        price_data = hist.copy()
        
        # RSI (Relative Strength Index)
        delta = price_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        price_data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        price_data['EMA12'] = price_data['Close'].ewm(span=12, adjust=False).mean()
        price_data['EMA26'] = price_data['Close'].ewm(span=26, adjust=False).mean()
        price_data['MACD'] = price_data['EMA12'] - price_data['EMA26']
        price_data['Signal'] = price_data['MACD'].ewm(span=9, adjust=False).mean()
        price_data['MACD_Hist'] = price_data['MACD'] - price_data['Signal']
        
        # Bollinger Bands
        price_data['MA20'] = price_data['Close'].rolling(window=20).mean()
        price_data['SD20'] = price_data['Close'].rolling(window=20).std()
        price_data['Upper_Band'] = price_data['MA20'] + (price_data['SD20'] * 2)
        price_data['Lower_Band'] = price_data['MA20'] - (price_data['SD20'] * 2)
        
        # Create tabs for different indicators
        ind_tab1, ind_tab2, ind_tab3 = st.tabs(["RSI", "MACD", "Bollinger Bands"])
        
        with ind_tab1:
            # RSI Chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#00f2ff', width=2)
            ))
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line", line_color="red", line_width=1, opacity=0.3,
                x0=price_data.index[0], x1=price_data.index[-1], y0=70, y1=70
            )
            fig.add_shape(
                type="line", line_color="green", line_width=1, opacity=0.3,
                x0=price_data.index[0], x1=price_data.index[-1], y0=30, y1=30
            )
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)',
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(
                    title='RSI', 
                    range=[0, 100]
                ),
                annotations=[
                    dict(x=price_data.index[10], y=70, text="Overbought", showarrow=False, font=dict(color="red")),
                    dict(x=price_data.index[10], y=30, text="Oversold", showarrow=False, font=dict(color="green"))
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI interpretation
            latest_rsi = price_data['RSI'].iloc[-1]
            rsi_status = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
            
            st.markdown(
                f"""
                <div class="custom-info-box">
                    <p><b>Current RSI:</b> {latest_rsi:.2f} - {rsi_status}</p>
                    <p>RSI (Relative Strength Index) measures the speed and change of price movements. 
                    Values above 70 indicate overbought conditions (potential sell signal), 
                    while values below 30 indicate oversold conditions (potential buy signal).</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with ind_tab2:
            # MACD Chart
            fig = go.Figure()
            
            # MACD Line
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='#00f2ff', width=2)
            ))
            
            # Signal Line
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#ff00dd', width=2)
            ))
            
            # MACD Histogram
            colors = ['#00ff00' if val >= 0 else '#ff0000' for val in price_data['MACD_Hist']]
            
            fig.add_trace(go.Bar(
                x=price_data.index,
                y=price_data['MACD_Hist'],
                name='Histogram',
                marker_color=colors
            ))
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)',
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(title='MACD')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # MACD interpretation
            latest_macd = price_data['MACD'].iloc[-1]
            latest_signal = price_data['Signal'].iloc[-1]
            latest_hist = price_data['MACD_Hist'].iloc[-1]
            
            if latest_macd > latest_signal and latest_hist > 0:
                macd_signal = "Bullish"
            elif latest_macd < latest_signal and latest_hist < 0:
                macd_signal = "Bearish"
            else:
                macd_signal = "Neutral"
            
            st.markdown(
                f"""
                <div class="custom-info-box">
                    <p><b>MACD:</b> {latest_macd:.4f}</p>
                    <p><b>Signal:</b> {latest_signal:.4f}</p>
                    <p><b>Histogram:</b> {latest_hist:.4f}</p>
                    <p><b>Signal:</b> {macd_signal}</p>
                    <p>MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator. 
                    When MACD crosses above the signal line, it's a bullish signal. 
                    When it crosses below, it's a bearish signal.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with ind_tab3:
            # Bollinger Bands Chart
            fig = go.Figure()
            
            # Price Line
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='white', width=1)
            ))
            
            # Upper Band
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Upper_Band'],
                mode='lines',
                name='Upper Band',
                line=dict(color='#00f2ff', width=1)
            ))
            
            # Middle Band (20-day MA)
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['MA20'],
                mode='lines',
                name='20-day MA',
                line=dict(color='yellow', width=1)
            ))
            
            # Lower Band
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Lower_Band'],
                mode='lines',
                name='Lower Band',
                line=dict(color='#ff00dd', width=1),
                fill='tonexty',
                fillcolor='rgba(0, 242, 255, 0.05)'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)',
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(title='Price')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bollinger Bands interpretation
            latest_close = price_data['Close'].iloc[-1]
            latest_upper = price_data['Upper_Band'].iloc[-1]
            latest_lower = price_data['Lower_Band'].iloc[-1]
            latest_ma20 = price_data['MA20'].iloc[-1]
            
            bb_width = (latest_upper - latest_lower) / latest_ma20
            
            if latest_close > latest_upper:
                bb_signal = "Potentially Overbought"
            elif latest_close < latest_lower:
                bb_signal = "Potentially Oversold"
            else:
                bb_signal = "Within Normal Range"
            
            st.markdown(
                f"""
                <div class="custom-info-box">
                    <p><b>Current Close:</b> ${latest_close:.2f}</p>
                    <p><b>Upper Band:</b> ${latest_upper:.2f}</p>
                    <p><b>20-day MA:</b> ${latest_ma20:.2f}</p>
                    <p><b>Lower Band:</b> ${latest_lower:.2f}</p>
                    <p><b>Bandwidth:</b> {bb_width:.4f}</p>
                    <p><b>Signal:</b> {bb_signal}</p>
                    <p>Bollinger Bands measure volatility. Price reaching the upper band may indicate overbought conditions, 
                    while reaching the lower band may indicate oversold conditions. 
                    The width of the bands indicates volatility.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Volume Analysis
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Volume Analysis")
        
        # Volume Chart
        fig = go.Figure()
        
        # Add volume bars
        colors = ['#00f2ff' if row['Close'] >= row['Open'] else '#ff00dd' for _, row in price_data.iterrows()]
        
        fig.add_trace(go.Bar(
            x=price_data.index,
            y=price_data['Volume'],
            name='Volume',
            marker_color=colors
        ))
        
        # Add 20-day average volume line
        price_data['Volume_MA20'] = price_data['Volume'].rolling(window=20).mean()
        
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Volume_MA20'],
            name='20-day Avg Volume',
            line=dict(color='yellow', width=2)
        ))
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(17, 17, 17, 0.9)',
            paper_bgcolor='rgba(17, 17, 17, 0)',
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title='Volume')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume metrics
        latest_volume = price_data['Volume'].iloc[-1]
        avg_volume = price_data['Volume_MA20'].iloc[-1]
        volume_change = (latest_volume / avg_volume - 1) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Latest Volume", f"{latest_volume:,.0f}")
        
        with col2:
            st.metric("20-day Avg Volume", f"{avg_volume:,.0f}")
        
        with col3:
            st.metric("Volume Change", f"{volume_change:.2f}%", f"{volume_change:.2f}%")
        
        # Volume interpretation
        if latest_volume > avg_volume * 1.5:
            volume_signal = "Significant increase in volume, indicating strong market interest."
        elif latest_volume < avg_volume * 0.5:
            volume_signal = "Significant decrease in volume, indicating lower market interest."
        else:
            volume_signal = "Normal trading volume compared to recent average."
        
        st.markdown(
            f"""
            <div class="custom-info-box">
                <p>{volume_signal}</p>
                <p>High volume confirms price movements and can indicate potential trend reversals or continuations.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown(
            """
            <div class="custom-info-box">
                <p>Enter a stock symbol and click "Analyze Stock" to view detailed stock analysis.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# ========================= ML ANALYTICS PAGE =========================
elif st.session_state.page == 'ml':
    st.markdown('<h1 class="main-header">ML ANALYTICS</h1>', unsafe_allow_html=True)
    
    # Check if data is available
    if st.session_state.data is not None or any(st.session_state.stock_data):
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Machine Learning Model Selection")
        
        # Data source selection
        data_source_options = ["Uploaded Data"]
        for stock in st.session_state.selected_stocks:
            data_source_options.append(f"Stock Data: {stock}")
        
        data_source = st.selectbox("Select Data Source", data_source_options)
        
        # Get the selected data
        if data_source == "Uploaded Data":
            ml_data = st.session_state.data
            data_type = "custom"
        else:
            stock_symbol = data_source.split(": ")[1]
            if stock_symbol in st.session_state.stock_data:
                ml_data = st.session_state.stock_data[stock_symbol]['hist'].copy()
                ml_data = ml_data.reset_index()
                data_type = "stock"
            else:
                ml_data = None
                data_type = None
        
        if ml_data is not None:
            # Display the data
            st.subheader("Data Preview")
            st.dataframe(ml_data.head(), use_container_width=True)
            
            # Model selection
            model_type = st.selectbox(
                "Select ML Model",
                ["Linear Regression", "Logistic Regression", "K-Means Clustering"],
                format_func=lambda x: x
            )
            
            # Display model description
            if model_type == "Linear Regression":
                model_key = "linear"
                st.markdown(
                    """
                    <div class="custom-info-box">
                        <p><b>Linear Regression:</b> Predicts continuous values by finding the best-fitting line through data points. 
                        Useful for forecasting stock prices or returns based on historical data.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif model_type == "Logistic Regression":
                model_key = "logistic"
                st.markdown(
                    """
                    <div class="custom-info-box">
                        <p><b>Logistic Regression:</b> Predicts binary outcomes (e.g., price increase/decrease).
                        Useful for predicting market direction or investment decisions.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:  # K-Means
                model_key = "kmeans"
                st.markdown(
                    """
                    <div class="custom-info-box">
                        <p><b>K-Means Clustering:</b> Groups similar data points together based on their features.
                        Useful for identifying patterns, market regimes, or stock behavior clusters.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Feature and target selection
            if data_type == "stock":
                # For stock data, create some features
                ml_data['Return'] = ml_data['Close'].pct_change()
                ml_data['Return_Lag1'] = ml_data['Return'].shift(1)
                ml_data['Return_Lag2'] = ml_data['Return'].shift(2)
                ml_data['MA5'] = ml_data['Close'].rolling(window=5).mean()
                ml_data['MA20'] = ml_data['Close'].rolling(window=20).mean()
                ml_data['MA5_MA20_Ratio'] = ml_data['MA5'] / ml_data['MA20']
                ml_data['Volume_Change'] = ml_data['Volume'].pct_change()
                ml_data['Price_Range'] = (ml_data['High'] - ml_data['Low']) / ml_data['Open']
                ml_data['Target_Direction'] = (ml_data['Close'].shift(-1) > ml_data['Close']).astype(int)
                ml_data['Target_Return'] = ml_data['Close'].pct_change(periods=5).shift(-5)
                
                # Drop NaN values
                ml_data = ml_data.dropna()
                
                # Default feature and target columns
                if model_key == "linear":
                    default_target = "Target_Return"
                    default_features = ["Return", "Return_Lag1", "MA5_MA20_Ratio", "Volume_Change", "Price_Range"]
                elif model_key == "logistic":
                    default_target = "Target_Direction"
                    default_features = ["Return", "Return_Lag1", "MA5_MA20_Ratio", "Volume_Change", "Price_Range"]
                else:  # kmeans
                    default_target = None  # No target for clustering
                    default_features = ["Return", "Return_Lag1", "Volume_Change", "Price_Range"]
                
            else:  # custom data
                # For custom data, let user select from available columns
                default_target = ml_data.columns[-1]  # Just a guess for default
                default_features = ml_data.columns[:-1]  # All except last column
            
            # Let user select features and target
            if model_key != "kmeans":
                target_col = st.selectbox(
                    "Select Target Column",
                    ml_data.columns.tolist(),
                    index=ml_data.columns.tolist().index(default_target) if default_target in ml_data.columns else 0
                )
            else:
                target_col = None
            
            # Multiselect for features
            feature_cols = st.multiselect(
                "Select Feature Columns",
                [col for col in ml_data.columns if col != target_col] if target_col else ml_data.columns,
                default=[col for col in default_features if col in ml_data.columns][:5]  # Limit to 5 defaults for UI
            )
            
            # Train model button
            def train_model_handler():
                if not feature_cols:
                    st.error("Please select at least one feature column.")
                elif model_key != "kmeans" and not target_col:
                    st.error("Please select a target column.")
                else:
                    with st.spinner("Training model..."):
                        model, predictions, performance, error_msg = create_ml_model(
                            ml_data, model_key, target_col, feature_cols
                        )
                        
                        if error_msg:
                            st.error(error_msg)
                        else:
                            # Store results
                            st.session_state.ml_results[model_key] = {
                                'model': model,
                                'predictions': predictions,
                                'performance': performance,
                                'data': ml_data,
                                'feature_cols': feature_cols,
                                'target_col': target_col,
                                'model_type': model_key
                            }
                            st.success(f"Model trained successfully!")
            
            st.button("Train Model", key="train_ml_model", on_click=train_model_handler)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display model results if available
        if 'ml_results' in st.session_state and st.session_state.ml_results:
            results = st.session_state.ml_results
            
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Model Performance")
            
            if results['model_type'] == 'linear':
                # Linear regression results
                perf = results['performance']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{perf['r2_score']:.4f}")
                with col2:
                    st.metric("Mean Squared Error", f"{perf['mse']:.4f}")
                with col3:
                    st.metric("Root MSE", f"{perf['rmse']:.4f}")
                
                # Feature importance
                st.subheader("Feature Coefficients")
                coef_data = pd.DataFrame({
                    'Feature': list(perf['coefficients'].keys()),
                    'Coefficient': list(perf['coefficients'].values())
                }).sort_values('Coefficient', ascending=False)
                
                fig = px.bar(
                    coef_data, 
                    x='Feature', 
                    y='Coefficient',
                    color='Coefficient',
                    color_continuous_scale=['#ff00dd', '#ffffff', '#00f2ff'],
                    template='plotly_dark'
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Predictions vs Actual
                st.subheader("Predictions vs Actual")
                
                test_data = results['data'].tail(len(results['predictions']))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=test_data[results['target_col']],
                    mode='lines',
                    name='Actual',
                    line=dict(color='#00f2ff', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=results['predictions'],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='#ff00dd', width=2)
                ))
                
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                    yaxis_title=results['target_col']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model insights
                st.subheader("Model Insights")
                
                most_important = coef_data.iloc[0]['Feature']
                least_important = coef_data.iloc[-1]['Feature']
                
                st.markdown(
                    f"""
                    <div class="custom-info-box">
                        <p>This Linear Regression model explains <b>{perf['r2_score']:.2%}</b> of the variance in {results['target_col']}.</p>
                        <p>The most influential feature is <b>{most_important}</b> with a coefficient of <b>{coef_data.iloc[0]['Coefficient']:.4f}</b>.</p>
                        <p>The least influential feature is <b>{least_important}</b> with a coefficient of <b>{coef_data.iloc[-1]['Coefficient']:.4f}</b>.</p>
                        <p>With a Root Mean Squared Error (RMSE) of <b>{perf['rmse']:.4f}</b>, predictions are on average off by this amount.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Potential future prediction (simplified)
                if data_type == "stock" and results['target_col'] == "Target_Return":
                    latest_data = results['data'].iloc[-1:].copy()
                    for col in results['feature_cols']:
                        if col not in latest_data.columns:
                            latest_data[col] = 0  # Default for missing features
                    
                    try:
                        future_return = results['model'].predict(latest_data[results['feature_cols']])[0]
                        latest_close = latest_data['Close'].iloc[0]
                        predicted_price = latest_close * (1 + future_return)
                        
                        st.markdown(
                            f"""
                            <div class="custom-warning-box">
                                <p><b>Forecasted 5-Day Return:</b> {future_return:.2%}</p>
                                <p><b>Current Price:</b> ${latest_close:.2f}</p>
                                <p><b>Forecasted Price (5 days):</b> ${predicted_price:.2f}</p>
                                <p><small>Note: This is a simplified prediction and should not be used as investment advice.</small></p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.warning(f"Could not generate price forecast: {e}")
            
            elif results['model_type'] == 'logistic':
                # Logistic regression results
                perf = results['performance']
                
                st.metric("Accuracy", f"{perf['accuracy']:.4f}")
                
                # Feature importance
                st.subheader("Feature Coefficients")
                coef_data = pd.DataFrame({
                    'Feature': list(perf['coefficients'].keys()),
                    'Coefficient': list(perf['coefficients'].values())
                }).sort_values('Coefficient', ascending=False)
                
                fig = px.bar(
                    coef_data, 
                    x='Feature', 
                    y='Coefficient',
                    color='Coefficient',
                    color_continuous_scale=['#ff00dd', '#ffffff', '#00f2ff'],
                    template='plotly_dark'
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Confusion Matrix (only for binary target)
                if results['data'][results['target_col']].nunique() == 2:
                    st.subheader("Prediction Results")
                    
                    test_data = results['data'].tail(len(results['predictions']))
                    actual = test_data[results['target_col']].values
                    
                    # Calculate confusion matrix
                    tp = np.sum((predictions == 1) & (actual == 1))
                    fp = np.sum((predictions == 1) & (actual == 0))
                    tn = np.sum((predictions == 0) & (actual == 0))
                    fn = np.sum((predictions == 0) & (actual == 1))
                    
                    conf_matrix = np.array([[tn, fp], [fn, tp]])
                    
                    # Plot confusion matrix
                    fig = px.imshow(
                        conf_matrix,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Negative (0)', 'Positive (1)'],
                        y=['Negative (0)', 'Positive (1)'],
                        text_auto=True,
                        color_continuous_scale=['#111111', '#00f2ff']
                    )
                    
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='rgba(17, 17, 17, 0.9)',
                        paper_bgcolor='rgba(17, 17, 17, 0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate additional metrics
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision", f"{precision:.4f}")
                    with col2:
                        st.metric("Recall", f"{recall:.4f}")
                    with col3:
                        st.metric("F1 Score", f"{f1:.4f}")
                
                # Model insights
                st.subheader("Model Insights")
                
                most_important = coef_data.iloc[0]['Feature']
                least_important = coef_data.iloc[-1]['Feature']
                
                if data_type == "stock" and results['target_col'] == "Target_Direction":
                    target_desc = "price increase"
                else:
                    target_desc = "positive outcome"
                
                st.markdown(
                    f"""
                    <div class="custom-info-box">
                        <p>This Logistic Regression model achieves an accuracy of <b>{perf['accuracy']:.2%}</b> in predicting {results['target_col']}.</p>
                        <p>The feature most predictive of a {target_desc} is <b>{most_important}</b> with a coefficient of <b>{coef_data.iloc[0]['Coefficient']:.4f}</b>.</p>
                        <p>The feature least predictive of a {target_desc} is <b>{least_important}</b> with a coefficient of <b>{coef_data.iloc[-1]['Coefficient']:.4f}</b>.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Potential future prediction (simplified)
                if data_type == "stock" and results['target_col'] == "Target_Direction":
                    latest_data = results['data'].iloc[-1:].copy()
                    for col in results['feature_cols']:
                        if col not in latest_data.columns:
                            latest_data[col] = 0  # Default for missing features
                    
                    try:
                        prediction_prob = results['model'].predict_proba(latest_data[results['feature_cols']])[0]
                        up_probability = prediction_prob[1]
                        
                        direction = "UP ↑" if up_probability > 0.5 else "DOWN ↓"
                        confidence = max(up_probability, 1 - up_probability)
                        
                        st.markdown(
                            f"""
                            <div class="custom-warning-box">
                                <p><b>Forecasted Price Direction:</b> {direction}</p>
                                <p><b>Probability of Price Increase:</b> {up_probability:.2%}</p>
                                <p><b>Model Confidence:</b> {confidence:.2%}</p>
                                <p><small>Note: This is a simplified prediction and should not be used as investment advice.</small></p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.warning(f"Could not generate direction forecast: {e}")
            
            else:  # K-means clustering
                # K-means results
                perf = results['performance']
                
                st.metric("Optimal Clusters", perf['optimal_clusters'])
                st.metric("Inertia (Lower is Better)", f"{perf['inertia']:.4f}")
                
                # Visualize clusters in 2D (using first 2 features)
                if len(results['feature_cols']) >= 2:
                    st.subheader("Cluster Visualization")
                    
                    # Get test data
                    test_data = results['data'].copy()
                    test_features = test_data[results['feature_cols']].values
                    
                    # Scale the data
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    test_features_scaled = scaler.fit_transform(test_features)
                    
                    # Add cluster labels to the data
                    test_data['Cluster'] = results['model'].predict(test_features_scaled)
                    
                    # Create a 2D plot of the first two features
                    fig = px.scatter(
                        test_data, 
                        x=results['feature_cols'][0], 
                        y=results['feature_cols'][1],
                        color='Cluster',
                        color_continuous_scale='plasma',
                        template='plotly_dark'
                    )
                    
                    # Add cluster centers
                    centers = scaler.inverse_transform(results['model'].cluster_centers_)
                    centers_df = pd.DataFrame(centers, columns=results['feature_cols'])
                    
                    fig.add_trace(go.Scatter(
                        x=centers_df[results['feature_cols'][0]],
                        y=centers_df[results['feature_cols'][1]],
                        mode='markers',
                        marker=dict(
                            color='white',
                            size=15,
                            line=dict(
                                color='#00f2ff',
                                width=2
                            ),
                            symbol='star'
                        ),
                        name='Cluster Centers'
                    ))
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(17, 17, 17, 0.9)',
                        paper_bgcolor='rgba(17, 17, 17, 0)',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show cluster statistics
                    st.subheader("Cluster Statistics")
                    
                    cluster_stats = test_data.groupby('Cluster')[results['feature_cols']].mean()
                    st.dataframe(cluster_stats)
                    
                    # Additional cluster visualization - parallel coordinates
                    st.subheader("Parallel Coordinates View")
                    
                    fig = px.parallel_coordinates(
                        test_data, 
                        dimensions=results['feature_cols'],
                        color='Cluster',
                        color_continuous_scale='plasma',
                        template='plotly_dark'
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(17, 17, 17, 0.9)',
                        paper_bgcolor='rgba(17, 17, 17, 0)',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster insights
                    st.subheader("Cluster Insights")
                    
                    # Get cluster descriptions
                    cluster_desc = []
                    for i in range(perf['optimal_clusters']):
                        # Find distinguishing features
                        cluster_data = cluster_stats.loc[i]
                        ordered_features = cluster_data.abs().sort_values(ascending=False)
                        top_features = ordered_features.index[:min(3, len(ordered_features))].tolist()
                        
                        # Format feature values
                        feature_desc = []
                        for feat in top_features:
                            val = cluster_data[feat]
                            if abs(val) < 0.01:
                                val_str = f"{val:.4f}"
                            else:
                                val_str = f"{val:.2f}"
                            feature_desc.append(f"{feat}: {val_str}")
                        
                        # Count data points in cluster
                        count = (test_data['Cluster'] == i).sum()
                        percent = count / len(test_data) * 100
                        
                        cluster_desc.append(
                            f"""
                            <div style="margin-bottom: 10px;">
                                <b>Cluster {i} ({count} points, {percent:.1f}%):</b><br>
                                {', '.join(feature_desc)}
                            </div>
                            """
                        )
                    
                    st.markdown(
                        f"""
                        <div class="custom-info-box">
                            <p>The K-Means algorithm identified {perf['optimal_clusters']} distinct clusters in the data:</p>
                            {''.join(cluster_desc)}
                            <p>Clusters can help identify market regimes, trading patterns, or stock behavior groups.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown(
            """
            <div class="custom-info-box">
                <p>Upload a dataset or analyze stock data to use the ML tools.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# ========================= STOCK COMPARISON PAGE =========================
elif st.session_state.page == 'comparison':
    st.markdown('<h1 class="main-header">STOCK COMPARISON</h1>', unsafe_allow_html=True)
    
    if not st.session_state.selected_stocks:
        st.markdown(
            """
            <div class="custom-info-box">
                <p>No stocks selected for comparison. Go to the Stock Analysis page to analyze stocks first.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Select Stocks to Compare")
        
        # Let user select stocks to compare
        selected_for_comparison = st.multiselect(
            "Select stocks to compare",
            st.session_state.selected_stocks,
            default=st.session_state.selected_stocks[:min(3, len(st.session_state.selected_stocks))]
        )
        
        # Time period selection
        period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
        selected_period = st.selectbox("Time Period", list(period_options.keys()))
        period = period_options[selected_period]
        
        # Additional options
        normalize = st.checkbox("Normalize Prices (Set starting point to 100%)", value=True)
        include_volume = st.checkbox("Include Volume Analysis", value=True)
        include_volatility = st.checkbox("Include Volatility Analysis", value=True)
        
        def compare_stocks_handler():
            if len(selected_for_comparison) < 2:
                st.error("Please select at least two stocks to compare.")
            else:
                with st.spinner("Preparing comparison..."):
                    # Prepare data for comparison
                    comparison_data = {}
                    start_date = datetime.now() - timedelta(days=365 if period == "1y" else 730 if period == "2y" else 180 if period == "6mo" else 90 if period == "3mo" else 30)
                    
                    for symbol in selected_for_comparison:
                        if symbol in st.session_state.stock_data:
                            hist = st.session_state.stock_data[symbol]['hist'].copy()
                            
                            if not hist.empty:
                                # Filter by period
                                if isinstance(hist.index[0], pd.Timestamp):
                                    hist = hist[hist.index >= start_date]
                                
                                # Store data
                                comparison_data[symbol] = hist
                    
                    if comparison_data:
                        st.session_state.comparison_data = comparison_data
                        st.success("Comparison data prepared!")
                    else:
                        st.error("No valid data available for the selected stocks.")
        
        st.button("Compare Stocks", key="compare_stocks", on_click=compare_stocks_handler)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display comparison if data is available
        if st.session_state.comparison_data:
            comparison_data = st.session_state.comparison_data
            
            # Price comparison chart
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Price Comparison")
            
            fig = go.Figure()
            
            # Create price chart
            for symbol, hist in comparison_data.items():
                # Get close prices
                close_prices = hist['Close']
                
                # Normalize if requested
                if normalize:
                    close_prices = (close_prices / close_prices.iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=close_prices.index,
                    y=close_prices,
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)',
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(
                    title='Normalized Price (%)' if normalize else 'Price',
                    ticksuffix='%' if normalize else ''
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate performance metrics
            st.subheader("Performance Comparison")
            
            metrics = []
            for symbol, hist in comparison_data.items():
                close_prices = hist['Close']
                
                # Calculate returns for different periods
                daily_returns = close_prices.pct_change().dropna()
                
                # Performance metrics
                total_return = (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
                annual_return = (1 + total_return/100) ** (365 / (close_prices.index[-1] - close_prices.index[0]).days) - 1
                annual_return = annual_return * 100
                
                volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized
                sharpe_ratio = (annual_return / 100) / (volatility / 100) if volatility != 0 else 0
                max_drawdown = (close_prices / close_prices.cummax() - 1).min() * 100
                
                # Positive days percentage
                positive_days = (daily_returns > 0).sum() / len(daily_returns) * 100
                
                # Beta (if comparing against first stock)
                beta = None
                if symbol != list(comparison_data.keys())[0]:
                    benchmark = comparison_data[list(comparison_data.keys())[0]]['Close'].pct_change().dropna()
                    if len(benchmark) == len(daily_returns):
                        covariance = daily_returns.cov(benchmark)
                        variance = benchmark.var()
                        beta = covariance / variance if variance != 0 else None
                
                metrics.append({
                    'Symbol': symbol,
                    'Total Return (%)': f"{total_return:.2f}%",
                    'Annualized Return (%)': f"{annual_return:.2f}%",
                    'Volatility (%)': f"{volatility:.2f}%",
                    'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                    'Max Drawdown (%)': f"{max_drawdown:.2f}%",
                    'Positive Days (%)': f"{positive_days:.2f}%",
                    'Beta': f"{beta:.2f}" if beta is not None else "N/A"
                })
            
            # Display metrics table
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Correlation analysis
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Correlation Analysis")
            
            # Prepare data for correlation
            returns_data = pd.DataFrame()
            
            for symbol, hist in comparison_data.items():
                returns_data[symbol] = hist['Close'].pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_data.corr()
            
            # Plot heatmap
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                color_continuous_scale=['#ff00dd', '#ffffff', '#00f2ff'],
                zmin=-1, zmax=1
            )
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                """
                <div class="custom-info-box">
                    <p>Correlation values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).</p>
                    <p>Values close to 0 indicate little to no correlation between the stocks' price movements.</p>
                    <p>Diversification benefits typically come from including assets with low correlation.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Volume comparison if requested
            if include_volume:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.subheader("Volume Comparison")
                
                fig = go.Figure()
                
                for symbol, hist in comparison_data.items():
                    # Get volume data
                    volume = hist['Volume']
                    
                    # Normalize volume to better compare
                    normalized_volume = (volume / volume.mean()) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=volume.index,
                        y=normalized_volume,
                        mode='lines',
                        name=f"{symbol} Volume",
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(
                        title='Normalized Volume (%)',
                        ticksuffix='%'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume metrics
                volume_metrics = []
                
                for symbol, hist in comparison_data.items():
                    volume = hist['Volume']
                    
                    avg_volume = volume.mean()
                    recent_avg_volume = volume.tail(5).mean()
                    volume_trend = (recent_avg_volume / avg_volume - 1) * 100
                    max_volume = volume.max()
                    max_volume_date = volume.idxmax().strftime('%Y-%m-%d') if isinstance(volume.idxmax(), pd.Timestamp) else str(volume.idxmax())
                    
                    volume_metrics.append({
                        'Symbol': symbol,
                        'Avg. Daily Volume': f"{avg_volume:,.0f}",
                        'Recent Volume Trend': f"{volume_trend:+.2f}%",
                        'Max Volume': f"{max_volume:,.0f}",
                        'Max Volume Date': max_volume_date
                    })
                
                # Display volume metrics
                volume_df = pd.DataFrame(volume_metrics)
                st.dataframe(volume_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Volatility analysis if requested
            if include_volatility:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.subheader("Volatility Comparison")
                
                # Calculate rolling volatility (20-day window)
                fig = go.Figure()
                
                for symbol, hist in comparison_data.items():
                    # Calculate daily returns
                    returns = hist['Close'].pct_change().dropna()
                    
                    # Calculate rolling volatility (annualized)
                    rolling_vol = returns.rolling(window=20).std() * (252 ** 0.5) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol,
                        mode='lines',
                        name=f"{symbol} Volatility",
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(
                        title='Annualized Volatility (%)',
                        ticksuffix='%'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility distribution (histogram)
                st.subheader("Return Distribution")
                
                fig = go.Figure()
                
                for symbol, hist in comparison_data.items():
                    # Calculate daily returns
                    returns = hist['Close'].pct_change().dropna() * 100
                    
                    fig.add_trace(go.Histogram(
                        x=returns,
                        name=symbol,
                        opacity=0.6,
                        nbinsx=30
                    ))
                
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                    barmode='overlay',
                    xaxis=dict(
                        title='Daily Return (%)',
                        ticksuffix='%'
                    ),
                    yaxis=dict(
                        title='Frequency'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk/return scatterplot
                st.subheader("Risk vs. Return")
                
                risk_return_data = []
                
                for symbol, hist in comparison_data.items():
                    # Calculate daily returns
                    returns = hist['Close'].pct_change().dropna()
                    
                    # Calculate annualized return and volatility
                    annual_return = ((1 + returns.mean()) ** 252 - 1) * 100
                    annual_vol = returns.std() * (252 ** 0.5) * 100
                    
                    risk_return_data.append({
                        'Symbol': symbol,
                        'Return (%)': annual_return,
                        'Risk (%)': annual_vol
                    })
                
                risk_return_df = pd.DataFrame(risk_return_data)
                
                fig = px.scatter(
                    risk_return_df,
                    x='Risk (%)',
                    y='Return (%)',
                    text='Symbol',
                    size=[30] * len(risk_return_df),
                    color='Symbol',
                    color_discrete_sequence=px.colors.qualitative.Plasma
                )
                
                fig.update_traces(
                    textposition='top center',
                    marker=dict(line=dict(width=1, color='#ffffff'))
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(
                    """
                    <div class="custom-info-box">
                        <p>The Risk vs. Return chart shows the relationship between annualized volatility (risk) and annualized return.</p>
                        <p>Ideally, you want assets with higher returns and lower risk (top-left quadrant).</p>
                        <p>This analysis can help optimize portfolio allocation based on your risk tolerance.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Portfolio optimization suggestion
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Portfolio Optimization")
            
            st.markdown(
                """
                <div class="custom-info-box">
                    <p>Based on the correlation analysis and risk-return profiles, consider the following strategies:</p>
                    <ul>
                        <li>Combine stocks with lower correlation to improve diversification</li>
                        <li>Allocate more to stocks with better risk-adjusted returns (higher Sharpe ratio)</li>
                        <li>Consider the volatility profiles when determining position sizes</li>
                    </ul>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Portfolio allocation sliders
            st.subheader("Test Portfolio Allocations")
            
            # Create sliders for allocation
            allocations = {}
            for symbol in comparison_data.keys():
                allocations[symbol] = st.slider(f"{symbol} Allocation (%)", 0, 100, 100 // len(comparison_data))
            
            # Normalize allocations to 100%
            total_allocation = sum(allocations.values())
            if total_allocation != 0:
                normalized_allocations = {symbol: alloc / total_allocation for symbol, alloc in allocations.items()}
            else:
                normalized_allocations = {symbol: 0 for symbol in allocations.keys()}
            
            # Display pie chart of allocations
            fig = px.pie(
                values=list(normalized_allocations.values()),
                names=list(normalized_allocations.keys()),
                title="Portfolio Allocation",
                color_discrete_sequence=px.colors.sequential.Plasma_r
            )
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(17, 17, 17, 0.9)',
                paper_bgcolor='rgba(17, 17, 17, 0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate portfolio performance
            if total_allocation > 0:
                portfolio_returns = pd.DataFrame()
                
                for symbol, hist in comparison_data.items():
                    # Calculate daily returns
                    returns = hist['Close'].pct_change().dropna()
                    portfolio_returns[symbol] = returns * normalized_allocations[symbol]
                
                # Sum across rows to get portfolio returns
                portfolio_returns['Portfolio'] = portfolio_returns.sum(axis=1)
                
                # Calculate portfolio performance metrics
                portfolio_return = ((1 + portfolio_returns['Portfolio'].mean()) ** 252 - 1) * 100
                portfolio_vol = portfolio_returns['Portfolio'].std() * (252 ** 0.5) * 100
                portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol != 0 else 0
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Expected Annual Return", f"{portfolio_return:.2f}%")
                
                with col2:
                    st.metric("Expected Annual Volatility", f"{portfolio_vol:.2f}%")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
                
                # Plot cumulative returns
                st.subheader("Hypothetical Portfolio Performance")
                
                # Calculate cumulative returns
                cumulative_returns = (1 + portfolio_returns).cumprod()
                
                fig = go.Figure()
                
                for symbol in comparison_data.keys():
                    fig.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns[symbol],
                        mode='lines',
                        name=symbol,
                        line=dict(width=1, dash='dash')
                    ))
                
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns['Portfolio'],
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#00f2ff', width=3)
                ))
                
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(title='Cumulative Return (1 = Initial Investment)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Run once when starting the app
if 'first_run' not in st.session_state:
    st.session_state.first_run = False
