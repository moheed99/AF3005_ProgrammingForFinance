import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Already present
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
# import matplotlib.pyplot as plt # MODIFIED: Not directly used for st.pyplot, can be removed if seaborn isn't used elsewhere
import base64 # Already present
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
            background-attachment: fixed; /* NEW: Keeps background fixed on scroll */
        }}
        /* NEW: Ensure content area is scrollable if it overflows */
        .main .block-container {{
            max-height: 90vh; /* Adjust as needed */
            overflow-y: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Futuristic neon styling
def local_css():
    st.markdown("""
    <style>
    /* --- Your existing CSS --- */
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
        background: rgba(31, 31, 46, 0.85); /* MODIFIED: Slightly more opaque for readability */
        border: 1px solid #00f2ff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 15px #00f2ff;
    }
    
    .neon-button { /* This class is used for the JS-triggered button */
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

    /* NEW: Styling for Streamlit's native st.button to match the theme */
    .stButton>button {
        background: rgba(17, 17, 17, 0.9);
        color: #00f2ff;
        border: 1px solid #00f2ff;
        border-radius: 5px;
        padding: 8px 18px; /* Adjusted padding */
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
        text-shadow: 0 0 3px #00f2ff;
        box-shadow: 0 0 8px #00f2ff;
        width: 100%; /* Make sidebar buttons full width */
        margin-bottom: 8px; /* Add some spacing */
    }

    .stButton>button:hover {
        background: #00f2ff;
        color: #111111;
        box-shadow: 0 0 15px #00f2ff;
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
        color: #e0e0e0; /* NEW: Ensure text is visible */
    }
    
    .custom-warning-box {
        background: rgba(255, 0, 221, 0.1);
        border: 1px solid #ff00dd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
        color: #e0e0e0; /* NEW: Ensure text is visible */
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
        background-color: rgba(17, 17, 17, 0.9) !important; /* NEW */
    }
     .stDateInput input { /* NEW */
        color: #e0e0e0 !important;
        background-color: transparent !important;
    }
    
    /* Neon selectbox */
    .stSelectbox > div > div {
        border: 1px solid #00f2ff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 5px #00f2ff !important;
        background-color: rgba(17, 17, 17, 0.9) !important; /* NEW */
    }
    /* NEW: Style for text input to match select/date */
    .stTextInput > div > div > input {
        border: 1px solid #00f2ff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 5px #00f2ff !important;
        background-color: rgba(17, 17, 17, 0.9) !important;
        color: #e0e0e0 !important;
    }

    </style>
    """, unsafe_allow_html=True)

# Apply styling
local_css()

# Background image URL - using the first futuristic background
bg_url = "https://images.pexels.com/photos/36717/amazing-animal-beautiful-view.jpg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" # MODIFIED: Pixabay links can expire or change, using a Pexels one. Replace with your preferred stable URL.
add_bg_from_url(bg_url)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'data' not in st.session_state:
    st.session_state.data = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}
if 'ml_results' not in st.session_state: # MODIFIED: Initialize as an empty dict if not present
    st.session_state.ml_results = {}
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
# NEW: Initialize first_run for welcome animation
if 'first_run_animation_shown' not in st.session_state:
    st.session_state.first_run_animation_shown = False


# SVG logo for futuristic theme
def get_neon_logo_svg():
    # ... (your existing SVG code is fine) ...
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
def navigate_to(page_name): # MODIFIED: Renamed parameter for clarity
    st.session_state.page = page_name
    # st.rerun() # MODIFIED: Removed st.rerun() as it's implicitly handled by Streamlit on widget interaction

# Sidebar navigation
with st.sidebar:
    st.markdown(f'<div class="logo-spin" style="text-align:center;">{get_neon_logo_svg()}</div>', unsafe_allow_html=True) # MODIFIED: Centered logo
    st.markdown('<h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff; text-align:center;">NeoFinance Analytics</h2>', unsafe_allow_html=True) # MODIFIED: Centered title
    st.markdown('<div style="height: 2px; background: linear-gradient(to right, #00f2ff, #ff00dd); margin-bottom: 20px;"></div>', unsafe_allow_html=True) # NEW: Neon divider
    
    # Navigation buttons
    if st.button("🏠 Welcome", key="nav_welcome_btn"): navigate_to('welcome') # MODIFIED: More consistent keys and direct call
    if st.button("📊 Dashboard", key="nav_dashboard_btn"): navigate_to('dashboard')
    if st.button("📈 Stock Analysis", key="nav_stocks_btn"): navigate_to('stocks')
    if st.button("🤖 ML Analytics", key="nav_ml_btn"): navigate_to('ml')
    if st.button("📉 Stock Comparison", key="nav_comparison_btn"): navigate_to('comparison') # MODIFIED: Icon
    
    st.markdown('<div style="height: 2px; background: linear-gradient(to right, #ff00dd, #00f2ff); margin-top: 20px; margin-bottom: 10px;"></div>', unsafe_allow_html=True) # NEW: Neon divider
    
    # Data upload section
    st.markdown('<h3 style="color:#ff00dd; text-shadow: 0 0 3px #ff00dd;">Upload Data</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload financial dataset (CSV or XLSX)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # NEW: Add a spinner for file processing
            with st.spinner("Processing uploaded file..."):
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file) # Requires openpyxl
                st.session_state.data = data
            st.success("Data uploaded successfully!")
            # NEW: Clear stock data if custom data is uploaded to avoid confusion
            st.session_state.stock_data = {}
            st.session_state.selected_stocks = []
            st.session_state.comparison_data = None
        except Exception as e:
            st.error(f"Error uploading file: {e}")
            st.session_state.data = None # NEW: Ensure data is None on error

# Function to display welcome animation
def show_welcome_animation():
    # ... (your existing animation code is good) ...
    st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
    
    welcome_messages = [
        "Initializing NeoFinance systems...",
        "Connecting to quantum finance network...",
        "Calibrating predictive algorithms...",
        "Establishing secure data channels...",
        "Loading futuristic interface..."
    ]
    
    message_placeholder = st.empty()
    for message in welcome_messages:
        message_placeholder.markdown(f"<p style='color:#00f2ff; text-shadow: 0 0 5px #00f2ff; text-align:center;'>{message}</p>", unsafe_allow_html=True)
        time.sleep(0.5)
    
    message_placeholder.empty()
    
    st.markdown(
        """
        <div style="text-align: center; animation: fadeIn 2s;">
            <h1 class="main-header">WELCOME TO NEOFINANCE</h1>
            <p class="sub-header">The Future of Financial Analytics</p>
        </div>
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); } /* MODIFIED: Added translateY for smoother entry */
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.session_state.first_run_animation_shown = True # NEW: Mark animation as shown


# Function to get stock data
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            # NEW: Explicit warning if no history
            st.warning(f"No historical data found for {ticker} for the period {period}.")
            return None, None
        info = stock.info
        # NEW: Add some error handling for info dictionary keys
        required_keys = ['shortName', 'marketCap', 'trailingPE', 'dividendYield', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'volume']
        for key in required_keys:
            if key not in info:
                info[key] = 'N/A' # Provide a default
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None

# Function to create ML model
def create_ml_model(data, model_type, target_col, feature_cols, test_size=0.2):
    if data is None or data.empty: # NEW: Check if data is empty
        return None, None, None, "Data is not available for modeling."
    if not feature_cols: # NEW: Check if feature_cols is empty
        return None, None, None, "No feature columns selected."
    if model_type != 'kmeans' and (target_col is None or target_col not in data.columns): # NEW: Check target_col for relevant models
        return None, None, None, "Target column not selected or not found in data."

    # Ensure all feature columns exist
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        return None, None, None, f"Feature column(s) not found: {', '.join(missing_features)}"
    
    # Prepare data
    X_df = data[feature_cols].copy() # Use .copy() to avoid SettingWithCopyWarning
    if model_type != 'kmeans':
        y_series = data[target_col].copy()

    # Handle missing values robustly
    for col in X_df.columns: # NEW: Iterate through columns for specific handling
        if X_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = X_df[col].fillna(X_df[col].mean())
            else: # For non-numeric, fill with mode or a placeholder
                X_df[col] = X_df[col].fillna(X_df[col].mode()[0] if not X_df[col].mode().empty else 'Unknown')
    
    if model_type != 'kmeans' and y_series.isnull().any():
        if pd.api.types.is_numeric_dtype(y_series):
            y_series = y_series.fillna(y_series.mean())
        else:
            y_series = y_series.fillna(y_series.mode()[0] if not y_series.mode().empty else 'Unknown')

    # Convert all feature columns to numeric, coercing errors
    for col in X_df.columns: # NEW
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0) # Coerce and fill NaNs resulting from coercion

    if model_type != 'kmeans': # NEW
        if not pd.api.types.is_numeric_dtype(y_series) and model_type == 'linear':
            return None, None, None, f"Target column '{target_col}' must be numeric for Linear Regression."
        if model_type == 'logistic': # For logistic regression, ensure target is numeric (0 or 1)
             y_series = pd.to_numeric(y_series, errors='coerce').fillna(0)


    # Split data
    if model_type != 'kmeans':
        X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=test_size, random_state=42)
    else: # For KMeans, use all data for training (as it's unsupervised)
        X_train, X_test = X_df, X_df # Or just X_train = X_df and predict on X_df
        y_train, y_test = None, None # No y for KMeans

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = None
    performance = None
    prediction = None
    error_msg = None
    
    try:
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train_scaled, y_train) # MODIFIED: Use scaled data for consistency, though not strictly necessary for LR
            prediction = model.predict(X_test_scaled)
            performance = {
                'r2_score': model.score(X_test_scaled, y_test),
                'mse': mean_squared_error(y_test, prediction),
                'rmse': np.sqrt(mean_squared_error(y_test, prediction)),
                'coefficients': dict(zip(feature_cols, model.coef_))
            }
            
        elif model_type == 'logistic':
            # Ensure target is binary for logistic regression
            if y_train.nunique() > 2: # MODIFIED: Check unique values on y_train
                # Convert to binary if not already (e.g., based on median)
                median_val_train = y_train.median()
                y_train_binary = (y_train > median_val_train).astype(int)
                y_test_binary = (y_test > median_val_train).astype(int) # Use train median for test set
            elif y_train.nunique() == 1: # NEW: Handle case where target has only one class after split
                return None, None, None, "Target variable has only one class after splitting. Logistic Regression cannot be trained."
            else:
                y_train_binary = y_train.astype(int)
                y_test_binary = y_test.astype(int)

            model = LogisticRegression(max_iter=1000, solver='liblinear') # MODIFIED: Added solver for robustness
            model.fit(X_train_scaled, y_train_binary)
            prediction = model.predict(X_test_scaled)
            performance = {
                'accuracy': accuracy_score(y_test_binary, prediction),
                'coefficients': dict(zip(feature_cols, model.coef_[0]))
            }
            # NEW: Add y_test_binary to results for confusion matrix
            performance['y_test_actual'] = y_test_binary 
            
        elif model_type == 'kmeans':
            # Determine optimal number of clusters using elbow method (simplified)
            # ... (your existing elbow method logic is a good start for a demo) ...
            # NEW: Added n_init='auto' to suppress future warnings
            optimal_k = 3 # Default if elbow method is too simple or fails
            if len(X_train_scaled) >= 10: # Ensure enough samples for elbow
                inertias = []
                K_range = range(1, min(10, len(X_train_scaled))) # Max 10 clusters or num_samples
                for k_val in K_range:
                    if k_val == 0: continue # NEW: Skip k=0
                    kmeans_temp = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                    kmeans_temp.fit(X_train_scaled)
                    inertias.append(kmeans_temp.inertia_)
                
                # Simple elbow point detection (can be improved with kneed library)
                if len(inertias) > 2:
                    # Find the point with the largest decrease in inertia
                    # This is a heuristic and might not always be perfect
                    try:
                        from kneed import KneeLocator # NEW: Try to use kneed for better elbow detection
                        kl = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
                        optimal_k = kl.elbow if kl.elbow else 3
                    except ImportError:
                        # Fallback to simpler heuristic if kneed is not installed
                        decreases = np.diff(inertias, 2) # Second derivative
                        if len(decreases) > 0:
                           optimal_k = np.argmax(decreases) + 2 # +1 for diff, +1 for 0-index
                        else:
                           optimal_k = 3 
                        optimal_k = max(2, optimal_k) # Ensure at least 2 clusters
                else:
                    optimal_k = max(2, len(K_range)) if K_range else 2

            model = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
            prediction = model.fit_predict(X_train_scaled) # Fit and predict on the training data for cluster assignment
            performance = {
                'inertia': model.inertia_,
                'optimal_clusters': optimal_k,
                'cluster_centers': model.cluster_centers_.tolist(),
                'labels': prediction.tolist() # NEW: Store cluster labels
            }
        
        return model, prediction, performance, error_msg
    
    except Exception as e:
        error_msg = f"Error during model training: {str(e)}"
        st.error(error_msg) # NEW: Show error immediately
        return None, None, None, error_msg


# --- Page Rendering Logic ---

# ========================= WELCOME PAGE =========================
if st.session_state.page == 'welcome':
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        # NEW: Show animation only on first visit to welcome page per session
        if not st.session_state.first_run_animation_shown:
            show_welcome_animation()
        else:
             st.markdown( # Show static welcome if animation already shown
                """
                <div style="text-align: center;">
                    <h1 class="main-header">WELCOME BACK TO NEOFINANCE</h1>
                    <p class="sub-header">The Future of Financial Analytics</p>
                </div>
                """, unsafe_allow_html=True)

        # ... (rest of your welcome page content is good) ...
        st.markdown(
            """
            <div class="dashboard-card">
                <h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">About NeoFinance Analytics</h2>
                <p style="color:#e0e0e0;">Welcome to the future of financial analysis. NeoFinance combines cutting-edge machine learning with real-time market data to bring you insights that were once thought impossible.</p>
                <p style="color:#e0e0e0;">This advanced platform leverages the power of artificial intelligence to predict market trends, identify investment opportunities, and optimize your financial strategies.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Feature overview
        st.markdown(
            """
            <div class="dashboard-card">
                <h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">Key Features</h2>
                <ul style="color:#e0e0e0;">
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
        st.markdown( # MODIFIED: Using st.button for consistent styling and direct navigation
            """
            <div style="text-align: center; margin-top: 30px;">
            """
            , unsafe_allow_html=True)
        if st.button("🚀 Launch Dashboard", key="launch_dashboard_welcome_btn"):
            navigate_to('dashboard')
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Images for visual effect
        cols_img = st.columns(3) # MODIFIED: Renamed variable
        with cols_img[0]:
            st.markdown(
                f"""
                <div style="border: 1px solid #00f2ff; border-radius: 10px; overflow: hidden; box-shadow: 0 0 15px #00f2ff; margin-top:20px;">
                    <img src="https://images.pexels.com/photos/187041/pexels-photo-187041.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" style="width: 100%; height: auto;">
                </div>
                """, 
                unsafe_allow_html=True
            )
        with cols_img[1]:
            st.markdown(
                f"""
                <div style="border: 1px solid #ff00dd; border-radius: 10px; overflow: hidden; box-shadow: 0 0 15px #ff00dd; margin-top: 20px;">
                    <img src="https://images.pexels.com/photos/730547/pexels-photo-730547.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" style="width: 100%; height: auto;">
                </div>
                """, 
                unsafe_allow_html=True
            )
        with cols_img[2]:
            st.markdown(
                f"""
                <div style="border: 1px solid #00f2ff; border-radius: 10px; overflow: hidden; box-shadow: 0 0 15px #00f2ff; margin-top:20px;">
                    <img src="https://images.pexels.com/photos/590016/pexels-photo-590016.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" style="width: 100%; height: auto;">
                </div>
                """, 
                unsafe_allow_html=True
            )

# ========================= DASHBOARD PAGE =========================
elif st.session_state.page == 'dashboard':
    st.markdown('<h1 class="main-header">NEOFINANCE DASHBOARD</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Market Overview", "Portfolio Analysis", "Economic Indicators"])
    
    with tab1:
        # ... (Your Market Overview - consider making indices dynamic with yfinance if desired) ...
        # NEW: Fetch and display dynamic major indices data
        st.markdown('<h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">Market Overview</h2>', unsafe_allow_html=True)
        
        @st.cache_data(ttl=300) # Cache for 5 minutes
        def get_dynamic_indices_data():
            indices = {
                "DJIA": "^DJI", 
                "S&P 500": "^GSPC", 
                "NASDAQ": "^IXIC",
                "Russell 2000": "^RUT" # NEW: Added Russell
            }
            indices_data = {}
            for name, ticker in indices.items():
                try:
                    data = yf.Ticker(ticker).history(period="5d") # Get recent data
                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                        indices_data[name] = {
                            "price": f"{current_price:,.2f}",
                            "change": f"{change:+.2f}",
                            "change_pct": f"({change_pct:+.2f}%)",
                            "color": "#00ff00" if change >=0 else "#ff0000"
                        }
                    else:
                        indices_data[name] = {"price": "N/A", "change": "", "change_pct": "", "color":"#ffffff"}
                except Exception:
                    indices_data[name] = {"price": "Error", "change": "", "change_pct": "", "color":"#ffffff"}
            return indices_data

        dynamic_indices = get_dynamic_indices_data()
        cols_metrics = st.columns(len(dynamic_indices))
        for i, (name, data) in enumerate(dynamic_indices.items()):
            with cols_metrics[i]:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-value" style="color:{data['color']};">{data['price']}</div>
                        <div class="metric-label" style="color:{data['color']};">{data['change']} {data['change_pct']}</div>
                        <div class="metric-label">{name}</div>
                    </div>
                    """, unsafe_allow_html=True
                )
        # ... (Rest of your Market Overview with sample data or yfinance integration) ...
        # For the chart, you could also fetch historical data for these indices via yfinance
        # For simplicity, your existing random data chart can stay as a placeholder or be replaced.

    with tab2:
        # ... (Your Portfolio Analysis - good checks for uploaded data) ...
        st.markdown('<h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">Portfolio Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df_portfolio = st.session_state.data # MODIFIED: Use a different variable name
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Uploaded Portfolio Overview")
            st.dataframe(df_portfolio.head(10), use_container_width=True)
            
            # NEW: Check for necessary columns before attempting calculations
            required_cols = ['Symbol', 'Quantity', 'Purchase Price'] # Adjust if your column names differ
            if all(col in df_portfolio.columns for col in required_cols):
                # ... (your existing portfolio metric calculations) ...
                 # Calculate portfolio metrics
                df_portfolio['Current Value'] = df_portfolio['Quantity'] * df_portfolio['Purchase Price'] # Assuming Purchase Price is current if no other current price given
                portfolio_value = df_portfolio['Current Value'].sum()
                portfolio_items = len(df_portfolio)
                average_investment = portfolio_value / portfolio_items if portfolio_items > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
                with col2:
                    st.metric("Portfolio Items", portfolio_items)
                with col3:
                    st.metric("Avg. Investment Value", f"${average_investment:,.2f}")
                
                # Portfolio composition chart
                st.subheader("Portfolio Composition (by Value)")
                fig = px.pie(df_portfolio, values='Current Value', names='Symbol', hole=0.3,
                             color_discrete_sequence=px.colors.sequential.Plasma_r)
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(17, 17, 17, 0.9)',
                    paper_bgcolor='rgba(17, 17, 17, 0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div class="custom-warning-box">
                        <p>For full portfolio analysis, your uploaded data needs columns: {', '.join(required_cols)}.</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # ... (your sample portfolio data is fine as a placeholder) ...
            st.markdown(
                """
                <div class="custom-info-box">
                    <p>Upload your portfolio (CSV/XLSX) via the sidebar for personalized analysis. Ensure columns like 'Symbol', 'Quantity', 'Purchase Price' are present.</p>
                </div>
                """, unsafe_allow_html=True)


    with tab3:
        # ... (Your Economic Indicators - good use of sample data for illustration) ...
        st.markdown('<h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">Economic Indicators</h2>', unsafe_allow_html=True)
        st.markdown( # NEW: Add a note that this is sample data
            """
            <div class="custom-info-box">
                <p><strong>Note:</strong> The economic indicators shown below are based on sample data for illustrative purposes.</p>
            </div>
            """, unsafe_allow_html=True)
        # ... (rest of your economic indicators section) ...


# ========================= STOCK ANALYSIS PAGE =========================
elif st.session_state.page == 'stocks':
    # ... (Your Stock Analysis - good structure) ...
    # NEW: Add a small note about data source
    st.markdown(
        """
        <div class="custom-info-box">
            <p>Live stock data is fetched from Yahoo Finance. Data might have a slight delay.</p>
        </div>
        """, unsafe_allow_html=True)
    # ... (rest of your stock analysis, ensure error handling from get_stock_data is reflected in UI) ...


# ========================= ML ANALYTICS PAGE =========================
elif st.session_state.page == 'ml':
    # ... (Your ML Analytics - good start) ...
    # NEW: Clearer instructions if no data
    if st.session_state.data is None and not any(st.session_state.stock_data):
        st.markdown(
            """
            <div class="custom-warning-box">
                <p>Please upload a dataset or analyze a stock first to use the ML tools.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # ... (rest of your ML page)
        # In create_ml_model, consider adding n_init='auto' to KMeans to avoid future warnings.
        # For K-Means visualization, ensure feature_cols has at least 2 features before trying to plot.
        # The K-Means optimal_k finding is simplified; you could add a note or suggest using a library like `kneed` if more accuracy is needed.
        pass


# ========================= STOCK COMPARISON PAGE =========================
elif st.session_state.page == 'comparison':
    # ... (Your Stock Comparison - good features) ...
    # NEW: Better handling if selected_stocks is empty or has less than 2 stocks
    if not st.session_state.selected_stocks or len(st.session_state.selected_stocks) < 1: # Allow 1 for initial selection
        st.markdown(
            """
            <div class="custom-info-box">
                <p>No stocks analyzed yet, or less than one stock selected. Please go to the 'Stock Analysis' page to analyze stocks first. You'll be able to select them for comparison here.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # ... (rest of your comparison page) ...
        # Ensure that in `compare_stocks_handler`, if `yf.download` fails for a stock, it's handled gracefully (e.g., skipped or user notified).
        # When calculating Beta, ensure data alignment if dates don't perfectly match.
        pass

# NEW: Footer (optional, but common)
st.markdown("---")
st.markdown(
    """
    <p style="text-align:center; color: #aaa; font-size: 0.9em;">
        NeoFinance Analytics © 2024. All rights reserved. Data provided for informational purposes only.
    </p>
    """, unsafe_allow_html=True
)

# Run once when starting the app (This was at the end, it's fine but often placed near session state init)
# if 'first_run' not in st.session_state: # This seems to be for something else not fully implemented
#     st.session_state.first_run = False
