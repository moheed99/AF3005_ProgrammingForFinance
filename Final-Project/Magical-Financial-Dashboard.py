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
# import matplotlib.pyplot as plt # Not directly used for st.pyplot
import base64
from datetime import datetime, timedelta
import time
try:
    from kneed import KneeLocator # For K-Means elbow detection
except ImportError:
    KneeLocator = None # Handle if kneed is not installed

# Page configuration
st.set_page_config(
    page_title="NeoFinance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling (Your existing CSS is largely good, minor adjustments were made in previous turn) ---
def add_bg_from_url(url):
    # ... (keep your existing add_bg_from_url function)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main .block-container {{
            max-height: 95vh; /* Ensure enough space */
            overflow-y: auto;
            padding-top: 1rem; /* Add some padding at the top */
            padding-bottom: 3rem; /* Add some padding at the bottom */
        }}
        /* Custom scrollbar for main content area */
        .main .block-container::-webkit-scrollbar {{
            width: 8px;
        }}
        .main .block-container::-webkit-scrollbar-track {{
            background: rgba(31, 31, 46, 0.5);
            border-radius: 10px;
        }}
        .main .block-container::-webkit-scrollbar-thumb {{
            background: #00f2ff;
            border-radius: 10px;
        }}
        .main .block-container::-webkit-scrollbar-thumb:hover {{
            background: #ff00dd;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def local_css():
    # ... (keep your existing local_css function, ensuring .stButton > button is styled)
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
        background: rgba(31, 31, 46, 0.85);
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

    .stButton>button {
        background: rgba(17, 17, 17, 0.9);
        color: #00f2ff;
        border: 1px solid #00f2ff;
        border-radius: 5px;
        padding: 8px 18px;
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
        text-shadow: 0 0 3px #00f2ff;
        box-shadow: 0 0 8px #00f2ff;
        width: 100%; 
        margin-bottom: 8px; 
    }

    .stButton>button:hover {
        background: #00f2ff;
        color: #111111;
        box-shadow: 0 0 15px #00f2ff;
    }
     .stButton>button:active { /* NEW: Active state for button */
        transform: scale(0.98);
        box-shadow: 0 0 5px #00f2ff;
    }
    
    .metric-container { /* ... keep ... */ }
    .metric-value { /* ... keep ... */ }
    .metric-label { /* ... keep ... */ }
    @keyframes glow { /* ... keep ... */ }
    .stTabs [data-baseweb="tab-list"] { /* ... keep ... */ }
    .stTabs [data-baseweb="tab"] { /* ... keep ... */ }
    .stTabs [aria-selected="true"] { /* ... keep ... */ }
    .loader { /* ... keep ... */ }
    @keyframes loading { /* ... keep ... */ }
    .stDataFrame { /* ... keep ... */ }
    .custom-info-box {
        background: rgba(0, 242, 255, 0.1);
        border: 1px solid #00f2ff;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
        color: #e0e0e0; 
    }
    .custom-warning-box {
        background: rgba(255, 0, 221, 0.1);
        border: 1px solid #ff00dd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
        color: #e0e0e0; 
    }
    .logo-spin { /* ... keep ... */ }
    @keyframes spin { /* ... keep ... */ }
    .stDateInput > div > div {
        border: 1px solid #00f2ff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 5px #00f2ff !important;
        background-color: rgba(17, 17, 17, 0.9) !important; 
    }
     .stDateInput input { 
        color: #e0e0e0 !important;
        background-color: transparent !important;
    }
    .stSelectbox > div > div {
        border: 1px solid #00f2ff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 5px #00f2ff !important;
        background-color: rgba(17, 17, 17, 0.9) !important;
    }
    .stSelectbox div[data-baseweb="select"] > div { /* NEW: Ensure select text is visible */
        color: #e0e0e0 !important;
    }
    .stTextInput > div > div > input {
        border: 1px solid #00f2ff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 5px #00f2ff !important;
        background-color: rgba(17, 17, 17, 0.9) !important;
        color: #e0e0e0 !important;
    }
    .stMultiSelect > div > div { /* NEW: Style multiselect */
        border: 1px solid #00f2ff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 5px #00f2ff !important;
        background-color: rgba(17, 17, 17, 0.9) !important;
    }
    .stMultiSelect span[data-baseweb="tag"] { /* NEW: Style multiselect tags */
        background-color: #00f2ff !important;
        color: #111 !important;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()
bg_url = "https://images.pexels.com/photos/36717/amazing-animal-beautiful-view.jpg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
add_bg_from_url(bg_url)

# --- Session State Initialization ---
if 'page' not in st.session_state: st.session_state.page = 'welcome'
if 'data' not in st.session_state: st.session_state.data = None
if 'stock_data' not in st.session_state: st.session_state.stock_data = {}
if 'ml_results' not in st.session_state: st.session_state.ml_results = {}
if 'selected_stocks' not in st.session_state: st.session_state.selected_stocks = []
if 'comparison_data' not in st.session_state: st.session_state.comparison_data = None
if 'first_run_animation_shown' not in st.session_state: st.session_state.first_run_animation_shown = False

# --- SVG Logo ---
def get_neon_logo_svg():
    # ... (your existing SVG code) ...
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
        </defs>
        <circle cx="75" cy="75" r="60" fill="none" stroke="#00f2ff" stroke-width="2" filter="url(#neon1)"/>
        <circle cx="75" cy="75" r="45" fill="none" stroke="#00f2ff" stroke-width="3" filter="url(#neon1)"/>
        <path d="M55,55 L95,95 M55,95 L95,55" stroke="#00f2ff" stroke-width="4" filter="url(#neon1)"/>
        <circle cx="75" cy="75" r="30" fill="none" stroke="#ff00dd" stroke-width="2" filter="url(#neon1)"/>
        <text x="75" y="80" text-anchor="middle" fill="#00f2ff" font-family="Arial" font-size="12" filter="url(#neon1)">NEOFIN</text>
    </svg>
    '''
    return svg_code

# --- Navigation ---
def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun() # MODIFIED: Re-added st.rerun() for more forceful page updates, given the user's issue.

with st.sidebar:
    # ... (your sidebar logo and title) ...
    st.markdown(f'<div class="logo-spin" style="text-align:center;">{get_neon_logo_svg()}</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#00f2ff; text-shadow: 0 0 5px #00f2ff; text-align:center;">NeoFinance Analytics</h2>', unsafe_allow_html=True)
    st.markdown('<div style="height: 2px; background: linear-gradient(to right, #00f2ff, #ff00dd); margin-bottom: 20px;"></div>', unsafe_allow_html=True)
    
    if st.button("🏠 Welcome", key="nav_welcome_btn"): navigate_to('welcome')
    if st.button("📊 Dashboard", key="nav_dashboard_btn"): navigate_to('dashboard')
    if st.button("📈 Stock Analysis", key="nav_stocks_btn"): navigate_to('stocks')
    if st.button("🤖 ML Analytics", key="nav_ml_btn"): navigate_to('ml')
    if st.button("📉 Stock Comparison", key="nav_comparison_btn"): navigate_to('comparison')
    
    st.markdown('<div style="height: 2px; background: linear-gradient(to right, #ff00dd, #00f2ff); margin-top: 20px; margin-bottom: 10px;"></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color:#ff00dd; text-shadow: 0 0 3px #ff00dd;">Upload Data</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload financial dataset (CSV or XLSX)", type=["csv", "xlsx"], key="file_uploader_main") # NEW: Added key
    
    if uploaded_file is not None:
        # NEW: Process upload only once per file
        if 'last_uploaded_filename' not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
            try:
                with st.spinner("Processing uploaded file..."):
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                    st.session_state.data = data
                    st.session_state.last_uploaded_filename = uploaded_file.name # Store filename
                st.success("Data uploaded successfully!")
                st.session_state.stock_data = {} # Clear previous stock data
                st.session_state.selected_stocks = []
                st.session_state.comparison_data = None
                st.session_state.ml_results = {} # Clear ML results
                # MODIFIED: Don't navigate immediately, let user choose page. Or navigate to dashboard.
                # navigate_to('dashboard') 
            except Exception as e:
                st.error(f"Error uploading file: {e}")
                st.session_state.data = None
                st.session_state.last_uploaded_filename = None


# --- Utility Functions (Stock Data, ML Model) ---
def show_welcome_animation():
    # ... (your existing animation) ...
    st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
    welcome_messages = [
        "Initializing NeoFinance systems...", "Connecting to quantum finance network...",
        "Calibrating predictive algorithms...", "Establishing secure data channels...",
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
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)
    st.session_state.first_run_animation_shown = True

@st.cache_data(ttl=300) # MODIFIED: Shorter cache for stock data (5 mins)
def get_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            st.warning(f"No historical data found for {ticker} for the period {period}.")
            return None, None
        
        info = stock.info
        # Ensure essential keys exist in info, provide defaults if not
        defaults = {'shortName': ticker, 'marketCap': 0, 'trailingPE': float('nan'), 
                    'dividendYield': 0, 'fiftyTwoWeekLow': float('nan'), 
                    'fiftyTwoWeekHigh': float('nan'), 'volume': 0}
        for key, default_val in defaults.items():
            if info is None or key not in info or info[key] is None: # MODIFIED: Check if info itself is None
                 info = info or {} # Ensure info is a dict
                 info[key] = default_val
        return hist, info
    except Exception as e:
        # st.error(f"Error fetching data for {ticker}: {e}") # MODIFIED: Let calling function handle UI error
        print(f"Error in get_stock_data for {ticker}: {e}") # Log for debugging
        return None, None


def create_ml_model(data_df, model_type, target_col_name, feature_col_names, test_size=0.2):
    # --- Rigorous Data Validation and Preparation ---
    if data_df is None or data_df.empty:
        return None, None, None, "Input data is empty or not provided."
    if not feature_col_names:
        return None, None, None, "No feature columns were selected."
    if model_type != 'kmeans' and (target_col_name is None or target_col_name not in data_df.columns):
        return None, None, None, f"Target column '{target_col_name}' not found or not selected."

    # Create copies to avoid modifying original session state data
    df = data_df.copy()

    # Ensure all feature columns exist
    missing_features = [col for col in feature_col_names if col not in df.columns]
    if missing_features:
        return None, None, None, f"Feature column(s) not found: {', '.join(missing_features)}"

    X = df[feature_col_names].copy()
    y = None
    if model_type != 'kmeans':
        y = df[target_col_name].copy()

    # Handle missing values and ensure numeric types for features
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col].fillna(X[col].median(), inplace=True) # Use median for robustness
            else: # Convert non-numeric to numeric if possible, else drop or use mode
                try:
                    X[col] = pd.to_numeric(X[col])
                    X[col].fillna(X[col].median(), inplace=True)
                except ValueError: # If conversion fails, use a placeholder or drop
                    # For simplicity, let's fill with 0 after attempting conversion,
                    # but more sophisticated handling might be needed (e.g., one-hot encoding for categoricals)
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        elif not pd.api.types.is_numeric_dtype(X[col]): # Already not null, but not numeric
             X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)


    if y is not None:
        if y.isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(y):
                y.fillna(y.median(), inplace=True)
            else: # If target is categorical and has NaNs, fill with mode
                y.fillna(y.mode()[0] if not y.mode().empty else 'Unknown', inplace=True)
        
        if model_type == 'linear' and not pd.api.types.is_numeric_dtype(y):
            return None, None, None, f"Target column '{target_col_name}' must be numeric for Linear Regression."
        
        if model_type == 'logistic': # Ensure y is numeric for logistic
            if not pd.api.types.is_numeric_dtype(y):
                 # Attempt to convert or map categorical to numeric if not already done
                 y = pd.to_numeric(y, errors='coerce').fillna(0) # Fallback
            if y.nunique() > 2:
                median_y = y.median()
                y = (y > median_y).astype(int)
            elif y.nunique() == 1:
                 return None, None, None, "Target variable for Logistic Regression has only one class."
            else:
                y = y.astype(int)


    # --- Model Training ---
    scaler = StandardScaler()
    model, prediction, performance, error_msg = None, None, None, None

    try:
        if model_type == 'kmeans':
            X_scaled = scaler.fit_transform(X)
            optimal_k = 3 # Default
            if len(X_scaled) >= 10 and KneeLocator:
                inertias = []
                K_range = range(1, min(10, len(X_scaled)))
                for k_val in K_range:
                    if k_val == 0: continue
                    kmeans_temp = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                    kmeans_temp.fit(X_scaled)
                    inertias.append(kmeans_temp.inertia_)
                if len(inertias) > 2 :
                    kl = KneeLocator(list(K_range), inertias, curve='convex', direction='decreasing')
                    optimal_k = kl.elbow if kl.elbow else 3
            
            model = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
            labels = model.fit_predict(X_scaled)
            performance = {'inertia': model.inertia_, 'optimal_clusters': optimal_k, 'cluster_centers_scaled': model.cluster_centers_.tolist(), 'labels': labels.tolist()}
            # NEW: Inverse transform cluster centers for interpretability
            performance['cluster_centers_original'] = scaler.inverse_transform(model.cluster_centers_).tolist()

        else: # Linear or Logistic
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=(y if y.nunique() <= 10 and len(y) >= 2*y.nunique() else None) ) # MODIFIED: Stratify for classification if possible
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if model_type == 'linear':
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                prediction = model.predict(X_test_scaled)
                performance = {
                    'r2_score': model.score(X_test_scaled, y_test),
                    'mse': mean_squared_error(y_test, prediction),
                    'rmse': np.sqrt(mean_squared_error(y_test, prediction)),
                    'coefficients': dict(zip(feature_col_names, model.coef_)),
                    'y_test_actual': y_test.tolist() # For plotting
                }
            elif model_type == 'logistic':
                model = LogisticRegression(max_iter=1000, solver='liblinear')
                model.fit(X_train_scaled, y_train)
                prediction = model.predict(X_test_scaled)
                performance = {
                    'accuracy': accuracy_score(y_test, prediction),
                    'coefficients': dict(zip(feature_col_names, model.coef_[0])),
                    'y_test_actual': y_test.tolist() # For confusion matrix
                }
        return model, prediction, performance, error_msg
    except Exception as e:
        return None, None, None, f"Error during model operation: {str(e)}"


# --- Page Rendering Logic ---
current_page = st.session_state.page

if current_page == 'welcome':
    # ... (Welcome page logic - looks largely OK from previous version) ...
    pass # Placeholder, your existing welcome page logic would go here

elif current_page == 'dashboard':
    # ... (Dashboard logic - ensure dynamic indices work and portfolio checks are robust) ...
    pass

elif current_page == 'stocks':
    # ... (Stock analysis logic - ensure stock_symbol and period are handled) ...
    st.markdown('<h1 class="main-header">STOCK ANALYSIS</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Enter Stock Symbol")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_symbol_input = st.text_input("Stock Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL", key="stock_input_stocks_page")
    with col2:
        period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
        selected_period_label = st.selectbox("Time Period", list(period_options.keys()), key="period_select_stocks_page")
        period_input = period_options[selected_period_label]
    
    if st.button("Analyze Stock", key="analyze_stock_btn_stocks_page"):
        if not stock_symbol_input.strip():
            st.error("Please enter a stock symbol.")
        else:
            with st.spinner(f"Fetching stock data for {stock_symbol_input}..."):
                hist_data, info_data = get_stock_data(stock_symbol_input.upper(), period_input) #MODIFIED: Use .upper()
                
                if hist_data is not None and not hist_data.empty:
                    st.session_state.stock_data[stock_symbol_input.upper()] = {'hist': hist_data, 'info': info_data}
                    st.success(f"Data fetched successfully for {stock_symbol_input.upper()}")
                    if stock_symbol_input.upper() not in st.session_state.selected_stocks:
                        st.session_state.selected_stocks.append(stock_symbol_input.upper())
                else:
                    st.error(f"Failed to fetch data for {stock_symbol_input.upper()}. Please check the symbol and period.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Display data for the *last successfully analyzed stock* or the current input if different
    # MODIFIED: Determine which stock to display info for
    display_symbol = None
    if stock_symbol_input.upper() in st.session_state.stock_data:
        display_symbol = stock_symbol_input.upper()
    elif st.session_state.selected_stocks:
        display_symbol = st.session_state.selected_stocks[-1] # Default to last analyzed

    if display_symbol and display_symbol in st.session_state.stock_data:
        hist = st.session_state.stock_data[display_symbol]['hist']
        info = st.session_state.stock_data[display_symbol]['info']
        # ... (rest of your plotting and info display for the selected stock) ...
        # Make sure to use `hist` and `info` safely, checking for None or empty before access.
        # Example check:
        if info and hist is not None and not hist.empty:
             st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
             # Your info and chart display code for the stock
             st.markdown('</div>', unsafe_allow_html=True)
    # ... (your existing "else" for no data analyzed yet) ...

elif current_page == 'ml':
    # ... (ML Analytics logic - needs careful data selection and model running) ...
    st.markdown('<h1 class="main-header">ML ANALYTICS</h1>', unsafe_allow_html=True)
    
    data_available_for_ml = False
    data_source_options = []

    if st.session_state.data is not None and not st.session_state.data.empty:
        data_source_options.append("Uploaded Custom Data")
        data_available_for_ml = True
    
    # Add analyzed stocks to options
    # MODIFIED: Ensure stock_data[stock] and its 'hist' are valid before adding
    for stock_symbol_ml in st.session_state.selected_stocks:
        if stock_symbol_ml in st.session_state.stock_data and \
           st.session_state.stock_data[stock_symbol_ml].get('hist') is not None and \
           not st.session_state.stock_data[stock_symbol_ml]['hist'].empty:
            data_source_options.append(f"Stock Data: {stock_symbol_ml}")
            data_available_for_ml = True

    if not data_available_for_ml:
        st.markdown(
            """
            <div class="custom-warning-box">
                <p>No data available for ML. Please upload a dataset or analyze a stock on the 'Stock Analysis' page first.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Machine Learning Model Configuration")
        
        data_source_ml = st.selectbox("Select Data Source for ML", data_source_options, key="ml_data_source_select")
        
        ml_input_df = None
        if data_source_ml == "Uploaded Custom Data":
            ml_input_df = st.session_state.data
        elif data_source_ml and data_source_ml.startswith("Stock Data:"):
            selected_stock_for_ml = data_source_ml.split(": ")[1]
            if selected_stock_for_ml in st.session_state.stock_data:
                # Prepare stock data for ML (add features)
                temp_hist_df = st.session_state.stock_data[selected_stock_for_ml]['hist'].copy()
                temp_hist_df.reset_index(inplace=True)
                temp_hist_df['Return'] = temp_hist_df['Close'].pct_change()
                temp_hist_df['Return_Lag1'] = temp_hist_df['Return'].shift(1)
                temp_hist_df['Volume_Change'] = temp_hist_df['Volume'].pct_change()
                # Add more features as needed
                ml_input_df = temp_hist_df.dropna() # Drop NaNs created by shifts/pct_change

        if ml_input_df is not None and not ml_input_df.empty:
            st.markdown("#### Data Preview for ML:")
            st.dataframe(ml_input_df.head(), height=200)

            model_type_selection = st.selectbox(
                "Select ML Model",
                ["Linear Regression", "Logistic Regression", "K-Means Clustering"],
                key="ml_model_type_select"
            )
            model_key_map = {
                "Linear Regression": "linear",
                "Logistic Regression": "logistic",
                "K-Means Clustering": "kmeans"
            }
            selected_model_key = model_key_map[model_type_selection]

            all_columns = ml_input_df.columns.tolist()
            # Ensure 'Date' or other non-numeric/ID columns are not default features/targets
            potential_numeric_cols = ml_input_df.select_dtypes(include=np.number).columns.tolist()

            target_col_ml = None
            if selected_model_key != 'kmeans':
                target_col_ml = st.selectbox(
                    "Select Target Column", 
                    [col for col in potential_numeric_cols if col in all_columns], # Only show valid numeric columns
                    index=0 if not potential_numeric_cols else min(len(potential_numeric_cols)-1, 0) , # Handle empty list
                    key="ml_target_select"
                )
            
            feature_cols_ml = st.multiselect(
                "Select Feature Columns",
                [col for col in potential_numeric_cols if col != target_col_ml and col in all_columns],
                default=[col for col in potential_numeric_cols if col != target_col_ml and col in all_columns][:min(3, len(potential_numeric_cols)- (1 if target_col_ml else 0) )], # Sensible default
                key="ml_features_select"
            )

            if st.button("Train Model", key="ml_train_button"):
                if not feature_cols_ml:
                    st.error("Please select at least one feature column.")
                elif selected_model_key != 'kmeans' and not target_col_ml:
                    st.error("Please select a target column for this model type.")
                else:
                    with st.spinner(f"Training {model_type_selection}..."):
                        # Pass copies to avoid modifying the DataFrame in session state directly
                        model, predictions, performance, error_msg = create_ml_model(
                            ml_input_df.copy(), selected_model_key, target_col_ml, feature_cols_ml.copy()
                        )
                        if error_msg:
                            st.error(f"Model Training Error: {error_msg}")
                            st.session_state.ml_results = {} # Clear previous results on error
                        elif model and performance:
                            st.session_state.ml_results = {
                                'model_type_display': model_type_selection, # For display
                                'model_key': selected_model_key, # For logic
                                'model': model,
                                'predictions': predictions,
                                'performance': performance,
                                'data_snapshot': ml_input_df.head(), # Store a snapshot for reference
                                'feature_cols': feature_cols_ml,
                                'target_col': target_col_ml
                            }
                            st.success(f"{model_type_selection} model trained successfully!")
                        else:
                            st.error("Model training failed for an unknown reason.")
                            st.session_state.ml_results = {}
        else:
            st.info("Selected data source is empty or could not be prepared for ML.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Display ML results
        if st.session_state.ml_results:
            results = st.session_state.ml_results
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader(f"Results for: {results.get('model_type_display', 'ML Model')}")
            perf_data = results.get('performance', {})
            
            if results.get('model_key') == 'linear':
                st.metric("R² Score", f"{perf_data.get('r2_score', 'N/A'):.4f}")
                # ... (rest of linear regression display)
            elif results.get('model_key') == 'logistic':
                st.metric("Accuracy", f"{perf_data.get('accuracy', 'N/A'):.4f}")
                # ... (rest of logistic regression display including confusion matrix using y_test_actual)
                # Example for Confusion Matrix if y_test_actual is available
                # actual_labels = perf_data.get('y_test_actual')
                # predicted_labels = results.get('predictions')
                # if actual_labels is not None and predicted_labels is not None:
                # from sklearn.metrics import confusion_matrix
                # cm = confusion_matrix(actual_labels, predicted_labels)
                # fig_cm = px.imshow(cm, text_auto=True, ...)
                # st.plotly_chart(fig_cm)

            elif results.get('model_key') == 'kmeans':
                st.metric("Optimal Clusters", f"{perf_data.get('optimal_clusters', 'N/A')}")
                st.metric("Inertia", f"{perf_data.get('inertia', 'N/A'):.2f}")
                # ... (K-Means visualization using scaled data for plot and inverse_transform for centers)
                # For K-Means plot:
                # data_for_plot = ml_input_df[results['feature_cols']].copy() # Use the original selected features
                # data_for_plot['Cluster'] = perf_data['labels']
                # if len(results['feature_cols']) >= 2:
                #    fig_kmeans = px.scatter(data_for_plot, x=results['feature_cols'][0], y=results['feature_cols'][1], color='Cluster', ...)
                #    # For centers: scaler.inverse_transform(perf_data['cluster_centers_scaled'])
                #    st.plotly_chart(fig_kmeans)

            st.markdown('</div>', unsafe_allow_html=True)


elif current_page == 'comparison':
    # ... (Stock Comparison logic - ensure data for selected stocks is valid) ...
    pass


# NEW: Footer
st.markdown("---")
st.markdown(
    """
    <p style="text-align:center; color: #aaa; font-size: 0.9em;">
        NeoFinance Analytics © 2024. For illustrative and educational purposes only. Not financial advice.
    </p>
    """, unsafe_allow_html=True
)
