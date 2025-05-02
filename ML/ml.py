import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries # Added Alpha Vantage import
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA
import base64
from datetime import datetime, timedelta
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Financial ML Application",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def add_custom_styling():
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #1e3d59;
        color: white;
    }
    .stButton>button {
        background-color: #ff6e40;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 15px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .step-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    /* Specific style for API key input */
    .stTextInput [type="password"] {
        -webkit-text-security: disc; /* Mask input */
        color: #eee; /* Lighter text color when masked */
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_styling()

# --- Initialize Session State ---
default_session_state = {
    'stage': 0,
    'data': None,
    'features': None,
    'target': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'X_train_scaled': None, # Scaled training data
    'X_test_scaled': None,  # Scaled test data (for Lin/Log Reg)
    'X_scaled': None,       # Scaled data for clustering
    'scaler': None,         # Store the scaler object
    'model': None,
    'model_type': None,
    'predictions': None,
    'probabilities': None,
    'metrics': {},
    'feature_importance': None,
    'processed_data': None,
    'processing_stats': {},
    'clusters': None,
    'data_source': "Upload Dataset", # Default changed
    'split_done': False,
    'training_done': False,
    'evaluation_done': False,
    'api_key': None # Store API key in session state if needed later
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions ---

# Function to reset the application state
def reset_app():
    # Keep API key if entered previously, reset others
    api_key = st.session_state.get('api_key')
    for key in list(st.session_state.keys()):
        st.session_state[key] = default_session_state.get(key) # Reset to default
    st.session_state['stage'] = 0
    st.session_state['api_key'] = api_key # Restore API key
    st.session_state['data_source'] = "Upload Dataset" # Ensure default is reset


# Function to display GIFs
def display_gif(gif_url, width=None):
    if width:
        st.markdown(f'<img src="{gif_url}" width="{width}px" style="display: block; margin-left: auto; margin-right: auto;">', unsafe_allow_html=True)
    else:
        st.markdown(f'<img src="{gif_url}" style="display: block; margin-left: auto; margin-right: auto;">', unsafe_allow_html=True)

# Function to get stock data from Alpha Vantage
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_alphavantage_data(ticker, api_key, output_size='compact'):
    """
    Fetches daily adjusted stock data from Alpha Vantage.

    Args:
        ticker (str): The stock symbol (e.g., "IBM").
        api_key (str): Your Alpha Vantage API key.
        output_size (str): 'compact' (last 100) or 'full' (all history).

    Returns:
        tuple: (pandas.DataFrame or None, str) - DataFrame on success, None on failure, and a status message.
    """
    if not api_key:
        return None, "Error: Alpha Vantage API Key is required."
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize=output_size)

        if data.empty:
             return None, f"No data found for ticker {ticker}. Check the symbol or API key."

        # --- Standardize DataFrame ---
        # Rename columns (Alpha Vantage format: '1. open', '2. high', ...)
        rename_map = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Unadjusted Close', # Keep unadjusted close if needed, or remove
            '5. adjusted close': 'Close', # Use Adjusted Close as 'Close'
            '6. volume': 'Volume'
            # Add others if needed: '7. dividend amount', '8. split coefficient'
        }
        data.rename(columns=rename_map, inplace=True)

        # Select only the desired standard columns
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[col for col in standard_cols if col in data.columns]] # Keep only available ones

        # Convert index to datetime and make it a column 'Date'
        data.index = pd.to_datetime(data.index)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Date'}, inplace=True)

        # Sort by date ascending (Alpha Vantage usually returns descending)
        data.sort_values(by='Date', inplace=True)

        return data, f"Successfully fetched {output_size} data for {ticker} from Alpha Vantage."

    except ValueError as ve:
         # Often indicates invalid API key or symbol format
         return None, f"Error fetching data for {ticker} from Alpha Vantage: {ve}. Check API key/symbol."
    except Exception as e:
        # Catch other potential errors (network, rate limits etc.)
        return None, f"Error fetching data for {ticker} from Alpha Vantage: {e}"


# Function to create a download link for dataframe
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# --- Sidebar ---
st.sidebar.title("Financial ML Application")
st.sidebar.markdown("---")

# Navigation
if st.session_state['stage'] >= 1:
    st.sidebar.subheader("Navigation")
    stages = ["1. Load Data", "2. Preprocessing", "3. Feature Engineering",
              "4. Train/Test Split", "5. Model Training", "6. Evaluation", "7. Results Visualization"]
    current_stage_index = min(st.session_state['stage'] - 1, len(stages) - 1)

    def update_stage_nav():
        st.session_state['stage'] = stages.index(st.session_state.selectbox_stage) + 1

    st.sidebar.selectbox(
        "Go to stage:",
        stages,
        index=current_stage_index,
        key='selectbox_stage',
        on_change=update_stage_nav
    )

st.sidebar.markdown("---")

# Data Source Selection
st.sidebar.subheader("Data Source")

# Make radio button update session state immediately
def update_data_source():
    st.session_state['data_source'] = st.session_state.data_source_radio

data_source_option = st.sidebar.radio(
    "Choose data source:",
    ["Upload Dataset", "Fetch from Alpha Vantage"],
    key='data_source_radio',
    index=0 if st.session_state['data_source'] == "Upload Dataset" else 1,
    on_change=update_data_source
)

# Store the selection explicitly
st.session_state['data_source'] = data_source_option


# Alpha Vantage Options
if data_source_option == "Fetch from Alpha Vantage":
    av_api_key = st.sidebar.text_input(
        "Enter Alpha Vantage API Key:",
        type="password", # Mask the input
        value=st.session_state.get('api_key', '7N101LKDQGERC9HG'), # Persist key within session
        help="Get your free key from alphavantage.co"
    )
    stock_symbol_av = st.sidebar.text_input("Enter stock symbol (e.g., IBM, MSFT):", "IBM")
    output_size_av = st.sidebar.selectbox("Select data size:", ["compact", "full"], index=0) # Compact: ~100 points, Full: All history

    if st.sidebar.button("Fetch Alpha Vantage Data"):
        if not av_api_key:
            st.sidebar.error("Please enter your Alpha Vantage API Key.")
        elif not stock_symbol_av:
            st.sidebar.error("Please enter a stock symbol.")
        else:
            st.session_state['api_key'] = av_api_key # Store the key
            with st.spinner(f"Fetching data for {stock_symbol_av}..."):
                data, message = get_alphavantage_data(stock_symbol_av, av_api_key, output_size_av)
                if data is not None:
                    st.session_state['data'] = data
                    st.session_state['stage'] = 1 # Move to data loading stage
                    st.session_state['processed_data'] = None # Reset processed data
                    st.sidebar.success(message)
                    st.experimental_rerun() # Rerun to reflect the new stage and data
                else:
                    st.sidebar.error(message)

# Upload Dataset Option
if data_source_option == "Upload Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
    if uploaded_file is not None:
        # Load only if the user just uploaded a file or if data is not already loaded
        # This prevents reloading if user switches back and forth in radio buttons
        if st.session_state['data'] is None or getattr(uploaded_file, '_file_id', None) != st.session_state.get('_last_uploaded_file_id'):
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state['data'] = data
                st.session_state['stage'] = 1 # Move to data loading stage
                st.session_state['processed_data'] = None # Reset processed data
                st.session_state['_last_uploaded_file_id'] = getattr(uploaded_file, '_file_id', None) # Track file ID
                st.sidebar.success(f"Successfully loaded {uploaded_file.name}")
                st.experimental_rerun() # Rerun to reflect the new stage and data
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")

st.sidebar.markdown("---")

# Model Selection
if st.session_state['stage'] >= 1:
    st.sidebar.subheader("Model Selection")
    model_options = ["Linear Regression", "Logistic Regression", "K-Means Clustering"]

    def update_model_type():
        # Reset downstream states if model type changes
        if st.session_state['model_type'] != st.session_state.selectbox_model:
            st.session_state['target'] = None
            st.session_state['features'] = None
            st.session_state['model'] = None
            st.session_state['X_train'] = None # etc.
            st.session_state['split_done'] = False
            st.session_state['training_done'] = False
            st.session_state['evaluation_done'] = False
        st.session_state['model_type'] = st.session_state.selectbox_model


    st.sidebar.selectbox(
        "Choose a model:",
        model_options,
        key='selectbox_model',
        index=model_options.index(st.session_state['model_type']) if st.session_state['model_type'] else 0,
        on_change=update_model_type
    )
    if st.session_state['model_type'] is None:
        st.session_state['model_type'] = model_options[0] # Default selection


# Reset Button
st.sidebar.markdown("---")
if st.sidebar.button("Reset Application"):
    reset_app()
    st.experimental_rerun()

# --- Main Application Logic ---

# Welcome Page
if st.session_state['stage'] == 0:
    st.title("Welcome to the Financial ML Application! 📈")
    finance_gif_url = "https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif" # Keep or find another relevant one
    display_gif(finance_gif_url, width=400)

    st.markdown("""
    ## About This Application
    This interactive machine learning application allows you to:

    *   **Load Data:** Upload financial datasets or fetch real-time stock market data from **Alpha Vantage**.
    *   **Preprocess:** Handle missing values and detect/treat outliers.
    *   **Feature Engineer:** Create new features from dates, technical indicators, or interactions.
    *   **Split Data:** Divide your data into training and testing sets.
    *   **Train Models:** Apply Linear Regression, Logistic Regression, or K-Means Clustering (with proper scaling).
    *   **Evaluate:** Assess model performance using relevant metrics.
    *   **Visualize:** Interpret results through various plots and charts.

    ### Get Started
    Select a data source from the sidebar (Upload or Alpha Vantage) and provide necessary inputs (like API key for Alpha Vantage) to begin your analysis.
    """)
    st.info("ℹ️ Please select a data source and load data using the sidebar to continue.")

# 1. Load Data
elif st.session_state['stage'] == 1:
    st.title("1️⃣ Load Data")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state['data'] is not None:
            df_display = st.session_state['data'].copy()

            # Attempt to convert date columns for display and plotting
            potential_date_cols = [col for col in df_display.columns if 'date' in col.lower() or 'time' in col.lower()]
            main_date_col = None
            if potential_date_cols:
                main_date_col = potential_date_cols[0] # Usually 'Date' after standardization
                try:
                    df_display[main_date_col] = pd.to_datetime(df_display[main_date_col])
                except Exception as e:
                    st.warning(f"Could not convert column '{main_date_col}' to datetime: {e}. Plotting might be affected.")
                    main_date_col = None # Prevent using it if conversion failed

            # Data Preview
            st.subheader("Data Preview")
            st.dataframe(df_display.head())

            # Data Information
            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Number of rows:** {df_display.shape[0]}")
                st.write(f"**Number of columns:** {df_display.shape[1]}")
            with col2:
                buffer = io.StringIO()
                df_display.info(buf=buffer)
                st.text(buffer.getvalue())

            # Summary Statistics
            st.subheader("Summary Statistics")
            try:
                st.write(df_display.describe(include='all', datetime_is_numeric=True)) # Include descriptive stats for non-numeric too
            except Exception as e:
                st.error(f"Could not generate summary statistics: {e}")
                st.write(df_display.describe(include=np.number)) # Fallback to numeric only

            # Data Visualization
            st.subheader("Data Visualization")
            numeric_cols = df_display.select_dtypes(include=np.number).columns.tolist()
            # date_cols_for_plot = df_display.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist() # Use the validated main_date_col

            if main_date_col and numeric_cols:
                # date_col_plot = st.selectbox("Select date/time column for plotting:", date_cols_for_plot) # No need to select if we have 'Date'
                numeric_col_plot = st.selectbox("Select numeric column for time series plot:", numeric_cols, index=numeric_cols.index('Close') if 'Close' in numeric_cols else 0)
                if numeric_col_plot:
                    try:
                        fig = px.line(df_display, x=main_date_col, y=numeric_col_plot, title=f"{numeric_col_plot} Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting time series: {e}")

            # Correlation Heatmap
            if len(numeric_cols) > 1:
                st.subheader("Correlation Heatmap")
                numeric_data = df_display[numeric_cols]
                # Only compute correlation on valid numeric data (drop potential non-numeric if any slipped through)
                numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
                if len(numeric_data.columns) > 1:
                    corr = numeric_data.corr()
                    fig = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Matrix of Numeric Features")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough valid numeric columns to compute correlation matrix.")


            # Download Button
            st.markdown(get_download_link(st.session_state['data'], 'loaded_financial_data.csv', 'Download Raw Data as CSV'), unsafe_allow_html=True)

            if st.button("Proceed to Data Preprocessing"):
                st.session_state['stage'] = 2
                st.success("Moving to data preprocessing!")
                st.experimental_rerun()
        else:
            st.warning("No data loaded. Please select a data source from the sidebar.")

        st.markdown('</div>', unsafe_allow_html=True)


# 2. Preprocessing
elif st.session_state['stage'] == 2:
    st.title("2️⃣ Data Preprocessing")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state['data'] is not None:
            # Use processed_data if it exists, otherwise start from original data
            if st.session_state['processed_data'] is None:
                # Create a fresh copy when starting preprocessing
                data_to_process = st.session_state['data'].copy()
                st.session_state['processed_data'] = data_to_process # Initialize processed data
            else:
                # Use the already modified data for further steps
                data_to_process = st.session_state['processed_data'].copy()


            st.subheader("Current Data State")
            st.dataframe(data_to_process.head())
            st.write(f"Shape: {data_to_process.shape}")

            # Ensure Date/Time columns are handled correctly (attempt conversion)
            for col in data_to_process.select_dtypes(include=['object']).columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        data_to_process[col] = pd.to_datetime(data_to_process[col], errors='ignore')
                    except Exception:
                         pass # Ignore if conversion fails

            # --- Missing Values ---
            st.subheader("Missing Values Analysis")
            missing_values = data_to_process.isnull().sum()
            missing_percent = (missing_values / len(data_to_process)) * 100
            missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
            missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)

            if not missing_df.empty:
                st.write(missing_df)
                fig = px.bar(missing_df.reset_index(), x='index', y='Percentage', title='Missing Values by Column (%)', labels={'index':'Column'}, color='Percentage', color_continuous_scale='Oranges')
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Handle Missing Values")
                handling_method = st.selectbox(
                    "Select method:",
                    ["Drop rows with any missing values",
                     "Fill numeric with Mean",
                     "Fill numeric with Median",
                     "Fill numeric with Zero",
                     "Fill categorical with Mode",
                     "Skip Missing Value Handling"],
                    index=5 # Default to skip
                )

                if st.button("Apply Missing Values Treatment"):
                    with st.spinner("Processing missing values..."):
                        before_rows = len(data_to_process)
                        processed_data = data_to_process.copy() # Work on a copy

                        if handling_method == "Drop rows with any missing values":
                            processed_data.dropna(inplace=True)
                            st.info(f"Dropped {before_rows - len(processed_data)} rows containing NaNs.")
                        elif handling_method == "Fill numeric with Mean":
                            num_cols = processed_data.select_dtypes(include=np.number).columns
                            processed_data[num_cols] = processed_data[num_cols].fillna(processed_data[num_cols].mean())
                            st.info(f"Filled numeric NaNs with column means for columns: {', '.join(num_cols)}")
                        elif handling_method == "Fill numeric with Median":
                            num_cols = processed_data.select_dtypes(include=np.number).columns
                            processed_data[num_cols] = processed_data[num_cols].fillna(processed_data[num_cols].median())
                            st.info(f"Filled numeric NaNs with column medians for columns: {', '.join(num_cols)}")
                        elif handling_method == "Fill numeric with Zero":
                            num_cols = processed_data.select_dtypes(include=np.number).columns
                            processed_data[num_cols] = processed_data[num_cols].fillna(0)
                            st.info(f"Filled numeric NaNs with zero for columns: {', '.join(num_cols)}")
                        elif handling_method == "Fill categorical with Mode":
                             cat_cols = processed_data.select_dtypes(include='object').columns
                             for col in cat_cols:
                                 if processed_data[col].isnull().any():
                                     mode_val = processed_data[col].mode()
                                     if not mode_val.empty:
                                         processed_data[col].fillna(mode_val[0], inplace=True)
                             st.info(f"Attempted to fill categorical NaNs with column modes for columns: {', '.join(cat_cols)}")
                        elif handling_method == "Skip Missing Value Handling":
                             st.info("Skipped missing value handling.")

                        st.session_state['processed_data'] = processed_data # Update session state
                        st.success("Missing value treatment applied (or skipped).")
                        st.experimental_rerun() # Rerun to reflect changes

            else:
                st.success("✅ No missing values found in the current dataset!")

            # --- Outlier Detection ---
            st.subheader("Outlier Detection & Handling")
            # Make sure to use the potentially updated data_to_process from missing value handling
            data_to_process = st.session_state['processed_data'].copy() # Refresh from state
            numeric_cols = data_to_process.select_dtypes(include=np.number).columns.tolist()

            if numeric_cols:
                selected_col_outlier = st.selectbox("Select column to analyze for outliers:", numeric_cols, key="outlier_col_select")

                col1, col2 = st.columns(2)
                with col1:
                    try:
                        fig_box = px.box(data_to_process, y=selected_col_outlier, title=f"Box Plot for {selected_col_outlier}")
                        st.plotly_chart(fig_box, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not plot boxplot for {selected_col_outlier}: {e}")
                with col2:
                    try:
                        fig_hist = px.histogram(data_to_process, x=selected_col_outlier, marginal="box", title=f"Distribution of {selected_col_outlier}")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not plot histogram for {selected_col_outlier}: {e}")


                outlier_method = st.selectbox(
                    f"Select method to handle outliers in '{selected_col_outlier}':",
                    ["No treatment",
                     "Remove outliers (IQR method)",
                     "Cap outliers (IQR method)",
                     "Log Transform (handle non-positive)"],
                    index=0 # Default to no treatment
                )

                if st.button("Apply Outlier Treatment"):
                    with st.spinner(f"Applying treatment to {selected_col_outlier}..."):
                        processed_data = data_to_process.copy() # Work on a copy
                        before_rows = len(processed_data)
                        col = selected_col_outlier

                        if col not in processed_data.columns:
                            st.error(f"Column '{col}' not found in the data. This might happen after previous processing steps.")
                        elif not pd.api.types.is_numeric_dtype(processed_data[col]):
                             st.error(f"Column '{col}' is not numeric. Cannot apply outlier treatment.")
                        else:
                            if outlier_method == "Remove outliers (IQR method)":
                                Q1 = processed_data[col].quantile(0.25)
                                Q3 = processed_data[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                processed_data = processed_data[(processed_data[col] >= lower_bound) & (processed_data[col] <= upper_bound)]
                                st.info(f"Removed {before_rows - len(processed_data)} outlier rows based on {col}.")
                            elif outlier_method == "Cap outliers (IQR method)":
                                Q1 = processed_data[col].quantile(0.25)
                                Q3 = processed_data[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                outliers_count = ((processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)).sum()
                                processed_data[col] = processed_data[col].clip(lower=lower_bound, upper=upper_bound)
                                st.info(f"Capped {outliers_count} outlier values in {col}.")
                            elif outlier_method == "Log Transform (handle non-positive)":
                                if (processed_data[col] <= 0).any():
                                    min_val = processed_data[col][processed_data[col] > 0].min() # Find min positive value
                                    shift = min_val / 2 if pd.notna(min_val) and min_val > 0 else 1e-9 # Add small positive value
                                    processed_data[f"{col}_log"] = np.log(processed_data[col] + shift)
                                    st.info(f"Applied log transform to {col} (added approx {shift:.2e} before log). New column: {col}_log")
                                else:
                                    processed_data[f"{col}_log"] = np.log(processed_data[col])
                                    st.info(f"Applied log transform to {col}. New column: {col}_log")
                            elif outlier_method == "No treatment":
                                st.info("No outlier treatment applied.")

                            st.session_state['processed_data'] = processed_data # Update session state
                            st.success("Outlier treatment applied (or skipped).")
                            st.experimental_rerun() # Rerun to reflect changes
            else:
                st.warning("No numeric columns found for outlier analysis.")

            # --- Feature Scaling (REMOVED - Now done after Train/Test split) ---
            # st.subheader("Feature Scaling") - Section removed

            # --- Proceed Button ---
            st.markdown("---")
            st.subheader("Final Processed Data Preview (after this stage)")
            st.dataframe(st.session_state['processed_data'].head())

            if st.button("Confirm Preprocessing & Proceed to Feature Engineering"):
                # Final check before moving on
                if st.session_state['processed_data'] is None:
                     st.session_state['processed_data'] = data_to_process # Ensure data is saved if no steps were applied
                st.session_state['stage'] = 3
                # Reset downstream states that depend on features/target
                st.session_state['features'] = None
                st.session_state['target'] = None
                st.session_state['X_train'] = None
                st.session_state['X_test'] = None
                st.session_state['y_train'] = None
                st.session_state['y_test'] = None
                st.session_state['scaler'] = None
                st.session_state['model'] = None
                st.session_state['split_done'] = False
                st.session_state['training_done'] = False
                st.session_state['evaluation_done'] = False

                st.success("Preprocessing steps confirmed! Moving to Feature Engineering.")
                st.experimental_rerun()
        else:
            st.warning("No data available. Please load data first in Stage 1.")

        st.markdown('</div>', unsafe_allow_html=True)


# 3. Feature Engineering
elif st.session_state['stage'] == 3:
    st.title("3️⃣ Feature Engineering")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state['processed_data'] is not None:
            # Use the latest processed data
            data = st.session_state['processed_data'].copy()

            st.subheader("Current Data State")
            st.dataframe(data.head())
            st.write(f"Shape: {data.shape}")

            # Display available columns and types
            st.subheader("Available Columns")
            col_info = pd.DataFrame({
                'Column': data.columns,
                'DataType': data.dtypes.astype(str)
            })
            st.table(col_info)

            # --- Target Variable Selection (for Supervised Learning) ---
            st.subheader("Target Variable Selection")
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            all_cols = data.columns.tolist()
            target_col = st.session_state.get('target') # Get existing target if set

            # Ensure target exists in current columns (might have been dropped/renamed)
            if target_col and target_col not in data.columns:
                 st.warning(f"Previously selected target '{target_col}' no longer exists. Please select again.")
                 target_col = None
                 st.session_state['target'] = None


            if st.session_state['model_type'] == "Linear Regression":
                possible_targets = numeric_cols
                if not possible_targets:
                    st.error("Linear Regression requires numeric columns for the target. None found in the current data.")
                else:
                    target_col = st.selectbox(
                        "Select target variable (must be numeric):",
                        possible_targets,
                        index=possible_targets.index(target_col) if target_col and target_col in possible_targets else 0,
                        key="linreg_target"
                    )
                    st.session_state['target'] = target_col

            elif st.session_state['model_type'] == "Logistic Regression":
                # Allow selecting numeric or low-cardinality categorical/object
                possible_targets_cat = [col for col in data.columns if data[col].nunique() < 10 and col not in numeric_cols]
                possible_targets = numeric_cols + possible_targets_cat
                if not possible_targets:
                    st.error("Logistic Regression requires a target variable. No suitable columns (numeric or low unique values) found.")
                else:
                    target_col = st.selectbox(
                        "Select target variable (numeric or low unique values):",
                        possible_targets,
                        index=possible_targets.index(target_col) if target_col and target_col in possible_targets else 0,
                        key="logreg_target"
                    )
                    st.session_state['target'] = target_col

                    # Binarization option if target is numeric or > 2 classes
                    if target_col and (pd.api.types.is_numeric_dtype(data[target_col]) or data[target_col].nunique() > 2):
                         st.info(f"Target '{target_col}' has {data[target_col].nunique()} unique values.")
                         if st.checkbox(f"Binarize target '{target_col}' for Logistic Regression?"):
                            if pd.api.types.is_numeric_dtype(data[target_col]):
                                median_val = data[target_col].median()
                                threshold = st.slider(f"Select threshold for '{target_col}' (> threshold = 1):", float(data[target_col].min()), float(data[target_col].max()), float(median_val))
                                if st.button("Apply Binarization"):
                                     binary_target_name = f"{target_col}_binary"
                                     data[binary_target_name] = (data[target_col] > threshold).astype(int)
                                     st.session_state['target'] = binary_target_name # Update target
                                     st.session_state['processed_data'] = data # Save changes
                                     st.success(f"Created binary target '{binary_target_name}' based on threshold {threshold:.2f}.")
                                     st.experimental_rerun()
                            else: # Handle categorical binarization (e.g., select one category vs rest)
                                unique_vals = data[target_col].unique()
                                positive_class = st.selectbox("Select the 'positive' class (will be mapped to 1):", unique_vals)
                                if st.button("Apply Binarization"):
                                    binary_target_name = f"{target_col}_binary_{positive_class}"
                                    data[binary_target_name] = (data[target_col] == positive_class).astype(int)
                                    st.session_state['target'] = binary_target_name # Update target
                                    st.session_state['processed_data'] = data # Save changes
                                    st.success(f"Created binary target '{binary_target_name}' where '{positive_class}' is 1.")
                                    st.experimental_rerun()


            elif st.session_state['model_type'] == "K-Means Clustering":
                st.info("K-Means is unsupervised and does not require a target variable.")
                st.session_state['target'] = None
                target_col = None # Ensure target_col is None for later logic

            # --- Feature Selection ---
            st.subheader("Feature Selection")
            # Select only numeric features from the current data state
            available_features = data.select_dtypes(include=np.number).columns.tolist()
            if target_col and target_col in available_features:
                available_features.remove(target_col) # Don't use target as a feature

            # Also remove binary target's original column if it exists
            if target_col and target_col.endswith("_binary") and target_col[:-7] in available_features:
                available_features.remove(target_col[:-7])
            if target_col and "_binary_" in target_col: # Handle categorical binary case
                 original_col_name = target_col.split("_binary_")[0]
                 if original_col_name in available_features:
                      available_features.remove(original_col_name)


            if not available_features:
                st.warning("No numeric features available for selection (after excluding target/non-numeric). Check previous steps.")
                selected_features = []
                st.session_state['features'] = [] # Ensure it's an empty list
            else:
                st.write("Select numeric features to use for the model:")
                 # Use multiselect for easier selection
                selected_features = st.multiselect(
                     "Features:",
                     available_features,
                     default=st.session_state.get('features', []) # Default to previous selection or empty list
                )

            if st.button("Confirm Selected Features"):
                if not selected_features:
                    st.error("Please select at least one feature.")
                else:
                    st.session_state['features'] = selected_features
                    st.success(f"Confirmed {len(selected_features)} features: {', '.join(selected_features)}")
                    # Optionally show feature importance preview here if target is selected
                    if target_col and selected_features:
                        try:
                            # Ensure all selected columns exist and handle NaNs for preview
                            cols_to_preview = selected_features + [target_col]
                            if not all(c in data.columns for c in cols_to_preview):
                                st.warning("Some selected columns for preview are missing. Skipping importance preview.")
                            else:
                                preview_data = data[cols_to_preview].copy().dropna()
                                X_preview = preview_data[selected_features]
                                y_preview = preview_data[target_col]


                                if len(X_preview) > 1 and len(y_preview.unique()) > 1 : # Need at least 2 samples and variability in target
                                    with st.spinner("Calculating feature importance preview..."):
                                        score_func = None
                                        score_func_name = ""
                                        if st.session_state['model_type'] == "Linear Regression" and pd.api.types.is_numeric_dtype(y_preview):
                                            score_func = f_regression
                                            score_func_name = "F-Regression Score"
                                        elif st.session_state['model_type'] == "Logistic Regression" and y_preview.nunique() > 1: # Check if classification is possible
                                            score_func = f_classif
                                            score_func_name = "F-Classification Score"

                                        if score_func:
                                            selector = SelectKBest(score_func, k='all')
                                            selector.fit(X_preview, y_preview)
                                            scores = selector.scores_
                                            importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': scores}).sort_values('Importance', ascending=False)
                                            st.session_state['feature_importance'] = importance_df # Store importance
                                            fig_imp = px.bar(importance_df, x='Feature', y='Importance', title=f'Feature Importance Preview ({score_func_name})', color='Importance', color_continuous_scale='Viridis')
                                            st.plotly_chart(fig_imp, use_container_width=True)
                                        else:
                                            st.info("Feature importance preview not applicable for this model type or target variable characteristics.")
                                else:
                                    st.warning("Not enough data or target variability after cleaning NaNs to calculate feature importance preview.")

                        except Exception as e:
                            st.warning(f"Could not calculate feature importance preview: {e}")

                    # Rerun needed if features change to update multiselect default
                    st.experimental_rerun()


            # --- Feature Creation Options ---
            st.subheader("Feature Creation (Optional)")
            st.warning("⚠️ **Potential Data Leakage:** Creating features like rolling averages or technical indicators *before* the train/test split can leak future information into the training set, potentially leading to overly optimistic results. It's generally safer to calculate these *after* splitting, though this UI doesn't support that flow easily. Proceed with caution.")


            # Date Features
            date_cols_create = data.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
            if date_cols_create:
                with st.expander("📅 Create Date-Based Features"):
                    selected_date_col_create = st.selectbox("Select date column:", date_cols_create, key="date_feat_col")
                    create_year = st.checkbox("Year", key="df_year")
                    create_month = st.checkbox("Month", key="df_month")
                    create_day = st.checkbox("Day", key="df_day")
                    create_dayofweek = st.checkbox("Day of Week", key="df_dow")
                    create_quarter = st.checkbox("Quarter", key="df_quarter")

                    if st.button("Create Selected Date Features"):
                        with st.spinner("Creating date features..."):
                            new_data = data.copy() # Work on a copy
                            if create_year: new_data[f'{selected_date_col_create}_year'] = new_data[selected_date_col_create].dt.year
                            if create_month: new_data[f'{selected_date_col_create}_month'] = new_data[selected_date_col_create].dt.month
                            if create_day: new_data[f'{selected_date_col_create}_day'] = new_data[selected_date_col_create].dt.day
                            if create_dayofweek: new_data[f'{selected_date_col_create}_dayofweek'] = new_data[selected_date_col_create].dt.dayofweek
                            if create_quarter: new_data[f'{selected_date_col_create}_quarter'] = new_data[selected_date_col_create].dt.quarter
                            st.session_state['processed_data'] = new_data # Update data
                            # Reset features list as new columns are added
                            st.session_state['features'] = None
                            st.success("Date features created. Please re-confirm features.")
                            st.experimental_rerun()

            # Technical Indicators (if OHLCV data exists)
            req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            has_ohlcv = all(col in data.columns for col in req_cols)
            if has_ohlcv:
                with st.expander("📊 Create Technical Indicators"):
                    st.info("Requires 'Open', 'High', 'Low', 'Close', 'Volume' columns.")
                    create_ma = st.checkbox("Moving Averages (5, 20, 50)", key="ti_ma")
                    create_rsi = st.checkbox("RSI (14)", key="ti_rsi")
                    create_bb = st.checkbox("Bollinger Bands (20, 2)", key="ti_bb")
                    create_macd = st.checkbox("MACD (12, 26, 9)", key="ti_macd")

                    if st.button("Create Selected Indicators"):
                        with st.spinner("Creating indicators..."):
                            new_data = data.copy().sort_values(by='Date') # Ensure data is sorted by date for rolling calcs
                            rows_before = len(new_data)

                            if create_ma:
                                for period in [5, 20, 50]: new_data[f'MA_{period}'] = new_data['Close'].rolling(window=period, min_periods=1).mean()
                            if create_rsi:
                                delta = new_data['Close'].diff()
                                gain = delta.where(delta > 0, 0).fillna(0)
                                loss = -delta.where(delta < 0, 0).fillna(0)
                                avg_gain = gain.rolling(window=14, min_periods=1).mean()
                                avg_loss = loss.rolling(window=14, min_periods=1).mean()
                                rs = avg_gain / (avg_loss + 1e-9) # Add epsilon to prevent division by zero
                                new_data['RSI_14'] = 100 - (100 / (1 + rs))
                                new_data['RSI_14'].fillna(50, inplace=True) # Fill initial NaNs with neutral 50
                            if create_bb:
                                ma20 = new_data['Close'].rolling(window=20, min_periods=1).mean()
                                sd20 = new_data['Close'].rolling(window=20, min_periods=1).std().fillna(0)
                                new_data['BB_Upper'] = ma20 + (sd20 * 2)
                                new_data['BB_Lower'] = ma20 - (sd20 * 2)
                            if create_macd:
                                ema12 = new_data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
                                ema26 = new_data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
                                new_data['MACD'] = ema12 - ema26
                                new_data['MACD_Signal'] = new_data['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()

                            # Option to drop rows with NaNs created by rolling funcs (or keep and handle later)
                            handle_indicator_nans = st.radio(
                                "Handle NaNs created by indicators?",
                                ["Drop rows with NaNs", "Keep rows (handle later)"], index=1
                            )
                            if handle_indicator_nans == "Drop rows with NaNs":
                                new_data.dropna(inplace=True)
                                st.info(f"Dropped {rows_before - len(new_data)} rows with NaNs from indicator calculation.")

                            st.session_state['processed_data'] = new_data
                             # Reset features list as new columns are added
                            st.session_state['features'] = None
                            st.success("Technical indicators created. Please re-confirm features.")
                            st.experimental_rerun()

            # Interaction Features
            current_numeric_features = data.select_dtypes(include=np.number).columns.tolist()
            if target_col and target_col in current_numeric_features:
                 current_numeric_features.remove(target_col)

            if len(current_numeric_features) >= 2:
                 with st.expander("🔗 Create Interaction Features"):
                    col1_int, col2_int = st.columns(2)
                    with col1_int: feat1 = st.selectbox("Select first feature:", current_numeric_features, key="interact_1")
                    with col2_int: feat2 = st.selectbox("Select second feature:", [c for c in current_numeric_features if c != feat1], key="interact_2")

                    interaction_type = st.selectbox("Select interaction type:", ["Multiplication", "Division", "Addition", "Subtraction"])

                    if st.button("Create Interaction Feature"):
                        if feat1 and feat2:
                            with st.spinner("Creating interaction..."):
                                new_data = data.copy()
                                new_feat_name = ""
                                if interaction_type == "Multiplication":
                                    new_feat_name = f'{feat1}_x_{feat2}'
                                    new_data[new_feat_name] = new_data[feat1] * new_data[feat2]
                                elif interaction_type == "Addition":
                                    new_feat_name = f'{feat1}_plus_{feat2}'
                                    new_data[new_feat_name] = new_data[feat1] + new_data[feat2]
                                elif interaction_type == "Subtraction":
                                    new_feat_name = f'{feat1}_minus_{feat2}'
                                    new_data[new_feat_name] = new_data[feat1] - new_data[feat2]
                                elif interaction_type == "Division":
                                    new_feat_name = f'{feat1}_div_{feat2}'
                                    # Add small epsilon to prevent division by zero
                                    new_data[new_feat_name] = new_data[feat1] / (new_data[feat2] + 1e-9)
                                    new_data[new_feat_name].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle potential inf

                                st.session_state['processed_data'] = new_data
                                # Reset features list as new columns are added
                                st.session_state['features'] = None
                                st.success(f"Created '{new_feat_name}' feature. Please re-confirm features.")
                                st.experimental_rerun()
                        else:
                            st.warning("Please select two distinct features for interaction.")


            # --- Proceed Button ---
            st.markdown("---")
            if st.session_state.get('features') is not None: # Check if features have been confirmed (not None)
                proceed_split = st.button("Confirm Features & Proceed to Train/Test Split")

                if proceed_split:
                     # --- FINAL CHECK before splitting ---
                    final_data = st.session_state['processed_data']
                    final_features = st.session_state['features']
                    final_target = st.session_state.get('target') # Might be None for K-Means

                    # Verify columns exist
                    cols_to_check = final_features + ([final_target] if final_target else [])
                    missing_cols = [c for c in cols_to_check if c not in final_data.columns]
                    if missing_cols:
                         st.error(f"Error: The following selected columns are missing from the processed data: {', '.join(missing_cols)}. This might be due to removal in previous steps. Please go back and re-select features/target.")
                    else:
                        # Check for NaNs in selected columns
                        nan_counts = final_data[cols_to_check].isnull().sum()
                        if nan_counts.sum() > 0:
                            st.warning(f"Warning: {nan_counts.sum()} NaN values found in selected columns: {list(nan_counts[nan_counts > 0].index)}. These rows will be dropped during Train/Test split.")
                            # Display NaN counts for affected columns
                            st.dataframe(nan_counts[nan_counts > 0].reset_index().rename(columns={'index':'Column', 0:'NaN Count'}))


                        st.session_state['stage'] = 4
                        # Reset downstream states
                        st.session_state['X_train'] = None
                        st.session_state['X_test'] = None
                        st.session_state['y_train'] = None
                        st.session_state['y_test'] = None
                        st.session_state['scaler'] = None
                        st.session_state['model'] = None
                        st.session_state['split_done'] = False
                        st.session_state['training_done'] = False
                        st.session_state['evaluation_done'] = False

                        st.success("Feature selection/engineering confirmed! Moving to Train/Test Split.")
                        st.experimental_rerun()
            else:
                st.warning("Please select features using the 'Features' dropdown and click 'Confirm Selected Features' before proceeding.")

        else:
            st.warning("No processed data available. Please complete Stage 2 first.")

        st.markdown('</div>', unsafe_allow_html=True)


# 4. Train/Test Split
elif st.session_state['stage'] == 4:
    st.title("4️⃣ Train/Test Split")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        # Check if essential data exists
        if st.session_state['processed_data'] is None:
             st.warning("Processed data not found. Please complete previous stages.")
        elif st.session_state.get('features') is None:
             st.warning("Features not selected/confirmed. Please go back to Stage 3.")
        else:
            data = st.session_state['processed_data'].copy()
            features = st.session_state['features']
            target = st.session_state.get('target')

            # Verify columns again (safety check)
            cols_to_use = features + ([target] if target else [])
            if not all(c in data.columns for c in cols_to_use):
                 st.error("Selected feature/target columns not found in the data. Please revisit Stage 3.")
            else:
                 # --- Supervised Learning Split ---
                if st.session_state['model_type'] in ["Linear Regression", "Logistic Regression"]:
                    if not target:
                        st.error("Target variable not selected for supervised learning. Please go back to Stage 3.")
                    else:
                        st.subheader("Split Data for Supervised Learning")
                        st.write(f"**Features:** {', '.join(features)}")
                        st.write(f"**Target:** {target}")

                        # Prepare X and y, explicitly handling potential NaNs
                        X = data[features].copy()
                        y = data[target].copy()
                        initial_rows = len(X)

                        # Drop rows where target or any feature is NaN
                        combined = pd.concat([X, y], axis=1)
                        rows_before_na = len(combined)
                        combined.dropna(subset=features + [target], inplace=True)
                        rows_after_na = len(combined)

                        if rows_after_na < rows_before_na:
                            st.warning(f"Removed {rows_before_na - rows_after_na} rows due to NaN values in selected features or target before splitting.")

                        if len(combined) < 2:
                            st.error("Not enough data remaining after handling NaNs to perform train/test split (need at least 2 samples).")
                        else:
                            X = combined[features]
                            y = combined[target]

                            test_size = st.slider("Select test set size (%):", min_value=10, max_value=50, value=25, step=1, key="split_slider") / 100
                            random_state = st.number_input("Random state (for reproducibility):", min_value=0, value=42, step=1, key="split_random")

                            # Stratify option for classification
                            stratify_option = None
                            if st.session_state['model_type'] == "Logistic Regression":
                                 # Only stratify if target is categorical-like (few unique values) and has enough samples per class
                                 if y.nunique() < 10 and y.nunique() > 1:
                                     min_class_count = y.value_counts().min()
                                     # Need at least 2 samples per class for stratification typically
                                     if min_class_count >= 2:
                                         stratify_option = y
                                         st.info("Stratifying split based on target variable.")
                                     else:
                                          st.warning(f"Cannot stratify: Target class has only {min_class_count} sample(s). Proceeding without stratification.")
                                 else:
                                     st.info("Stratification not applied (target is numeric, has too many unique values, or only one class).")


                            if st.button("Split Data"):
                                with st.spinner("Splitting data..."):
                                    try:
                                        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y,
                                            test_size=test_size,
                                            random_state=int(random_state),
                                            stratify=stratify_option # Apply stratification if determined above
                                        )

                                        st.session_state['X_train'] = X_train
                                        st.session_state['X_test'] = X_test
                                        st.session_state['y_train'] = y_train
                                        st.session_state['y_test'] = y_test
                                        st.session_state['scaler'] = None # Reset scaler before training
                                        st.session_state['model'] = None # Reset model
                                        st.session_state['training_done'] = False
                                        st.session_state['evaluation_done'] = False


                                        st.success(f"Data split: Train ({X_train.shape[0]} samples), Test ({X_test.shape[0]} samples)")

                                        # Visualize split
                                        split_df = pd.DataFrame({'Set': ['Train', 'Test'], 'Samples': [len(X_train), len(X_test)]})
                                        fig_pie = px.pie(split_df, values='Samples', names='Set', title='Train/Test Split Ratio', hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
                                        st.plotly_chart(fig_pie)

                                        st.subheader("Train Data Preview")
                                        st.dataframe(X_train.head())
                                        st.subheader("Test Data Preview")
                                        st.dataframe(X_test.head())

                                        # Proceed button enabled after successful split
                                        st.session_state['split_done'] = True
                                        st.experimental_rerun() # Rerun to show proceed button

                                    except ValueError as ve:
                                        st.error(f"Error during split: {ve}. This might happen if a class has too few samples for stratification or other input issues.")
                                    except Exception as e:
                                        st.error(f"An unexpected error occurred during split: {e}")

                # --- Unsupervised Learning Preparation ---
                elif st.session_state['model_type'] == "K-Means Clustering":
                    st.subheader("Prepare Data for Clustering")
                    st.write(f"**Features:** {', '.join(features)}")

                    X = data[features].copy()
                    rows_before_na = len(X)
                    X.dropna(inplace=True) # Drop rows with any NaNs in selected features
                    rows_after_na = len(X)

                    if rows_after_na < rows_before_na:
                        st.warning(f"Removed {rows_before_na - rows_after_na} rows due to NaN values in selected features.")

                    if len(X) < 2:
                        st.error("Not enough data remaining after handling NaNs for clustering (need at least 2 samples).")
                    else:
                        st.info("K-Means uses the full dataset (after NaN removal). No train/test split needed. Scaling will be applied during training.")
                        # Store the prepared data
                        st.session_state['X_train'] = X # Using X_train key for consistency, represents full dataset for clustering
                        st.session_state['X_test'] = None
                        st.session_state['y_train'] = None
                        st.session_state['y_test'] = None
                        st.session_state['scaler'] = None # Reset scaler before training
                        st.session_state['model'] = None # Reset model
                        st.session_state['training_done'] = False
                        st.session_state['evaluation_done'] = False

                        st.success(f"Data prepared for clustering: {X.shape[0]} samples, {X.shape[1]} features.")
                        st.dataframe(X.head())
                        st.session_state['split_done'] = True # Mark as ready to proceed
                        st.experimental_rerun() # Rerun to show proceed button


                # --- Proceed Button ---
                if st.session_state.get('split_done'):
                    if st.button("Proceed to Model Training"):
                        st.session_state['stage'] = 5
                        st.success("Data ready! Moving to Model Training.")
                        st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# 5. Model Training
elif st.session_state['stage'] == 5:
    st.title("5️⃣ Model Training")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        # Ensure split was done and data is available
        if not st.session_state.get('split_done') or st.session_state.get('X_train') is None:
            st.warning("Train/Test split not completed or data not found. Please complete Stage 4 first.")
        else:
            model_type = st.session_state['model_type']
            st.subheader(f"Train {model_type} Model")

            X_train = st.session_state['X_train']
            y_train = st.session_state.get('y_train') # Will be None for K-Means

            # --- SCALING (Refactored for LinReg/LogReg) ---
            scaler = None
            X_train_final = X_train # Default to original if no scaling
            if model_type in ["Linear Regression", "Logistic Regression"]:
                st.subheader("Feature Scaling (Applied before Training)")
                scaling_method_train = st.selectbox(
                    "Select scaling method for training data:",
                    ["No scaling", "StandardScaler (Z-score)", "MinMaxScaler (0-1)"],
                    index=0, # Default to no scaling
                    key="train_scaling"
                )

                if scaling_method_train == "StandardScaler (Z-score)":
                    scaler = StandardScaler()
                    X_train_final = scaler.fit_transform(X_train)
                    st.info("Applied StandardScaler to training features.")
                elif scaling_method_train == "MinMaxScaler (0-1)":
                    scaler = MinMaxScaler()
                    X_train_final = scaler.fit_transform(X_train)
                    st.info("Applied MinMaxScaler to training features.")
                else:
                    st.info("No scaling applied to training features.")

                st.session_state['scaler'] = scaler # Store the FITTED scaler (or None)
                st.session_state['X_train_scaled'] = X_train_final # Store scaled data if needed


            # --- Linear Regression Training ---
            if model_type == "Linear Regression":
                if y_train is None:
                     st.error("Training data target (y_train) not found. Please ensure split was successful.")
                else:
                    # Parameters - Keep it simple for LinReg
                    # fit_intercept = st.checkbox("Fit intercept", value=True) # Default True

                    if st.button("Train Linear Regression Model"):
                        with st.spinner("Training Linear Regression..."):
                            try:
                                model = LinearRegression() # Use defaults
                                model.fit(X_train_final, y_train) # Use potentially scaled data

                                st.session_state['model'] = model
                                st.session_state['training_done'] = True # Mark training as complete
                                st.success("Linear Regression model trained successfully!")

                                # Display Coefficients
                                coef_df = pd.DataFrame({
                                    'Feature': st.session_state['features'],
                                    'Coefficient': model.coef_
                                }).sort_values('Coefficient', key=abs, ascending=False) # Sort by absolute value
                                st.write("**Model Coefficients:**")
                                fig_coef = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Coefficients', color='Coefficient', color_continuous_scale='RdBu')
                                st.plotly_chart(fig_coef, use_container_width=True)
                                st.write(f"**Intercept:** {model.intercept_:.4f}")

                                st.experimental_rerun()

                            except Exception as e:
                                st.error(f"Error training Linear Regression: {e}")
                                st.session_state['training_done'] = False


            # --- Logistic Regression Training ---
            elif model_type == "Logistic Regression":
                 if y_train is None:
                     st.error("Training data target (y_train) not found. Please ensure split was successful.")
                 else:
                     # Parameters
                     solver = st.selectbox("Solver:", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"], index=0, key="logreg_solver")
                     max_iter = st.slider("Maximum iterations:", min_value=100, max_value=1000, value=100, step=100, key="logreg_iter")
                     C = st.slider("Regularization strength (C - Smaller values = stronger regularization):", min_value=0.01, max_value=10.0, value=1.0, step=0.1, format="%.2f", key="logreg_c")

                     if st.button("Train Logistic Regression Model"):
                         with st.spinner("Training Logistic Regression..."):
                             try:
                                 model = LogisticRegression(solver=solver, max_iter=max_iter, C=C, random_state=42)
                                 model.fit(X_train_final, y_train) # Use potentially scaled data

                                 st.session_state['model'] = model
                                 st.session_state['training_done'] = True # Mark training as complete
                                 st.success("Logistic Regression model trained successfully!")

                                 # Display Coefficients (handle multi-class if necessary)
                                 n_classes = len(model.classes_)
                                 if n_classes > 2:
                                      st.info(f"Multi-class classification ({n_classes} classes) detected. Showing coefficients for class '{model.classes_[1]}' vs rest.")
                                      coefs = model.coef_[1] # Show coefs for the second class vs others
                                      intercept = model.intercept_[1]
                                 elif n_classes == 2:
                                      coefs = model.coef_[0] # Coefs for class 1
                                      intercept = model.intercept_[0]
                                 else: # Should not happen if split worked
                                     st.warning("Model trained on only one class.")
                                     coefs = [0] * len(st.session_state['features'])
                                     intercept = 0


                                 coef_df = pd.DataFrame({
                                     'Feature': st.session_state['features'],
                                     'Coefficient': coefs
                                 }).sort_values('Coefficient', key=abs, ascending=False) # Sort by abs value
                                 st.write("**Model Coefficients:**")
                                 fig_coef = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Coefficients', color='Coefficient', color_continuous_scale='RdBu')
                                 st.plotly_chart(fig_coef, use_container_width=True)
                                 st.write(f"**Intercept:** {intercept:.4f}")

                                 st.experimental_rerun()

                             except Exception as e:
                                 st.error(f"Error training Logistic Regression: {e}")
                                 st.session_state['training_done'] = False


            # --- K-Means Clustering Training ---
            elif model_type == "K-Means Clustering":
                 # Parameters
                 n_clusters = st.slider("Number of clusters (k):", min_value=2, max_value=15, value=3, step=1, key="kmeans_k")
                 init_method = st.selectbox("Initialization method:", ["k-means++", "random"], index=0, key="kmeans_init")
                 n_init = st.slider("Number of initializations (n_init):", min_value=1, max_value=20, value=10, step=1, key="kmeans_ninit", help="Number of times the k-means algorithm will be run with different centroid seeds.")
                 max_iter_kmeans = st.slider("Maximum iterations per run:", min_value=100, max_value=1000, value=300, step=50, key="kmeans_maxiter")

                 if st.button("Train K-Means Clustering Model"):
                     with st.spinner("Running K-Means Clustering..."):
                         try:
                             X = st.session_state['X_train'] # This is the full dataset prepared earlier

                             # ALWAYS scale data for K-Means before fitting
                             st.info("Applying StandardScaler to features before K-Means clustering.")
                             scaler_kmeans = StandardScaler()
                             X_scaled = scaler_kmeans.fit_transform(X)
                             st.session_state['X_scaled'] = X_scaled # Save scaled data specific to clustering
                             st.session_state['scaler'] = scaler_kmeans # Store the scaler used for Kmeans


                             model = KMeans(
                                 n_clusters=n_clusters,
                                 init=init_method,
                                 n_init=n_init, # Use the parameter name expected by sklearn
                                 max_iter=max_iter_kmeans,
                                 random_state=42
                             )
                             model.fit(X_scaled)

                             st.session_state['model'] = model
                             st.session_state['clusters'] = model.labels_ # Store cluster labels
                             st.session_state['training_done'] = True # Mark training as complete
                             st.success(f"K-Means clustering completed with {n_clusters} clusters!")

                             # Display Cluster Centers (original scale if possible)
                             st.subheader("Cluster Centers")
                             centers_scaled = model.cluster_centers_
                             try:
                                centers_original = scaler_kmeans.inverse_transform(centers_scaled)
                                centers_df = pd.DataFrame(centers_original, columns=st.session_state['features'])
                                centers_df.index.name = 'Cluster'
                                st.write("Cluster Centers (Original Scale):")
                                st.dataframe(centers_df.style.format("{:.2f}"))
                             except Exception as e:
                                 st.warning(f"Could not inverse transform centers to original scale: {e}")
                                 st.write("Cluster Centers (Scaled Data):")
                                 centers_df_scaled = pd.DataFrame(centers_scaled, columns=st.session_state['features'])
                                 centers_df_scaled.index.name = 'Cluster'
                                 st.dataframe(centers_df_scaled.style.format("{:.2f}"))


                             # Display Inertia
                             st.write(f"**Inertia (Within-cluster sum of squares):** {model.inertia_:.2f}")

                             # Display Cluster Sizes
                             cluster_sizes = pd.Series(model.labels_).value_counts().sort_index()
                             cluster_sizes.index.name = 'Cluster #'
                             cluster_sizes_df = cluster_sizes.reset_index()
                             cluster_sizes_df.columns = ['Cluster', 'Number of Samples']
                             st.write("**Cluster Sizes:**")
                             st.dataframe(cluster_sizes_df)
                             fig_sizes = px.bar(cluster_sizes_df, x='Cluster', y='Number of Samples', title='Cluster Sizes')
                             st.plotly_chart(fig_sizes, use_container_width=True)


                             st.experimental_rerun()

                         except Exception as e:
                             st.error(f"Error running K-Means: {e}")
                             st.session_state['training_done'] = False

            # --- Proceed Button ---
            if st.session_state.get('training_done'):
                if st.button("Proceed to Evaluation"):
                    st.session_state['stage'] = 6
                    # Reset evaluation specific states
                    st.session_state['predictions'] = None
                    st.session_state['probabilities'] = None
                    st.session_state['metrics'] = {}
                    st.session_state['evaluation_done'] = False
                    st.success("Model training complete! Moving to Evaluation.")
                    st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# 6. Evaluation
elif st.session_state['stage'] == 6:
    st.title("6️⃣ Model Evaluation")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if not st.session_state.get('training_done') or not st.session_state.get('model'):
             st.warning("Model not trained yet. Please complete Stage 5 first.")
        else:
            model = st.session_state['model']
            model_type = st.session_state['model_type']
            st.subheader(f"Evaluation Metrics for {model_type}")

            # --- Linear Regression Evaluation ---
            if model_type == "Linear Regression":
                 if st.session_state.get('X_test') is None or st.session_state.get('y_test') is None:
                     st.warning("Test data (X_test, y_test) not found. Cannot evaluate. Please ensure Train/Test split was successful.")
                 else:
                     X_test = st.session_state['X_test']
                     y_test = st.session_state['y_test']
                     scaler = st.session_state.get('scaler') # Get the fitted scaler

                     # --- Apply SCALING to Test Data ---
                     X_test_final = X_test
                     if scaler:
                         try:
                            X_test_final = scaler.transform(X_test)
                            st.info("Applied the scaler (fitted on train data) to the test data.")
                         except Exception as e:
                             st.error(f"Error applying scaler to test data: {e}. Proceeding with unscaled test data.")
                             X_test_final = X_test # Fallback
                     else:
                         st.info("No scaler was used during training. Evaluating on unscaled test data.")

                     # Make Predictions
                     if 'predictions' not in st.session_state or st.session_state['predictions'] is None:
                           try:
                               st.session_state['predictions'] = model.predict(X_test_final)
                           except Exception as e:
                               st.error(f"Error making predictions on test data: {e}")


                     y_pred = st.session_state.get('predictions')

                     if y_pred is not None:
                         # Calculate Metrics
                         try:
                             mse = mean_squared_error(y_test, y_pred)
                             r2 = r2_score(y_test, y_pred)
                             rmse = np.sqrt(mse)
                             st.session_state['metrics'] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

                             st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
                             st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
                             st.write(f"**R² Score:** {r2:.4f}")
                             st.markdown("---")

                             # Actual vs Predicted Plot
                             st.subheader("Actual vs. Predicted Values")
                             fig_pred = go.Figure()
                             fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions', marker=dict(color='#ff6e40', opacity=0.7)))
                             min_val = min(y_test.min(), y_pred.min())
                             max_val = max(y_test.max(), y_pred.max())
                             fig_pred.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal Line (y=x)', line=dict(color='#1e3d59', dash='dash')))
                             fig_pred.update_layout(title='Actual vs. Predicted Values', xaxis_title='Actual Values', yaxis_title='Predicted Values', legend_title="Legend")
                             st.plotly_chart(fig_pred, use_container_width=True)

                             # Residuals Plot
                             st.subheader("Residuals Analysis")
                             residuals = y_test - y_pred
                             fig_res = px.scatter(x=y_pred, y=residuals, title='Residuals vs. Predicted Values', labels={'x': 'Predicted Values', 'y': 'Residuals'}, opacity=0.7)
                             fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                             st.plotly_chart(fig_res, use_container_width=True)

                             fig_res_hist = px.histogram(residuals, nbins=30, title='Distribution of Residuals', labels={'value': 'Residual'})
                             st.plotly_chart(fig_res_hist, use_container_width=True)

                             st.session_state['evaluation_done'] = True
                             st.experimental_rerun() # Rerun to show proceed button

                         except Exception as e:
                              st.error(f"Error calculating or plotting metrics: {e}")
                     else:
                         st.error("Predictions could not be generated. Cannot evaluate.")


            # --- Logistic Regression Evaluation ---
            elif model_type == "Logistic Regression":
                 if st.session_state.get('X_test') is None or st.session_state.get('y_test') is None:
                     st.warning("Test data (X_test, y_test) not found. Cannot evaluate. Please ensure Train/Test split was successful.")
                 else:
                     X_test = st.session_state['X_test']
                     y_test = st.session_state['y_test']
                     scaler = st.session_state.get('scaler') # Get the fitted scaler

                     # --- Apply SCALING to Test Data ---
                     X_test_final = X_test
                     if scaler:
                         try:
                            X_test_final = scaler.transform(X_test)
                            st.info("Applied the scaler (fitted on train data) to the test data.")
                         except Exception as e:
                             st.error(f"Error applying scaler to test data: {e}. Proceeding with unscaled test data.")
                             X_test_final = X_test # Fallback
                     else:
                         st.info("No scaler was used during training. Evaluating on unscaled test data.")

                     # Make Predictions and Probabilities
                     y_pred, y_prob = None, None
                     if 'predictions' not in st.session_state or st.session_state['predictions'] is None:
                           try:
                               st.session_state['predictions'] = model.predict(X_test_final)
                           except Exception as e:
                               st.error(f"Error making predictions on test data: {e}")
                     y_pred = st.session_state.get('predictions')


                     if 'probabilities' not in st.session_state or st.session_state['probabilities'] is None:
                         if hasattr(model, "predict_proba"):
                            try:
                                st.session_state['probabilities'] = model.predict_proba(X_test_final)
                            except Exception as e:
                                st.error(f"Error getting probabilities from test data: {e}")
                         else:
                             st.warning("Model does not support probability predictions.")
                         y_prob = st.session_state.get('probabilities')


                     if y_pred is not None:
                         # Calculate Metrics
                         try:
                             accuracy = accuracy_score(y_test, y_pred)
                             conf_matrix = confusion_matrix(y_test, y_pred)
                             class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                             st.session_state['metrics'] = {'Accuracy': accuracy, 'Confusion Matrix': conf_matrix, 'Classification Report': class_report}

                             st.write(f"**Accuracy:** {accuracy:.4f}")
                             st.markdown("---")

                             # Confusion Matrix Plot
                             st.subheader("Confusion Matrix")
                             try:
                                 # Ensure labels match the unique values in y_test/y_pred potentially subset by CM
                                 labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
                                 if len(labels) == conf_matrix.shape[0]: # Basic check
                                     fig_conf = px.imshow(conf_matrix, text_auto=True, labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                                        x=[str(l) for l in labels], y=[str(l) for l in labels], color_continuous_scale='Blues', title="Confusion Matrix")
                                     fig_conf.update_xaxes(side="bottom", type='category')
                                     fig_conf.update_yaxes(type='category')
                                     st.plotly_chart(fig_conf, use_container_width=True)
                                 else:
                                     st.warning("Label mismatch with confusion matrix dimensions. Displaying raw matrix.")
                                     st.dataframe(pd.DataFrame(conf_matrix, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels]))

                             except Exception as e:
                                st.error(f"Could not plot confusion matrix: {e}")
                                st.write("Raw Confusion Matrix:")
                                st.write(conf_matrix)


                             # Classification Report
                             st.subheader("Classification Report")
                             report_df = pd.DataFrame(class_report).transpose()
                             st.dataframe(report_df.style.format("{:.3f}"))
                             st.markdown("---")

                             # ROC Curve (if binary and probabilities available)
                             n_classes = len(np.unique(y_test)) # Use actual unique classes in test set
                             if y_prob is not None and y_prob.shape[1] == 2 and n_classes == 2:
                                 st.subheader("ROC Curve & AUC")
                                 fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1]) # Probability of the positive class (class 1)
                                 roc_auc = auc(fpr, tpr)
                                 st.session_state['metrics']['AUC'] = roc_auc # Store AUC

                                 fig_roc = go.Figure()
                                 fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.3f})', line=dict(color='#ff6e40')))
                                 fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='#1e3d59', dash='dash')))
                                 fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                                 st.plotly_chart(fig_roc, use_container_width=True)
                             elif y_prob is None:
                                 st.info("ROC curve cannot be plotted as probability scores are unavailable.")
                             elif n_classes != 2:
                                 st.info(f"ROC curve is typically shown for binary classification (found {n_classes} classes in test set).")


                             # Precision-Recall Curve (if binary and probabilities available)
                             if y_prob is not None and y_prob.shape[1] == 2 and n_classes == 2:
                                 st.subheader("Precision-Recall Curve")
                                 precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
                                 avg_precision = average_precision_score(y_test, y_prob[:, 1])
                                 st.session_state['metrics']['Avg Precision'] = avg_precision # Store Avg Precision

                                 fig_pr = px.area(x=recall, y=precision, title=f'Precision-Recall Curve (Avg Precision = {avg_precision:.3f})',
                                                  labels={'x': 'Recall', 'y': 'Precision'})
                                 # Baseline = ratio of positive class
                                 positive_ratio = y_test.mean() # Assumes y_test is 0/1
                                 fig_pr.add_shape(type='line', x0=0, y0=positive_ratio, x1=1, y1=positive_ratio, line=dict(dash='dash', color='grey'), name=f'Baseline ({positive_ratio:.2f})')
                                 fig_pr.update_yaxes(range=[0.0, 1.05])
                                 fig_pr.update_xaxes(range=[0.0, 1.0])
                                 st.plotly_chart(fig_pr, use_container_width=True)
                             # No PR curve info needed if ROC info was already shown


                             st.session_state['evaluation_done'] = True
                             st.experimental_rerun() # Rerun to show proceed button

                         except Exception as e:
                              st.error(f"Error calculating or plotting metrics: {e}")

                     else:
                         st.error("Predictions could not be generated. Cannot evaluate.")


            # --- K-Means Clustering Evaluation ---
            elif model_type == "K-Means Clustering":
                st.info("K-Means evaluation often involves analyzing cluster characteristics and potentially using silhouette scores or the elbow method (relative measures).")

                if 'clusters' in st.session_state and st.session_state['clusters'] is not None:
                     st.write(f"**Number of Clusters Found (k):** {model.n_clusters}")
                     st.write(f"**Inertia (sum of squared distances to closest centroid):** {model.inertia_:.2f}")
                     st.markdown("---")

                     # Elbow Method Plot (Optional Calculation)
                     st.subheader("Elbow Method for Optimal k (Guideline)")
                     # Check if elbow data already calculated to avoid re-computation
                     if 'elbow_data' not in st.session_state:
                         st.session_state['elbow_data'] = None

                     if st.session_state['elbow_data'] is None:
                         if st.button("Calculate Elbow Curve (may take time)"):
                             with st.spinner("Calculating inertia for different k values..."):
                                if 'X_scaled' in st.session_state and st.session_state['X_scaled'] is not None:
                                    X_scaled_elbow = st.session_state['X_scaled']
                                    inertia_values = []
                                    k_range = range(2, 16) # Check k from 2 to 15
                                    k_list = list(k_range)
                                    try:
                                        for k_val in k_list:
                                            kmeans_elbow = KMeans(n_clusters=k_val, init='k-means++', n_init=10, max_iter=300, random_state=42)
                                            kmeans_elbow.fit(X_scaled_elbow)
                                            inertia_values.append(kmeans_elbow.inertia_)

                                        elbow_df = pd.DataFrame({'k': k_list, 'Inertia': inertia_values})
                                        st.session_state['elbow_data'] = elbow_df # Store calculated data
                                        st.success("Elbow curve data calculated.")
                                        st.experimental_rerun() # Rerun to display the plot below
                                    except Exception as e:
                                         st.error(f"Error calculating elbow curve: {e}")

                                else:
                                    st.warning("Scaled data not found. Cannot calculate elbow curve.")
                     else:
                          st.info("Elbow curve previously calculated.")
                          elbow_df = st.session_state['elbow_data']
                          fig_elbow = px.line(elbow_df, x='k', y='Inertia', title='Elbow Method for Optimal k', markers=True)
                          fig_elbow.update_layout(xaxis_title='Number of Clusters (k)', yaxis_title='Inertia (Within-Cluster Sum of Squares)')
                          st.plotly_chart(fig_elbow, use_container_width=True)
                          st.info("Look for the 'elbow' point where the rate of decrease in inertia slows down significantly. This suggests a reasonable k, but domain knowledge is also important.")
                          if st.button("Recalculate Elbow Curve"):
                               st.session_state['elbow_data'] = None
                               st.experimental_rerun()


                     # Note: Silhouette Score could be added here, but requires pairwise distances -> computationally expensive for large datasets.

                     st.session_state['evaluation_done'] = True
                     st.experimental_rerun() # Rerun to show proceed button
                else:
                     st.warning("Clustering results not found. Please train the K-Means model first in Stage 5.")

            # --- Proceed Button ---
            if st.session_state.get('evaluation_done'):
                if st.button("Proceed to Results Visualization"):
                    st.session_state['stage'] = 7
                    st.success("Evaluation complete! Moving to Results Visualization.")
                    st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# 7. Results Visualization
elif st.session_state['stage'] == 7:
    st.title("7️⃣ Results Visualization & Summary")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if not st.session_state.get('evaluation_done') or not st.session_state.get('model'):
             st.warning("Model evaluation not completed or model not found. Please complete Stage 6 first.")
        else:
            model_type = st.session_state['model_type']
            st.subheader(f"Visualizing Final Results for {model_type}")

            # --- Linear Regression Visualization ---
            if model_type == "Linear Regression":
                if st.session_state.get('predictions') is not None and st.session_state.get('y_test') is not None:
                    st.write("Revisiting key evaluation plots and model summary:")

                    # Metrics Summary
                    st.subheader("Performance Metrics")
                    metrics = st.session_state['metrics']
                    st.write(f"- **R² Score:** {metrics.get('R2', 'N/A'):.4f}")
                    st.write(f"- **Mean Squared Error (MSE):** {metrics.get('MSE', 'N/A'):.4f}")
                    st.write(f"- **Root Mean Squared Error (RMSE):** {metrics.get('RMSE', 'N/A'):.4f}")
                    st.markdown("---")


                    # Re-display Actual vs Predicted
                    y_test = st.session_state['y_test']
                    y_pred = st.session_state['predictions']
                    st.subheader("Actual vs. Predicted Values")
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions', marker=dict(color='#ff6e40', opacity=0.7)))
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig_pred.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal Line (y=x)', line=dict(color='#1e3d59', dash='dash')))
                    fig_pred.update_layout(xaxis_title='Actual Values', yaxis_title='Predicted Values', legend_title="Legend")
                    st.plotly_chart(fig_pred, use_container_width=True)


                    # Re-display Residuals Plot
                    st.subheader("Residuals vs. Predicted Values")
                    residuals = y_test - y_pred
                    fig_res = px.scatter(x=y_pred, y=residuals, title=None, labels={'x': 'Predicted Values', 'y': 'Residuals'}, opacity=0.7)
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res, use_container_width=True)

                    # Feature Importance / Coefficients
                    if 'model' in st.session_state:
                         model = st.session_state['model']
                         features = st.session_state.get('features', [])
                         if len(features) == len(model.coef_):
                            coef_df = pd.DataFrame({
                                    'Feature': features,
                                    'Coefficient': model.coef_
                                }).sort_values('Coefficient', key=abs, ascending=False) # Sort by absolute value for importance
                            st.subheader("Feature Importance (Coefficient Magnitude)")
                            fig_coef_imp = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Coefficients (Sorted by Absolute Value)', color='Coefficient', color_continuous_scale='RdBu')
                            st.plotly_chart(fig_coef_imp, use_container_width=True)
                            st.write(f"**Intercept:** {model.intercept_:.4f}")
                         else:
                            st.warning("Mismatch between number of features and model coefficients. Cannot display feature importance.")

                else:
                     st.warning("Prediction results not found. Please run Evaluation first.")

            # --- Logistic Regression Visualization ---
            elif model_type == "Logistic Regression":
                 if st.session_state.get('metrics') and st.session_state.get('y_test') is not None:
                    st.write("Revisiting key evaluation plots and model summary:")

                    # Metrics Summary
                    st.subheader("Performance Metrics")
                    metrics = st.session_state['metrics']
                    st.write(f"- **Accuracy:** {metrics.get('Accuracy', 'N/A'):.4f}")
                    st.write(f"- **AUC (Area Under ROC):** {metrics.get('AUC', 'N/A') if metrics.get('AUC') else 'N/A (Binary only)'}")
                    st.write(f"- **Average Precision:** {metrics.get('Avg Precision', 'N/A') if metrics.get('Avg Precision') else 'N/A (Binary only)'}")
                    st.markdown("---")


                    # Re-display Confusion Matrix
                    st.subheader("Confusion Matrix")
                    conf_matrix = metrics.get('Confusion Matrix')
                    if conf_matrix is not None:
                        y_test = st.session_state['y_test']
                        y_pred = st.session_state['predictions']
                        labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
                        if len(labels) == conf_matrix.shape[0]:
                            fig_conf = px.imshow(conf_matrix, text_auto=True, labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                                 x=[str(l) for l in labels], y=[str(l) for l in labels], color_continuous_scale='Blues')
                            fig_conf.update_xaxes(side="bottom", type='category')
                            fig_conf.update_yaxes(type='category')
                            st.plotly_chart(fig_conf, use_container_width=True)
                        else:
                            st.dataframe(pd.DataFrame(conf_matrix)) # Fallback
                    else:
                         st.info("Confusion Matrix not available.")

                    # Re-display ROC Curve (if available)
                    if 'AUC' in metrics: # Check if AUC was calculated (implies binary, probs available)
                         st.subheader("ROC Curve")
                         y_prob = st.session_state['probabilities']
                         fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                         roc_auc = metrics['AUC']
                         fig_roc = go.Figure()
                         fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.3f})', line=dict(color='#ff6e40')))
                         fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='#1e3d59', dash='dash')))
                         fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                         st.plotly_chart(fig_roc, use_container_width=True)


                    # Feature Importance / Coefficients
                    if 'model' in st.session_state:
                         model = st.session_state['model']
                         features = st.session_state.get('features', [])
                         n_classes = len(model.classes_)
                         coefs, intercept = None, None

                         if n_classes > 2:
                             coefs = model.coef_[1] # Show second class vs rest
                             intercept = model.intercept_[1]
                         elif n_classes == 2:
                             coefs = model.coef_[0]
                             intercept = model.intercept_[0]

                         if coefs is not None and len(features) == len(coefs):
                            coef_df = pd.DataFrame({
                                    'Feature': features,
                                    'Coefficient': coefs
                                }).sort_values('Coefficient', key=abs, ascending=False) # Sort by absolute value
                            st.subheader("Feature Importance (Coefficient Magnitude)")
                            fig_coef_imp = px.bar(coef_df, x='Feature', y='Coefficient', title=f'Feature Coefficients (Class {model.classes_[1] if n_classes>2 else model.classes_[-1]} vs Rest/Baseline)', color='Coefficient', color_continuous_scale='RdBu')
                            st.plotly_chart(fig_coef_imp, use_container_width=True)
                            st.write(f"**Intercept:** {intercept:.4f}")
                         elif coefs is not None:
                             st.warning("Mismatch between number of features and model coefficients. Cannot display feature importance.")


                 else:
                      st.warning("Evaluation metrics or test data not found. Please run Evaluation first.")


            # --- K-Means Clustering Visualization ---
            elif model_type == "K-Means Clustering":
                 if 'clusters' in st.session_state and st.session_state['clusters'] is not None and 'X_scaled' in st.session_state:
                     X_scaled = st.session_state['X_scaled']
                     clusters = st.session_state['clusters']
                     features = st.session_state['features']
                     n_features = X_scaled.shape[1]

                     st.subheader("Cluster Visualization")

                     # Dimensionality Reduction for Plotting (if needed)
                     plot_df = None
                     x_axis, y_axis = None, None
                     title = "Cluster Visualization"
                     plot_type = 'scatter' # Default

                     if n_features > 2:
                         st.write("Using PCA to reduce dimensions to 2D for visualization.")
                         try:
                             pca = PCA(n_components=2, random_state=42)
                             X_pca = pca.fit_transform(X_scaled)
                             plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                             x_axis, y_axis = 'PC1', 'PC2'
                             title = f"Clusters in PCA Space (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})"
                         except Exception as e:
                              st.error(f"PCA failed: {e}")
                              plot_df = None # Prevent plotting if PCA fails

                     elif n_features == 2:
                         plot_df = pd.DataFrame(X_scaled, columns=features)
                         x_axis, y_axis = features[0], features[1]
                         title = f"Clusters by Features: {features[0]} vs {features[1]} (Scaled)"
                     elif n_features == 1: # n_features == 1
                         plot_df = pd.DataFrame({'Feature': X_scaled[:, 0]})
                         x_axis = 'Feature'
                         y_axis = None # Use histogram for 1D
                         plot_type = 'histogram'
                         title = f"Cluster Distribution for Feature: {features[0]} (Scaled)"
                     else: # n_features == 0
                          st.warning("No features available for cluster visualization.")


                     # Perform plotting if data is ready
                     if plot_df is not None:
                         plot_df['Cluster'] = clusters.astype(str) # Convert cluster numbers to strings for discrete colors

                         # Scatter Plot (2D) or Histogram (1D)
                         if plot_type == 'scatter' and x_axis and y_axis:
                             fig_clusters = px.scatter(plot_df, x=x_axis, y=y_axis, color='Cluster', title=title,
                                                       color_discrete_sequence=px.colors.qualitative.Pastel)
                             fig_clusters.update_traces(marker=dict(size=8, opacity=0.8))
                             st.plotly_chart(fig_clusters, use_container_width=True)
                         elif plot_type == 'histogram' and x_axis:
                             fig_clusters = px.histogram(plot_df, x=x_axis, color='Cluster', title=title, barmode='overlay', opacity=0.7,
                                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                             st.plotly_chart(fig_clusters, use_container_width=True)


                     # Parallel Coordinates Plot (only if multiple features)
                     if n_features > 1:
                         st.subheader("Parallel Coordinates Plot of Clusters")
                         try:
                             # Create DataFrame with original scaled data and cluster labels
                             parallel_df = pd.DataFrame(X_scaled, columns=features)
                             parallel_df['Cluster'] = clusters.astype(str) # Use string for discrete colors

                             # Sample if too large to avoid browser freezing
                             max_points_parallel = 5000
                             if len(parallel_df) > max_points_parallel:
                                  parallel_df_sampled = parallel_df.sample(max_points_parallel, random_state=42)
                                  st.info(f"Sampled {max_points_parallel} points for parallel coordinates plot due to large dataset size.")
                             else:
                                 parallel_df_sampled = parallel_df

                             fig_parallel = px.parallel_coordinates(parallel_df_sampled, dimensions=features, color='Cluster',
                                                                  title="Feature Values Across Clusters (Scaled Data)",
                                                                  color_discrete_sequence=px.colors.qualitative.Pastel) # Use discrete colors
                             st.plotly_chart(fig_parallel, use_container_width=True)
                         except Exception as e:
                              st.warning(f"Could not generate parallel coordinates plot: {e}")

                     # Feature Distributions per Cluster
                     if n_features >= 1:
                         st.subheader("Feature Distributions by Cluster")
                         feat_dist_select = st.selectbox("Select feature to compare distributions:", features, key="dist_feat_cluster")

                         # Use original data (before scaling) for interpretability if scaler exists
                         dist_df = None
                         y_label = ""
                         scaler_kmeans = st.session_state.get('scaler') # Get the scaler used for Kmeans
                         X_original = None

                         if scaler_kmeans:
                             try:
                                X_original = scaler_kmeans.inverse_transform(X_scaled)
                             except Exception as e:
                                 st.warning(f"Could not inverse transform data: {e}. Showing scaled distributions.")

                         if X_original is not None:
                            dist_df = pd.DataFrame(X_original, columns=features)
                            y_label = f"{feat_dist_select} (Original Scale)"
                         else: # Fallback to scaled data
                            dist_df = pd.DataFrame(X_scaled, columns=features)
                            y_label = f"{feat_dist_select} (Scaled)"


                         dist_df['Cluster'] = clusters.astype(str)
                         fig_dist = px.box(dist_df, x='Cluster', y=feat_dist_select, color='Cluster', title=f"Distribution of {feat_dist_select} by Cluster",
                                           labels={'Cluster': 'Cluster', feat_dist_select: y_label}, color_discrete_sequence=px.colors.qualitative.Pastel)
                         st.plotly_chart(fig_dist, use_container_width=True)


                 else:
                      st.warning("Clustering results or scaled data not found. Please run K-Means Training and Evaluation first.")

            # --- End of Analysis ---
            st.markdown("---")
            st.success("Analysis Complete!")
            st.balloons()
            if st.button("Start New Analysis (Reset)"):
                reset_app()
                st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# Fallback for unexpected stage values
elif st.session_state['stage'] > 7 or st.session_state['stage'] < 0:
    st.warning("Invalid application state detected. Resetting.")
    reset_app()
    st.experimental_rerun()
