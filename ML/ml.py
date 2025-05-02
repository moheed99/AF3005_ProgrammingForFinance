import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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
from sklearn.manifold import TSNE
import base64
from datetime import datetime, timedelta
import io
import warnings

# Suppress warnings (especially normalization warnings for LinearRegression)
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Financial ML Application",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    </style>
    """, unsafe_allow_html=True)

add_custom_styling()

# Initialize session state variables if they don't exist
default_session_state = {
    'stage': 0,
    'data': None,
    'features': None,
    'target': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'X_scaled': None, # For clustering scaled data
    'scaler': None,   # Store the scaler object
    'model': None,
    'model_type': None,
    'predictions': None,
    'probabilities': None, # For Logistic Regression probabilities
    'metrics': {},
    'feature_importance': None,
    'processed_data': None,
    'processing_stats': {},
    'clusters': None,
    'data_source': None
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Function to reset the application state
def reset_app():
    for key in list(st.session_state.keys()):
        if key != 'stage':  # Keep the stage to show welcome page
            st.session_state[key] = default_session_state.get(key) # Reset to default
    st.session_state['stage'] = 0

# Function to display GIFs
def display_gif(gif_url, width=None):
    if width:
        st.markdown(f'<img src="{gif_url}" width="{width}px" style="display: block; margin-left: auto; margin-right: auto;">', unsafe_allow_html=True)
    else:
        st.markdown(f'<img src="{gif_url}" style="display: block; margin-left: auto; margin-right: auto;">', unsafe_allow_html=True)

# Function to get stock data
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
             return None, f"No data found for ticker {ticker}. It might be delisted or invalid."
        df.reset_index(inplace=True)
        # Convert Date column to datetime if it's not already
        if 'Date' in df.columns:
             df['Date'] = pd.to_datetime(df['Date'])
        return df, f"Successfully fetched data for {ticker}"
    except Exception as e:
        return None, f"Error fetching data for {ticker}: {e}"

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

    def update_stage():
        st.session_state['stage'] = stages.index(st.session_state.selectbox_stage) + 1

    st.sidebar.selectbox(
        "Current stage:",
        stages,
        index=current_stage_index,
        key='selectbox_stage',
        on_change=update_stage
    )

st.sidebar.markdown("---")

# Data Source Selection
st.sidebar.subheader("Data Source")
data_source_option = st.sidebar.radio(
    "Choose data source:",
    ["Upload Kragle Dataset", "Fetch from Yahoo Finance"],
    key='data_source_radio',
    index=0 if st.session_state['data_source'] == "Upload Kragle Dataset" else 1,
    on_change=lambda: setattr(st.session_state, 'data_source', st.session_state.data_source_radio) # Update session state immediately
)
st.session_state['data_source'] = data_source_option # Ensure it's set initially

# Yahoo Finance Options
if data_source_option == "Fetch from Yahoo Finance":
    stock_symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL):", "AAPL")
    time_period = st.sidebar.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=3)

    if st.sidebar.button("Fetch Stock Data"):
        if not stock_symbol:
            st.sidebar.error("Please enter a stock symbol.")
        else:
            with st.spinner(f"Fetching data for {stock_symbol}..."):
                data, message = get_stock_data(stock_symbol, time_period)
                if data is not None:
                    st.session_state['data'] = data
                    st.session_state['stage'] = 1 # Move to data loading stage
                    st.sidebar.success(message)
                    st.experimental_rerun() # Rerun to reflect the new stage and data
                else:
                    st.sidebar.error(message)

# Upload Dataset Option
if data_source_option == "Upload Kragle Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
    if uploaded_file is not None and st.session_state['data'] is None: # Only load if data isn't already set
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state['data'] = data
            st.session_state['stage'] = 1 # Move to data loading stage
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
    finance_gif_url = "https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif"
    display_gif(finance_gif_url, width=400)

    st.markdown("""
    ## About This Application
    This interactive machine learning application allows you to:

    *   **Load Data:** Upload financial datasets or fetch real-time stock market data.
    *   **Preprocess:** Handle missing values, detect and treat outliers, and scale features.
    *   **Feature Engineer:** Create new features from dates, technical indicators, or interactions.
    *   **Split Data:** Divide your data into training and testing sets.
    *   **Train Models:** Apply Linear Regression, Logistic Regression, or K-Means Clustering.
    *   **Evaluate:** Assess model performance using relevant metrics.
    *   **Visualize:** Interpret results through various plots and charts.

    ### Get Started
    Select a data source from the sidebar (Upload or Yahoo Finance) to begin your analysis.
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
            if potential_date_cols:
                main_date_col = potential_date_cols[0]
                try:
                    df_display[main_date_col] = pd.to_datetime(df_display[main_date_col])
                except Exception:
                    st.warning(f"Could not automatically convert column '{main_date_col}' to datetime. Plotting might be affected.")

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
                st.write(df_display.describe(include='all')) # Include descriptive stats for non-numeric too
            except Exception as e:
                st.error(f"Could not generate summary statistics: {e}")
                st.write(df_display.describe()) # Fallback to numeric only

            # Data Visualization
            st.subheader("Data Visualization")
            numeric_cols = df_display.select_dtypes(include=np.number).columns.tolist()
            date_cols_for_plot = df_display.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()

            if date_cols_for_plot and numeric_cols:
                date_col_plot = st.selectbox("Select date/time column for plotting:", date_cols_for_plot)
                numeric_col_plot = st.selectbox("Select numeric column for time series plot:", numeric_cols)
                if date_col_plot and numeric_col_plot:
                    try:
                        fig = px.line(df_display, x=date_col_plot, y=numeric_col_plot, title=f"{numeric_col_plot} Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting time series: {e}")

            # Correlation Heatmap
            if len(numeric_cols) > 1:
                st.subheader("Correlation Heatmap")
                numeric_data = df_display[numeric_cols]
                corr = numeric_data.corr()
                fig = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Matrix of Numeric Features")
                st.plotly_chart(fig, use_container_width=True)

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
                data_to_process = st.session_state['data'].copy()
            else:
                data_to_process = st.session_state['processed_data'].copy()

            st.subheader("Current Data State")
            st.dataframe(data_to_process.head())
            st.write(f"Shape: {data_to_process.shape}")

            # Ensure Date/Time columns are handled correctly
            for col in data_to_process.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        # Attempt conversion, but don't overwrite if it fails silently
                        pd.to_datetime(data_to_process[col], errors='ignore')
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
                    ["Drop rows with missing values",
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

                        if handling_method == "Drop rows with missing values":
                            processed_data.dropna(inplace=True)
                            st.info(f"Dropped {before_rows - len(processed_data)} rows.")
                        elif handling_method == "Fill numeric with Mean":
                            for col in processed_data.select_dtypes(include=np.number).columns:
                                if processed_data[col].isnull().any():
                                    processed_data[col].fillna(processed_data[col].mean(), inplace=True)
                            st.info("Filled numeric NaNs with column means.")
                        elif handling_method == "Fill numeric with Median":
                            for col in processed_data.select_dtypes(include=np.number).columns:
                                if processed_data[col].isnull().any():
                                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
                            st.info("Filled numeric NaNs with column medians.")
                        elif handling_method == "Fill numeric with Zero":
                            for col in processed_data.select_dtypes(include=np.number).columns:
                                if processed_data[col].isnull().any():
                                    processed_data[col].fillna(0, inplace=True)
                            st.info("Filled numeric NaNs with zero.")
                        elif handling_method == "Fill categorical with Mode":
                            for col in processed_data.select_dtypes(include='object').columns:
                                if processed_data[col].isnull().any():
                                    processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
                            st.info("Filled categorical NaNs with column modes.")
                        elif handling_method == "Skip Missing Value Handling":
                             st.info("Skipped missing value handling.")

                        st.session_state['processed_data'] = processed_data # Update session state
                        st.success("Missing value treatment applied (or skipped).")
                        st.experimental_rerun() # Rerun to reflect changes

            else:
                st.success("✅ No missing values found in the current dataset!")

            # --- Outlier Detection ---
            st.subheader("Outlier Detection & Handling")
            numeric_cols = data_to_process.select_dtypes(include=np.number).columns.tolist()

            if numeric_cols:
                selected_col_outlier = st.selectbox("Select column to analyze for outliers:", numeric_cols, key="outlier_col_select")

                col1, col2 = st.columns(2)
                with col1:
                    fig_box = px.box(data_to_process, y=selected_col_outlier, title=f"Box Plot for {selected_col_outlier}")
                    st.plotly_chart(fig_box, use_container_width=True)
                with col2:
                    fig_hist = px.histogram(data_to_process, x=selected_col_outlier, marginal="box", title=f"Distribution of {selected_col_outlier}")
                    st.plotly_chart(fig_hist, use_container_width=True)

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
                                min_val = processed_data[col].min()
                                shift = 1 - min_val if min_val <= 0 else 0 # Add 1 if min is 0 or less
                                processed_data[f"{col}_log"] = np.log(processed_data[col] + shift)
                                st.info(f"Applied log transform to {col} (added {shift:.2f} before log). New column: {col}_log")
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

            # --- Feature Scaling ---
            st.subheader("Feature Scaling")
            numeric_cols_scale = data_to_process.select_dtypes(include=np.number).columns.tolist()

            if numeric_cols_scale:
                scaling_method = st.selectbox(
                    "Select scaling method for numeric features:",
                    ["No scaling", "StandardScaler (Z-score)", "MinMaxScaler (0-1)"],
                    index=0 # Default to no scaling
                )

                if st.button("Apply Scaling"):
                    with st.spinner("Scaling features..."):
                        processed_data = data_to_process.copy()
                        scaler_obj = None

                        if scaling_method == "StandardScaler (Z-score)":
                            scaler = StandardScaler()
                            processed_data[numeric_cols_scale] = scaler.fit_transform(processed_data[numeric_cols_scale])
                            scaler_obj = scaler
                            st.info("Applied StandardScaler to numeric features.")
                        elif scaling_method == "MinMaxScaler (0-1)":
                            scaler = MinMaxScaler()
                            processed_data[numeric_cols_scale] = scaler.fit_transform(processed_data[numeric_cols_scale])
                            scaler_obj = scaler
                            st.info("Applied MinMaxScaler to numeric features.")
                        elif scaling_method == "No scaling":
                            st.info("No scaling applied.")

                        st.session_state['processed_data'] = processed_data # Update session state
                        st.session_state['scaler'] = scaler_obj # Store the scaler if applied
                        st.success("Scaling applied (or skipped).")
                        st.experimental_rerun() # Rerun to reflect changes
            else:
                 st.warning("No numeric columns found for scaling.")


            # --- Proceed Button ---
            st.markdown("---")
            st.subheader("Final Processed Data Preview")
            st.dataframe(st.session_state['processed_data'].head() if st.session_state['processed_data'] is not None else data_to_process.head())

            if st.button("Confirm Preprocessing & Proceed to Feature Engineering"):
                if st.session_state['processed_data'] is None:
                     st.session_state['processed_data'] = data_to_process # Ensure data is saved if no steps were applied
                st.session_state['stage'] = 3
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

            if st.session_state['model_type'] == "Linear Regression":
                possible_targets = numeric_cols
                if not possible_targets:
                    st.error("Linear Regression requires numeric columns. None found.")
                else:
                    target_col = st.selectbox(
                        "Select target variable (must be numeric):",
                        possible_targets,
                        index=possible_targets.index(target_col) if target_col and target_col in possible_targets else 0
                    )
                    st.session_state['target'] = target_col

            elif st.session_state['model_type'] == "Logistic Regression":
                # Allow selecting numeric or low-cardinality categorical
                possible_targets = numeric_cols + [col for col in data.columns if data[col].nunique() < 10 and col not in numeric_cols]
                if not possible_targets:
                    st.error("Logistic Regression requires a target variable. No suitable columns found.")
                else:
                    target_col = st.selectbox(
                        "Select target variable (numeric or low unique values):",
                        possible_targets,
                        index=possible_targets.index(target_col) if target_col and target_col in possible_targets else 0
                    )
                    st.session_state['target'] = target_col

                    # Binarization option if target is numeric or > 2 classes
                    if data[target_col].dtype in [np.number, 'float64', 'int64'] or data[target_col].nunique() > 2:
                         st.warning(f"Target '{target_col}' has {data[target_col].nunique()} unique values. Consider binarizing for Logistic Regression.")
                         if st.checkbox(f"Binarize target '{target_col}'?"):
                            if data[target_col].dtype in [np.number, 'float64', 'int64']:
                                threshold = st.slider("Select threshold:", float(data[target_col].min()), float(data[target_col].max()), float(data[target_col].median()))
                                if st.button("Apply Binarization"):
                                     data[f"{target_col}_binary"] = (data[target_col] > threshold).astype(int)
                                     st.session_state['target'] = f"{target_col}_binary" # Update target
                                     st.session_state['processed_data'] = data # Save changes
                                     st.success(f"Created binary target '{st.session_state['target']}' based on threshold {threshold:.2f}.")
                                     st.experimental_rerun()
                            else:
                                st.info("Binarization for non-numeric target not yet implemented.")


            elif st.session_state['model_type'] == "K-Means Clustering":
                st.info("K-Means is unsupervised and does not require a target variable.")
                st.session_state['target'] = None
                target_col = None # Ensure target_col is None for later logic

            # --- Feature Selection ---
            st.subheader("Feature Selection")
            available_features = data.select_dtypes(include=np.number).columns.tolist()
            if target_col and target_col in available_features:
                available_features.remove(target_col) # Don't use target as a feature

            if not available_features:
                st.warning("No numeric features available for selection (after excluding target).")
                selected_features = []
            else:
                st.write("Select features to use for the model:")
                 # Use multiselect for easier selection
                selected_features = st.multiselect(
                     "Features:",
                     available_features,
                     default=st.session_state.get('features', available_features) # Default to previous selection or all
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
                            X = data[selected_features].copy()
                            y = data[target_col].copy()
                            # Handle potential NaNs introduced during processing/feature creation
                            combined = pd.concat([X, y], axis=1).dropna()
                            X_clean = combined[selected_features]
                            y_clean = combined[target_col]

                            if len(X_clean) > 1: # Need at least 2 samples
                                with st.spinner("Calculating feature importance preview..."):
                                     if st.session_state['model_type'] == "Linear Regression":
                                          selector = SelectKBest(f_regression, k='all')
                                          selector.fit(X_clean, y_clean)
                                          scores = selector.scores_
                                          score_func_name = "F-Regression Score"
                                     elif st.session_state['model_type'] == "Logistic Regression":
                                          selector = SelectKBest(f_classif, k='all')
                                          selector.fit(X_clean, y_clean)
                                          scores = selector.scores_
                                          score_func_name = "F-Classification Score"
                                     else:
                                          scores = None # No importance for K-Means preview

                                     if scores is not None:
                                         importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': scores}).sort_values('Importance', ascending=False)
                                         st.session_state['feature_importance'] = importance_df # Store importance
                                         fig_imp = px.bar(importance_df, x='Feature', y='Importance', title=f'Feature Importance Preview ({score_func_name})', color='Importance', color_continuous_scale='Viridis')
                                         st.plotly_chart(fig_imp, use_container_width=True)
                            else:
                                 st.warning("Not enough data after cleaning NaNs to calculate feature importance.")

                        except Exception as e:
                            st.warning(f"Could not calculate feature importance preview: {e}")

                    st.experimental_rerun() # Rerun to update default multiselect if needed


            # --- Feature Creation Options ---
            st.subheader("Feature Creation (Optional)")

            # Date Features
            date_cols_create = data.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
            if date_cols_create:
                with st.expander("📅 Create Date-Based Features"):
                    selected_date_col_create = st.selectbox("Select date column:", date_cols_create, key="date_feat_col")
                    create_year = st.checkbox("Year")
                    create_month = st.checkbox("Month")
                    create_day = st.checkbox("Day")
                    create_dayofweek = st.checkbox("Day of Week")
                    create_quarter = st.checkbox("Quarter")

                    if st.button("Create Selected Date Features"):
                        with st.spinner("Creating date features..."):
                            new_data = data.copy() # Work on a copy
                            if create_year: new_data[f'{selected_date_col_create}_year'] = new_data[selected_date_col_create].dt.year
                            if create_month: new_data[f'{selected_date_col_create}_month'] = new_data[selected_date_col_create].dt.month
                            if create_day: new_data[f'{selected_date_col_create}_day'] = new_data[selected_date_col_create].dt.day
                            if create_dayofweek: new_data[f'{selected_date_col_create}_dayofweek'] = new_data[selected_date_col_create].dt.dayofweek
                            if create_quarter: new_data[f'{selected_date_col_create}_quarter'] = new_data[selected_date_col_create].dt.quarter
                            st.session_state['processed_data'] = new_data # Update data
                            st.success("Date features created.")
                            st.experimental_rerun()

            # Technical Indicators (if OHLCV data exists)
            req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            has_ohlcv = all(col in data.columns for col in req_cols)
            if has_ohlcv:
                with st.expander("📊 Create Technical Indicators"):
                    st.info("Requires 'Open', 'High', 'Low', 'Close', 'Volume' columns.")
                    create_ma = st.checkbox("Moving Averages (5, 20, 50)")
                    create_rsi = st.checkbox("RSI (14)")
                    create_bb = st.checkbox("Bollinger Bands (20, 2)")
                    create_macd = st.checkbox("MACD (12, 26, 9)")

                    if st.button("Create Selected Indicators"):
                        with st.spinner("Creating indicators..."):
                            new_data = data.copy() # Work on a copy
                            if create_ma:
                                for period in [5, 20, 50]: new_data[f'MA_{period}'] = new_data['Close'].rolling(window=period).mean()
                            if create_rsi:
                                delta = new_data['Close'].diff()
                                gain = delta.where(delta > 0, 0).fillna(0)
                                loss = -delta.where(delta < 0, 0).fillna(0)
                                avg_gain = gain.rolling(window=14).mean()
                                avg_loss = loss.rolling(window=14).mean()
                                rs = avg_gain / avg_loss
                                new_data['RSI_14'] = 100 - (100 / (1 + rs))
                            if create_bb:
                                ma20 = new_data['Close'].rolling(window=20).mean()
                                sd20 = new_data['Close'].rolling(window=20).std()
                                new_data['BB_Upper'] = ma20 + (sd20 * 2)
                                new_data['BB_Lower'] = ma20 - (sd20 * 2)
                            if create_macd:
                                ema12 = new_data['Close'].ewm(span=12, adjust=False).mean()
                                ema26 = new_data['Close'].ewm(span=26, adjust=False).mean()
                                new_data['MACD'] = ema12 - ema26
                                new_data['MACD_Signal'] = new_data['MACD'].ewm(span=9, adjust=False).mean()

                            st.session_state['processed_data'] = new_data.dropna() # Drop NaNs from rolling calculations
                            st.success("Technical indicators created (rows with NaNs dropped).")
                            st.experimental_rerun()

            # Interaction Features
            if len(numeric_cols) >= 2:
                 with st.expander("🔗 Create Interaction Features"):
                    col1_int, col2_int = st.columns(2)
                    with col1_int: feat1 = st.selectbox("Select first feature:", numeric_cols, key="interact_1")
                    with col2_int: feat2 = st.selectbox("Select second feature:", [c for c in numeric_cols if c != feat1], key="interact_2")

                    interaction_type = st.selectbox("Select interaction type:", ["Multiplication", "Division", "Addition", "Subtraction"])

                    if st.button("Create Interaction Feature"):
                        with st.spinner("Creating interaction..."):
                            new_data = data.copy()
                            if interaction_type == "Multiplication": new_data[f'{feat1}_x_{feat2}'] = new_data[feat1] * new_data[feat2]
                            elif interaction_type == "Addition": new_data[f'{feat1}_plus_{feat2}'] = new_data[feat1] + new_data[feat2]
                            elif interaction_type == "Subtraction": new_data[f'{feat1}_minus_{feat2}'] = new_data[feat1] - new_data[feat2]
                            elif interaction_type == "Division":
                                # Add small epsilon to prevent division by zero
                                new_data[f'{feat1}_div_{feat2}'] = new_data[feat1] / (new_data[feat2] + 1e-6)

                            st.session_state['processed_data'] = new_data
                            st.success(f"Created {interaction_type.lower()} feature between {feat1} and {feat2}.")
                            st.experimental_rerun()


            # --- Proceed Button ---
            st.markdown("---")
            if st.session_state.get('features'): # Only enable if features have been confirmed
                if st.button("Confirm Features & Proceed to Train/Test Split"):
                    # Final check for NaNs in selected features/target before proceeding
                    final_data = st.session_state['processed_data']
                    cols_to_check = st.session_state['features'] + ([st.session_state['target']] if st.session_state['target'] else [])
                    nan_counts = final_data[cols_to_check].isnull().sum()
                    if nan_counts.sum() > 0:
                        st.warning(f"Warning: {nan_counts.sum()} NaN values found in selected columns. Consider handling them in preprocessing.")
                        st.write(nan_counts[nan_counts > 0])
                        # Optionally add a button here to force dropna or go back
                    st.session_state['stage'] = 4
                    st.success("Feature selection/engineering confirmed! Moving to Train/Test Split.")
                    st.experimental_rerun()
            else:
                st.warning("Please confirm selected features using the 'Confirm Selected Features' button before proceeding.")

        else:
            st.warning("No processed data available. Please complete Stage 2 first.")

        st.markdown('</div>', unsafe_allow_html=True)


# 4. Train/Test Split
elif st.session_state['stage'] == 4:
    st.title("4️⃣ Train/Test Split")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state['processed_data'] is not None and st.session_state.get('features'):
            data = st.session_state['processed_data'].copy()
            features = st.session_state['features']
            target = st.session_state.get('target')

            # --- Supervised Learning Split ---
            if st.session_state['model_type'] in ["Linear Regression", "Logistic Regression"]:
                if not target:
                    st.error("Target variable not selected. Please go back to Stage 3.")
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
                    combined.dropna(subset=features + [target], inplace=True)
                    X = combined[features]
                    y = combined[target]
                    rows_after_na = len(X)

                    if rows_after_na < initial_rows:
                        st.warning(f"Removed {initial_rows - rows_after_na} rows due to NaN values in selected features or target.")

                    if len(X) < 2:
                         st.error("Not enough data remaining after handling NaNs to perform train/test split.")
                    else:
                        test_size = st.slider("Select test set size (%):", min_value=10, max_value=50, value=25, step=1) / 100
                        random_state = st.number_input("Random state (for reproducibility):", min_value=0, value=42)

                        if st.button("Split Data"):
                             with st.spinner("Splitting data..."):
                                try:
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y if st.session_state['model_type'] == "Logistic Regression" and y.nunique() < 10 else None) # Stratify for classification if few classes

                                    st.session_state['X_train'] = X_train
                                    st.session_state['X_test'] = X_test
                                    st.session_state['y_train'] = y_train
                                    st.session_state['y_test'] = y_test

                                    st.success(f"Data split: Train ({X_train.shape[0]} samples), Test ({X_test.shape[0]} samples)")

                                    # Visualize split
                                    split_df = pd.DataFrame({'Set': ['Train', 'Test'], 'Samples': [len(X_train), len(X_test)]})
                                    fig_pie = px.pie(split_df, values='Samples', names='Set', title='Train/Test Split Ratio', hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
                                    st.plotly_chart(fig_pie)

                                    # Proceed button enabled after successful split
                                    st.session_state['split_done'] = True

                                except ValueError as ve:
                                     st.error(f"Error during split: {ve}. This might happen if a class has too few samples for stratification.")
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during split: {e}")

            # --- Unsupervised Learning Preparation ---
            elif st.session_state['model_type'] == "K-Means Clustering":
                st.subheader("Prepare Data for Clustering")
                st.write(f"**Features:** {', '.join(features)}")

                X = data[features].copy()
                initial_rows = len(X)
                X.dropna(inplace=True) # Drop rows with any NaNs in selected features
                rows_after_na = len(X)

                if rows_after_na < initial_rows:
                     st.warning(f"Removed {initial_rows - rows_after_na} rows due to NaN values in selected features.")

                if len(X) < 2:
                     st.error("Not enough data remaining after handling NaNs for clustering.")
                else:
                    st.info("K-Means uses the full dataset (after NaN removal). No train/test split needed.")
                    # Store the prepared data (consider storing the scaled version later)
                    st.session_state['X_train'] = X # Using X_train key for consistency, represents full dataset for clustering
                    st.session_state['X_test'] = None
                    st.session_state['y_train'] = None
                    st.session_state['y_test'] = None
                    st.success(f"Data prepared for clustering: {X.shape[0]} samples, {X.shape[1]} features.")
                    st.session_state['split_done'] = True # Mark as ready to proceed

            # --- Proceed Button ---
            if st.session_state.get('split_done'):
                if st.button("Proceed to Model Training"):
                    st.session_state['stage'] = 5
                    st.success("Data ready! Moving to Model Training.")
                    st.experimental_rerun()

        elif not st.session_state.get('processed_data'):
            st.warning("Processed data not found. Please complete Stage 2 & 3.")
        elif not st.session_state.get('features'):
            st.warning("Features not selected. Please go back to Stage 3 and confirm features.")

        st.markdown('</div>', unsafe_allow_html=True)

# 5. Model Training
elif st.session_state['stage'] == 5:
    st.title("5️⃣ Model Training")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state.get('X_train') is not None:
            model_type = st.session_state['model_type']
            st.subheader(f"Train {model_type} Model")

            # --- Linear Regression Training ---
            if model_type == "Linear Regression":
                if st.session_state.get('y_train') is None:
                     st.error("Training data (y_train) not found. Please complete previous steps.")
                else:
                    # Parameters
                    # fit_intercept = st.checkbox("Fit intercept", value=True) # Always True for sklearn's default
                    # normalize = st.checkbox("Normalize (Deprecated)", value=False) # Deprecated, scaling should be done in preprocessing

                    if st.button("Train Linear Regression Model"):
                        with st.spinner("Training Linear Regression..."):
                            try:
                                X_train = st.session_state['X_train']
                                y_train = st.session_state['y_train']
                                model = LinearRegression() # Use defaults
                                model.fit(X_train, y_train)

                                st.session_state['model'] = model
                                st.success("Linear Regression model trained successfully!")

                                # Display Coefficients
                                coef_df = pd.DataFrame({
                                    'Feature': st.session_state['features'],
                                    'Coefficient': model.coef_
                                }).sort_values('Coefficient', ascending=False)
                                st.write("**Model Coefficients:**")
                                fig_coef = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Coefficients', color='Coefficient', color_continuous_scale='RdBu')
                                st.plotly_chart(fig_coef, use_container_width=True)
                                st.write(f"**Intercept:** {model.intercept_:.4f}")

                                st.session_state['training_done'] = True # Mark training as complete

                            except Exception as e:
                                st.error(f"Error training Linear Regression: {e}")


            # --- Logistic Regression Training ---
            elif model_type == "Logistic Regression":
                 if st.session_state.get('y_train') is None:
                     st.error("Training data (y_train) not found. Please complete previous steps.")
                 else:
                     # Parameters
                     solver = st.selectbox("Solver:", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"], index=0)
                     max_iter = st.slider("Maximum iterations:", min_value=100, max_value=1000, value=100, step=100)
                     C = st.slider("Regularization strength (C):", min_value=0.01, max_value=10.0, value=1.0, step=0.1, format="%.2f")

                     if st.button("Train Logistic Regression Model"):
                         with st.spinner("Training Logistic Regression..."):
                             try:
                                 X_train = st.session_state['X_train']
                                 y_train = st.session_state['y_train']
                                 model = LogisticRegression(solver=solver, max_iter=max_iter, C=C, random_state=42)
                                 model.fit(X_train, y_train)

                                 st.session_state['model'] = model
                                 st.success("Logistic Regression model trained successfully!")

                                 # Display Coefficients (handle multi-class if necessary)
                                 if len(model.classes_) > 2:
                                      st.info("Multi-class classification detected. Showing coefficients for the first class vs rest.")
                                      coefs = model.coef_[0]
                                 else:
                                      coefs = model.coef_[0]

                                 coef_df = pd.DataFrame({
                                     'Feature': st.session_state['features'],
                                     'Coefficient': coefs
                                 }).sort_values('Coefficient', ascending=False)
                                 st.write("**Model Coefficients:**")
                                 fig_coef = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Coefficients', color='Coefficient', color_continuous_scale='RdBu')
                                 st.plotly_chart(fig_coef, use_container_width=True)
                                 st.write(f"**Intercept:** {model.intercept_[0]:.4f}")

                                 st.session_state['training_done'] = True # Mark training as complete

                             except Exception as e:
                                 st.error(f"Error training Logistic Regression: {e}")


            # --- K-Means Clustering Training ---
            elif model_type == "K-Means Clustering":
                 # Parameters
                 n_clusters = st.slider("Number of clusters (k):", min_value=2, max_value=15, value=3, step=1)
                 init_method = st.selectbox("Initialization method:", ["k-means++", "random"], index=0)
                 n_init = st.slider("Number of initializations:", min_value=1, max_value=20, value=10, step=1)
                 max_iter_kmeans = st.slider("Maximum iterations:", min_value=100, max_value=1000, value=300, step=50, key="kmeans_maxiter")

                 if st.button("Train K-Means Clustering Model"):
                     with st.spinner("Running K-Means Clustering..."):
                         try:
                             X = st.session_state['X_train'] # This is the full dataset prepared earlier
                             # ALWAYS scale data for K-Means
                             scaler = StandardScaler()
                             X_scaled = scaler.fit_transform(X)
                             st.session_state['X_scaled'] = X_scaled # Save scaled data
                             st.session_state['scaler'] = scaler # Save the scaler

                             model = KMeans(
                                 n_clusters=n_clusters,
                                 init=init_method,
                                 n_init=n_init,
                                 max_iter=max_iter_kmeans,
                                 random_state=42
                             )
                             model.fit(X_scaled)

                             st.session_state['model'] = model
                             st.session_state['clusters'] = model.labels_ # Store cluster labels
                             st.success(f"K-Means clustering completed with {n_clusters} clusters!")

                             # Display Cluster Centers (original scale)
                             st.subheader("Cluster Centers (Original Scale)")
                             centers_scaled = model.cluster_centers_
                             try:
                                centers_original = scaler.inverse_transform(centers_scaled)
                                centers_df = pd.DataFrame(centers_original, columns=st.session_state['features'])
                                centers_df.index.name = 'Cluster'
                                st.dataframe(centers_df.style.format("{:.2f}"))
                             except Exception as e:
                                 st.warning(f"Could not inverse transform centers: {e}")
                                 st.write("Scaled Cluster Centers:")
                                 centers_df_scaled = pd.DataFrame(centers_scaled, columns=st.session_state['features'])
                                 centers_df_scaled.index.name = 'Cluster'
                                 st.dataframe(centers_df_scaled.style.format("{:.2f}"))


                             # Display Inertia
                             st.write(f"**Inertia (Within-cluster sum of squares):** {model.inertia_:.2f}")

                             # Display Cluster Sizes
                             cluster_sizes = pd.Series(model.labels_).value_counts().sort_index()
                             cluster_sizes.index.name = 'Cluster'
                             st.write("**Cluster Sizes:**")
                             st.dataframe(cluster_sizes)

                             st.session_state['training_done'] = True # Mark training as complete

                         except Exception as e:
                             st.error(f"Error running K-Means: {e}")

            # --- Proceed Button ---
            if st.session_state.get('training_done'):
                if st.button("Proceed to Evaluation"):
                    st.session_state['stage'] = 6
                    st.success("Model training complete! Moving to Evaluation.")
                    st.experimental_rerun()

        else:
            st.warning("Training data not found. Please complete the Train/Test Split stage first.")

        st.markdown('</div>', unsafe_allow_html=True)

# 6. Evaluation
elif st.session_state['stage'] == 6:
    st.title("6️⃣ Model Evaluation")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state.get('model'):
            model = st.session_state['model']
            model_type = st.session_state['model_type']
            st.subheader(f"Evaluation Metrics for {model_type}")

            # --- Linear Regression Evaluation ---
            if model_type == "Linear Regression":
                 if st.session_state.get('X_test') is not None and st.session_state.get('y_test') is not None:
                     X_test = st.session_state['X_test']
                     y_test = st.session_state['y_test']

                     if 'predictions' not in st.session_state or st.session_state['predictions'] is None:
                           st.session_state['predictions'] = model.predict(X_test)
                     y_pred = st.session_state['predictions']

                     # Calculate Metrics
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
                     fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal', line=dict(color='#1e3d59', dash='dash')))
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

                 else:
                     st.warning("Test data (X_test, y_test) not found. Cannot evaluate.")


            # --- Logistic Regression Evaluation ---
            elif model_type == "Logistic Regression":
                 if st.session_state.get('X_test') is not None and st.session_state.get('y_test') is not None:
                     X_test = st.session_state['X_test']
                     y_test = st.session_state['y_test']

                     if 'predictions' not in st.session_state or st.session_state['predictions'] is None:
                           st.session_state['predictions'] = model.predict(X_test)
                     y_pred = st.session_state['predictions']

                     if 'probabilities' not in st.session_state or st.session_state['probabilities'] is None:
                         if hasattr(model, "predict_proba"):
                            st.session_state['probabilities'] = model.predict_proba(X_test)
                         else:
                             st.warning("Model does not support probability predictions.")
                             st.session_state['probabilities'] = None
                     y_prob = st.session_state['probabilities']


                     # Calculate Metrics
                     accuracy = accuracy_score(y_test, y_pred)
                     conf_matrix = confusion_matrix(y_test, y_pred)
                     class_report = classification_report(y_test, y_pred, output_dict=True)
                     st.session_state['metrics'] = {'Accuracy': accuracy, 'Confusion Matrix': conf_matrix, 'Classification Report': class_report}

                     st.write(f"**Accuracy:** {accuracy:.4f}")
                     st.markdown("---")

                     # Confusion Matrix Plot
                     st.subheader("Confusion Matrix")
                     try:
                         labels = model.classes_
                         fig_conf = px.imshow(conf_matrix, text_auto=True, labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                              x=labels, y=labels, color_continuous_scale='Blues', title="Confusion Matrix")
                         fig_conf.update_xaxes(side="bottom")
                         st.plotly_chart(fig_conf, use_container_width=True)
                     except Exception as e:
                          st.error(f"Could not plot confusion matrix: {e}")
                          st.write(conf_matrix)


                     # Classification Report
                     st.subheader("Classification Report")
                     report_df = pd.DataFrame(class_report).transpose()
                     st.dataframe(report_df.style.format("{:.3f}"))
                     st.markdown("---")

                     # ROC Curve (if binary and probabilities available)
                     if y_prob is not None and len(model.classes_) == 2:
                         st.subheader("ROC Curve & AUC")
                         fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                         roc_auc = auc(fpr, tpr)

                         fig_roc = go.Figure()
                         fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.3f})', line=dict(color='#ff6e40')))
                         fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='#1e3d59', dash='dash')))
                         fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                         st.plotly_chart(fig_roc, use_container_width=True)

                     # Precision-Recall Curve (if binary and probabilities available)
                     if y_prob is not None and len(model.classes_) == 2:
                         st.subheader("Precision-Recall Curve")
                         precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
                         avg_precision = average_precision_score(y_test, y_prob[:, 1])

                         fig_pr = px.area(x=recall, y=precision, title=f'Precision-Recall Curve (Avg Precision = {avg_precision:.3f})',
                                          labels={'x': 'Recall', 'y': 'Precision'})
                         fig_pr.add_shape(type='line', x0=0, y0=y_test.mean(), x1=1, y1=y_test.mean(), line=dict(dash='dash')) # Baseline
                         fig_pr.update_yaxes(range=[0.0, 1.05])
                         fig_pr.update_xaxes(range=[0.0, 1.0])
                         st.plotly_chart(fig_pr, use_container_width=True)


                     st.session_state['evaluation_done'] = True
                 else:
                     st.warning("Test data (X_test, y_test) not found. Cannot evaluate.")


            # --- K-Means Clustering Evaluation ---
            elif model_type == "K-Means Clustering":
                st.info("K-Means evaluation often involves analyzing cluster characteristics and potentially using silhouette scores or the elbow method.")

                if 'clusters' in st.session_state and st.session_state['clusters'] is not None:
                     st.write(f"**Number of Clusters (k):** {model.n_clusters}")
                     st.write(f"**Inertia:** {model.inertia_:.2f}")
                     st.markdown("---")

                     # Elbow Method Plot (Optional Calculation)
                     st.subheader("Elbow Method for Optimal k")
                     if st.button("Calculate Elbow Curve (may take time)"):
                         with st.spinner("Calculating inertia for different k values..."):
                              if 'X_scaled' in st.session_state and st.session_state['X_scaled'] is not None:
                                  X_scaled = st.session_state['X_scaled']
                                  inertia_values = []
                                  k_range = range(2, 16) # Check k from 2 to 15
                                  for k_val in k_range:
                                      kmeans_elbow = KMeans(n_clusters=k_val, init='k-means++', n_init=10, max_iter=300, random_state=42)
                                      kmeans_elbow.fit(X_scaled)
                                      inertia_values.append(kmeans_elbow.inertia_)

                                  elbow_df = pd.DataFrame({'k': k_range, 'Inertia': inertia_values})
                                  fig_elbow = px.line(elbow_df, x='k', y='Inertia', title='Elbow Method for Optimal k', markers=True)
                                  fig_elbow.update_layout(xaxis_title='Number of Clusters (k)', yaxis_title='Inertia (Within-Cluster Sum of Squares)')
                                  st.plotly_chart(fig_elbow, use_container_width=True)
                                  st.info("Look for the 'elbow' point where the rate of decrease in inertia slows down significantly.")
                              else:
                                   st.warning("Scaled data not found. Cannot calculate elbow curve.")

                     st.session_state['evaluation_done'] = True
                else:
                     st.warning("Clustering results not found. Please train the K-Means model first.")

            # --- Proceed Button ---
            if st.session_state.get('evaluation_done'):
                if st.button("Proceed to Results Visualization"):
                    st.session_state['stage'] = 7
                    st.success("Evaluation complete! Moving to Results Visualization.")
                    st.experimental_rerun()

        else:
            st.warning("Model not found. Please train a model in Stage 5 first.")

        st.markdown('</div>', unsafe_allow_html=True)


# 7. Results Visualization
elif st.session_state['stage'] == 7:
    st.title("7️⃣ Results Visualization")
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state.get('model'):
            model_type = st.session_state['model_type']
            st.subheader(f"Visualizing Results for {model_type}")

            # --- Linear Regression Visualization ---
            if model_type == "Linear Regression":
                if 'predictions' in st.session_state and st.session_state['predictions'] is not None:
                    st.write("Revisiting key evaluation plots:")
                    # Re-display Actual vs Predicted
                    y_test = st.session_state['y_test']
                    y_pred = st.session_state['predictions']
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions', marker=dict(color='#ff6e40', opacity=0.7)))
                    fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal', line=dict(color='#1e3d59', dash='dash')))
                    fig_pred.update_layout(title='Actual vs. Predicted Values', xaxis_title='Actual Values', yaxis_title='Predicted Values', legend_title="Legend")
                    st.plotly_chart(fig_pred, use_container_width=True)

                    # Re-display Residuals Plot
                    residuals = y_test - y_pred
                    fig_res = px.scatter(x=y_pred, y=residuals, title='Residuals vs. Predicted Values', labels={'x': 'Predicted Values', 'y': 'Residuals'}, opacity=0.7)
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res, use_container_width=True)

                    # Feature Importance / Coefficients
                    if 'model' in st.session_state:
                         model = st.session_state['model']
                         coef_df = pd.DataFrame({
                                'Feature': st.session_state['features'],
                                'Coefficient': model.coef_
                            }).sort_values('Coefficient', key=abs, ascending=False) # Sort by absolute value for importance
                         st.subheader("Feature Importance (Coefficient Magnitude)")
                         fig_coef_imp = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Coefficients (Sorted by Absolute Value)', color='Coefficient', color_continuous_scale='RdBu')
                         st.plotly_chart(fig_coef_imp, use_container_width=True)


                else:
                     st.warning("Prediction results not found. Please run Evaluation first.")

            # --- Logistic Regression Visualization ---
            elif model_type == "Logistic Regression":
                 if 'metrics' in st.session_state and st.session_state['metrics']:
                    st.write("Revisiting key evaluation plots:")
                    # Re-display Confusion Matrix
                    conf_matrix = st.session_state['metrics'].get('Confusion Matrix')
                    if conf_matrix is not None:
                        labels = st.session_state['model'].classes_
                        fig_conf = px.imshow(conf_matrix, text_auto=True, labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                             x=labels, y=labels, color_continuous_scale='Blues', title="Confusion Matrix")
                        fig_conf.update_xaxes(side="bottom")
                        st.plotly_chart(fig_conf, use_container_width=True)

                    # Re-display ROC Curve (if available)
                    if 'probabilities' in st.session_state and st.session_state['probabilities'] is not None and len(st.session_state['model'].classes_) == 2:
                         y_test = st.session_state['y_test']
                         y_prob = st.session_state['probabilities']
                         fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                         roc_auc = auc(fpr, tpr)
                         fig_roc = go.Figure()
                         fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.3f})', line=dict(color='#ff6e40')))
                         fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='#1e3d59', dash='dash')))
                         fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                         st.plotly_chart(fig_roc, use_container_width=True)
                    elif len(st.session_state['model'].classes_) != 2:
                         st.info("ROC curve is typically shown for binary classification.")

                    # Feature Importance / Coefficients
                    if 'model' in st.session_state:
                         model = st.session_state['model']
                         if len(model.classes_) > 2: coefs = model.coef_[0] # Show first class for multi-class
                         else: coefs = model.coef_[0]
                         coef_df = pd.DataFrame({
                                'Feature': st.session_state['features'],
                                'Coefficient': coefs
                            }).sort_values('Coefficient', key=abs, ascending=False) # Sort by absolute value
                         st.subheader("Feature Importance (Coefficient Magnitude)")
                         fig_coef_imp = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Coefficients (Sorted by Absolute Value)', color='Coefficient', color_continuous_scale='RdBu')
                         st.plotly_chart(fig_coef_imp, use_container_width=True)

                 else:
                      st.warning("Evaluation metrics not found. Please run Evaluation first.")


            # --- K-Means Clustering Visualization ---
            elif model_type == "K-Means Clustering":
                 if 'clusters' in st.session_state and st.session_state['clusters'] is not None and 'X_scaled' in st.session_state:
                     X_scaled = st.session_state['X_scaled']
                     clusters = st.session_state['clusters']
                     features = st.session_state['features']
                     n_features = X_scaled.shape[1]

                     st.subheader("Cluster Visualization")

                     # Dimensionality Reduction for Plotting
                     if n_features > 2:
                         st.write("Using PCA to reduce dimensions for visualization.")
                         pca = PCA(n_components=2, random_state=42)
                         X_pca = pca.fit_transform(X_scaled)
                         plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                         plot_df['Cluster'] = clusters.astype(str) # Convert cluster numbers to strings for discrete colors
                         x_axis, y_axis = 'PC1', 'PC2'
                         title = "Clusters in PCA Space (2 Components)"
                     elif n_features == 2:
                         plot_df = pd.DataFrame(X_scaled, columns=features)
                         plot_df['Cluster'] = clusters.astype(str)
                         x_axis, y_axis = features[0], features[1]
                         title = f"Clusters by Features: {features[0]} vs {features[1]}"
                     else: # n_features == 1
                         plot_df = pd.DataFrame({'Feature': X_scaled[:, 0], 'Cluster': clusters.astype(str)})
                         x_axis, y_axis = 'Feature', None # Use histogram for 1D
                         title = f"Cluster Distribution for Feature: {features[0]}"


                     # Scatter Plot (or Histogram for 1D)
                     if y_axis: # 2D plot
                         fig_clusters = px.scatter(plot_df, x=x_axis, y=y_axis, color='Cluster', title=title,
                                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                         fig_clusters.update_traces(marker=dict(size=8, opacity=0.8))
                         st.plotly_chart(fig_clusters, use_container_width=True)
                     else: # 1D plot
                         fig_clusters = px.histogram(plot_df, x=x_axis, color='Cluster', title=title, barmode='overlay', opacity=0.7,
                                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                         st.plotly_chart(fig_clusters, use_container_width=True)


                     # Parallel Coordinates Plot
                     if n_features > 1:
                         st.subheader("Parallel Coordinates Plot of Clusters")
                         try:
                             # Create DataFrame with original scaled data and cluster labels
                             parallel_df = pd.DataFrame(X_scaled, columns=features)
                             parallel_df['Cluster'] = clusters.astype(str)
                             # Sample if too large to avoid browser freezing
                             if len(parallel_df) > 5000:
                                  parallel_df = parallel_df.sample(5000, random_state=42)
                                  st.info("Sampled 5000 points for parallel coordinates plot due to large dataset size.")

                             fig_parallel = px.parallel_coordinates(parallel_df, dimensions=features, color='Cluster',
                                                                  title="Feature Values Across Clusters (Scaled Data)",
                                                                  color_continuous_scale=px.colors.sequential.Viridis, # Although cluster is categorical, sometimes helps visualize density
                                                                  color_continuous_midpoint=np.median(parallel_df['Cluster'].astype(int))) # Requires numeric cluster label
                             st.plotly_chart(fig_parallel, use_container_width=True)
                         except Exception as e:
                              st.warning(f"Could not generate parallel coordinates plot: {e}")

                     # Feature Distributions per Cluster
                     st.subheader("Feature Distributions by Cluster")
                     feat_dist_select = st.selectbox("Select feature to compare distributions:", features, key="dist_feat_cluster")
                     # Use original data (before scaling) for interpretability if scaler exists
                     if 'scaler' in st.session_state and st.session_state['scaler'] is not None:
                         try:
                            X_original = st.session_state['scaler'].inverse_transform(X_scaled)
                            dist_df = pd.DataFrame(X_original, columns=features)
                            y_label = f"{feat_dist_select} (Original Scale)"
                         except: # Fallback to scaled if inverse transform fails
                             dist_df = pd.DataFrame(X_scaled, columns=features)
                             y_label = f"{feat_dist_select} (Scaled)"
                     else: # Use scaled data if no scaler
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

        else:
            st.warning("No model results found. Please complete the previous stages.")

        st.markdown('</div>', unsafe_allow_html=True)

# Fallback for unexpected stage values
elif st.session_state['stage'] > 7:
    st.warning("Invalid application state. Resetting.")
    reset_app()
    st.experimental_rerun()