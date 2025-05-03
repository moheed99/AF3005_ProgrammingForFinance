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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, silhouette_score # Added silhouette_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE # TSNE was imported but not used, commented out
import base64
from datetime import datetime, timedelta
import io
import warnings
import traceback # For better error reporting

# Suppress warnings (use with caution)
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
    h1, h2, h3, h4, h5, h6 { /* Target all headers */
        color: #1e3d59;
        font-weight: bold; /* Make headers bolder */
    }
    .step-container {
        background-color: white;
        padding: 25px; /* Increased padding */
        border-radius: 10px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1); /* Enhanced shadow */
        margin-bottom: 25px; /* Increased margin */
        border: 1px solid #e0e0e0; /* Subtle border */
    }
    /* Style selectbox labels */
    label[data-baseweb="select"] {
        color: #1e3d59 !important;
        font-weight: bold !important;
    }
    /* Style slider labels */
     label[data-testid="stWidgetLabel"] > div:first-child {
        color: #1e3d59 !important;
        font-weight: bold !important;
    }
    /* Style text input labels */
    label[data-baseweb="input"] {
         color: #1e3d59 !important;
         font-weight: bold !important;
    }
     /* Style checkbox labels */
    label[data-baseweb="checkbox"] span:first-child {
          color: #1e3d59 !important;
         font-weight: bold !important;
    }
     /* Style radio labels */
     label[data-baseweb="radio"] span:first-child {
          color: #1e3d59 !important;
         font-weight: bold !important;
    }
    /* Add some breathing room after subheaders */
    h3 {
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
     h4 {
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        color: #ff6e40; /* Accent color for H4 */
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
    'X_train_scaled': None, # Specifically for scaled training data
    'X_test_scaled': None,  # Specifically for scaled test data
    'X_scaled': None, # For clustering scaled data (full dataset)
    'scaler': None,   # Store the scaler object
    'model': None,
    'model_type': "Linear Regression", # Default model type
    'predictions': None,
    'probabilities': None, # For Logistic Regression probabilities
    'metrics': {},
    'feature_importance': None,
    'processed_data': None,
    'processing_stats': {},
    'clusters': None,
    'data_source': "Upload Kragle Dataset", # Default data source
    'split_done': False, # Flag for stage 4
    'training_done': False, # Flag for stage 5
    'evaluation_done': False, # Flag for stage 6
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Function to reset the application state
def reset_app():
    # Preserve stage 0 for welcome page
    current_stage = st.session_state.stage
    # Clear all keys
    for key in list(st.session_state.keys()):
        # Optionally keep sidebar state or other persistent settings
        # if key not in ['sidebar_state', 'theme']:
        del st.session_state[key]
    # Re-initialize with defaults
    for key, value in default_session_state.items():
        st.session_state[key] = value
    st.session_state['stage'] = 0 # Ensure stage is reset to 0
    st.success("Application reset to initial state.")


# Function to display GIFs
def display_gif(gif_url, width=None):
    if width:
        st.markdown(f'<img src="{gif_url}" width="{width}px" style="display: block; margin-left: auto; margin-right: auto; border-radius: 10px;">', unsafe_allow_html=True)
    else:
        st.markdown(f'<img src="{gif_url}" style="display: block; margin-left: auto; margin-right: auto; border-radius: 10px;">', unsafe_allow_html=True)

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
            try:
                # Try converting, coercing errors to NaT
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                # Optional: drop rows where date conversion failed
                # original_len = len(df)
                # df.dropna(subset=['Date'], inplace=True)
                # if len(df) < original_len:
                #     st.warning(f"Dropped {original_len - len(df)} rows due to invalid date formats.")
            except Exception as date_e:
                 return None, f"Error converting Date column for {ticker}: {date_e}"
        return df, f"Successfully fetched data for {ticker}"
    except Exception as e:
        return None, f"Error fetching data for {ticker}: {e}"

# Function to create a download link for dataframe
def get_download_link(df, filename, text):
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        # Enhanced styling for download button
        href = f"""
        <a href="data:file/csv;base64,{b64}" download="{filename}"
           style="display: inline-block;
                  padding: 8px 15px;
                  background-color: #1e3d59; /* Sidebar color */
                  color: white;
                  font-weight: bold;
                  border-radius: 5px;
                  text-decoration: none;
                  border: none;
                  transition: background-color 0.3s ease;">
           ⬇️ {text}
        </a>
        """
        return href
    except Exception as e:
        st.error(f"Error creating download link: {e}")
        return "Download link unavailable."

# --- Sidebar ---
st.sidebar.title("Financial ML Tool 💹")
st.sidebar.markdown("Follow the steps to analyze your data.")
st.sidebar.markdown("---")

# Data Source Selection
st.sidebar.subheader("1. Data Source")
data_source_option = st.sidebar.radio(
    "Choose data source:",
    ["Upload Kragle Dataset", "Fetch from Yahoo Finance"],
    key='data_source_radio',
    index=["Upload Kragle Dataset", "Fetch from Yahoo Finance"].index(st.session_state.data_source), # Use current state for index
    # on_change removed, button press will handle logic
)
# Update state immediately if radio changes without button press needed for other widgets
if data_source_option != st.session_state.data_source:
     st.session_state.data_source = data_source_option
     # Reset data if source changes? Maybe not, allow comparing sources.
     # st.session_state.data = None
     st.rerun()


# Yahoo Finance Options
if st.session_state.data_source == "Fetch from Yahoo Finance":
    stock_symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL):", "AAPL", key="stock_symbol_input")
    time_period = st.sidebar.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=3, key="stock_period_select")

    if st.sidebar.button("Fetch Stock Data"):
        if not stock_symbol:
            st.sidebar.error("Please enter a stock symbol.")
        else:
            with st.spinner(f"Fetching data for {stock_symbol}..."):
                # Clear previous data before fetching new
                reset_app() # Reset everything except stage
                st.session_state.data_source = "Fetch from Yahoo Finance" # Keep data source
                data, message = get_stock_data(stock_symbol, time_period)
                if data is not None:
                    st.session_state.data = data
                    st.session_state.stage = 1 # Move to data loading stage
                    st.sidebar.success(message)
                    st.rerun() # Rerun to reflect the new stage and data
                else:
                    st.sidebar.error(message)

# Upload Dataset Option
if st.session_state.data_source == "Upload Kragle Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'], key="csv_uploader")
    if uploaded_file is not None:
         if st.sidebar.button("Load Uploaded Data"):
             # Clear previous data before loading new
            reset_app() # Reset everything except stage
            st.session_state.data_source = "Upload Kragle Dataset" # Keep data source
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.session_state.stage = 1 # Move to data loading stage
                st.sidebar.success(f"Successfully loaded {uploaded_file.name}")
                st.rerun() # Rerun to reflect the new stage and data
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                st.session_state.data = None # Ensure data is None on error

st.sidebar.markdown("---")

# Model Selection (only if data is loaded)
if st.session_state.stage >= 1:
    st.sidebar.subheader("2. Model Selection")
    model_options = ["Linear Regression", "Logistic Regression", "K-Means Clustering"]

    def update_model_type():
        # Reset downstream steps if model type changes
        if st.session_state.selectbox_model != st.session_state.model_type:
            keys_to_reset = ['features', 'target', 'X_train', 'X_test', 'y_train', 'y_test',
                             'X_train_scaled', 'X_test_scaled', 'X_scaled', 'scaler', 'model',
                             'predictions', 'probabilities', 'metrics', 'feature_importance',
                             'clusters', 'split_done', 'training_done', 'evaluation_done']
            for key in keys_to_reset:
                st.session_state[key] = default_session_state.get(key)
            # Decide which stage to revert to (e.g., Feature Engineering)
            st.session_state.stage = min(st.session_state.stage, 3)
        st.session_state.model_type = st.session_state.selectbox_model


    st.sidebar.selectbox(
        "Choose a model:",
        model_options,
        key='selectbox_model',
        index=model_options.index(st.session_state.model_type), # Use current state for index
        on_change=update_model_type
    )

# Reset Button
st.sidebar.markdown("---")
if st.sidebar.button("⚠️ Reset Application"):
    reset_app()
    st.rerun()

# --- Main Application Logic ---

# Welcome Page
if st.session_state.stage == 0:
    st.title("Welcome to the Financial ML Application! 📈")
    finance_gif_url = "https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif" # Original GIF
    # Alternative: https://media.giphy.com/media/3o7abAuA4JWCAsymvK/giphy.gif (Trading screen)
    display_gif(finance_gif_url, width=400)

    st.markdown("""
    ## About This Application
    This interactive machine learning application allows you to step through a typical data analysis workflow:

    *   **Load Data:** Upload financial datasets or fetch real-time stock market data from Yahoo Finance.
    *   **Preprocess:** Handle missing values, detect potential outliers, and scale features.
    *   **Feature Engineer:** Select relevant features and optionally create new ones (e.g., date components, technical indicators).
    *   **Split Data:** Divide your data into training and testing sets for supervised learning.
    *   **Train Models:** Apply Linear Regression, Logistic Regression, or K-Means Clustering.
    *   **Evaluate:** Assess model performance using relevant metrics (MSE, R², Accuracy, Confusion Matrix, Silhouette Score, etc.).
    *   **Visualize:** Interpret results through various plots and charts (Actual vs. Predicted, Residuals, ROC curves, Cluster plots).

    ### Get Started
    Select a data source from the sidebar (**Upload** or **Yahoo Finance**) and click the corresponding button to load data and begin your analysis journey.
    """)
    st.info("ℹ️ Use the sidebar to load data and choose your desired machine learning model.")

# 1. Load Data
elif st.session_state.stage == 1:
    st.title("1️⃣ Load & Explore Data")
    with st.container(border=False):
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state.data is not None:
            df_display = st.session_state.data.copy()

            # Ensure Date column is datetime
            potential_date_cols = [col for col in df_display.columns if 'date' in col.lower() or 'time' in col.lower()]
            main_date_col = None
            if potential_date_cols:
                main_date_col = potential_date_cols[0]
                if not pd.api.types.is_datetime64_any_dtype(df_display[main_date_col]):
                     try:
                        df_display[main_date_col] = pd.to_datetime(df_display[main_date_col], errors='coerce')
                     except Exception:
                        st.warning(f"Could not convert column '{main_date_col}' to datetime for display.")
                        main_date_col = None # Prevent using it if conversion failed

            st.subheader("Data Preview")
            st.dataframe(df_display.head())

            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Rows", df_display.shape[0])
            with col2:
                st.metric("Number of Columns", df_display.shape[1])

            with st.expander("Detailed Column Info"):
                buffer = io.StringIO()
                df_display.info(buf=buffer)
                st.text(buffer.getvalue())

            st.subheader("Summary Statistics")
            try:
                # Attempt to include non-numeric if practical
                if df_display.shape[1] < 50:
                    st.dataframe(df_display.describe(include='all', datetime_is_numeric=True))
                else:
                    st.dataframe(df_display.describe(datetime_is_numeric=True))
            except Exception as e:
                st.error(f"Could not generate summary statistics: {e}")
                st.dataframe(df_display.describe(datetime_is_numeric=True)) # Fallback

            # Data Visualization
            st.subheader("Initial Data Visualization")
            numeric_cols = df_display.select_dtypes(include=np.number).columns.tolist()

            if main_date_col and numeric_cols:
                st.markdown("#### Time Series Plot")
                default_numeric = 'Close' if 'Close' in numeric_cols else numeric_cols[0]
                numeric_col_plot = st.selectbox("Select numeric column for time series plot:", numeric_cols, index=numeric_cols.index(default_numeric) if default_numeric in numeric_cols else 0)
                try:
                    fig = px.line(df_display, x=main_date_col, y=numeric_col_plot, title=f"{numeric_col_plot} Over Time")
                    fig.update_layout(xaxis_title='Date', yaxis_title=numeric_col_plot)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting time series: {e}")
            elif numeric_cols:
                 st.info("No clear date column found for time series plotting.")


            if len(numeric_cols) > 1:
                st.markdown("#### Correlation Heatmap")
                numeric_data = df_display[numeric_cols]
                # Handle potential NaN values before calculating correlation
                numeric_data_cleaned = numeric_data.dropna()
                if len(numeric_data_cleaned) > 1:
                    corr = numeric_data_cleaned.corr()
                    fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Matrix of Numeric Features")
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("Not enough non-NaN data to calculate correlations.")


            st.markdown("---")
            # Download Button
            st.markdown(get_download_link(st.session_state.data, 'loaded_financial_data.csv', 'Download Raw Data'), unsafe_allow_html=True)

            st.markdown("---")
            if st.button("➡️ Proceed to Data Preprocessing"):
                # Ensure processed_data starts fresh or from raw data if skipping preprocessing steps later
                st.session_state.processed_data = st.session_state.data.copy()
                st.session_state.stage = 2
                st.success("Moving to data preprocessing!")
                st.rerun()
        else:
            st.warning("No data loaded. Please select a data source from the sidebar and load the data.")

        st.markdown('</div>', unsafe_allow_html=True)


# 2. Preprocessing
elif st.session_state.stage == 2:
    st.title("2️⃣ Data Preprocessing")
    with st.container(border=False):
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        # Always work on the 'processed_data' state from this point onwards
        if st.session_state.processed_data is not None:
            data_to_process = st.session_state.processed_data.copy()

            st.subheader("Current Data State")
            st.dataframe(data_to_process.head())
            st.write(f"Shape: {data_to_process.shape}")

            # --- Missing Values ---
            st.markdown("---")
            st.subheader("Missing Values Analysis")
            missing_values = data_to_process.isnull().sum()
            missing_percent = (missing_values / len(data_to_process)) * 100
            missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage': missing_percent})
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False)

            if not missing_df.empty:
                st.write("Columns with Missing Values:")
                st.dataframe(missing_df.style.format({'Percentage': '{:.2f}%'}))
                try:
                    fig = px.bar(missing_df.reset_index(), x='index', y='Percentage', title='Missing Values by Column (%)', labels={'index':'Column'}, color='Percentage', color_continuous_scale='Oranges')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot missing values: {e}")

                st.markdown("#### Handle Missing Values")
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
                        processed_data_mv = data_to_process.copy() # Work on a copy for this step
                        rows_before = len(processed_data_mv)

                        try:
                            if handling_method == "Drop rows with any missing values":
                                processed_data_mv.dropna(inplace=True)
                                st.info(f"Dropped {rows_before - len(processed_data_mv)} rows.")
                            elif handling_method == "Fill numeric with Mean":
                                for col in processed_data_mv.select_dtypes(include=np.number).columns:
                                    if processed_data_mv[col].isnull().any():
                                        processed_data_mv[col].fillna(processed_data_mv[col].mean(), inplace=True)
                                st.info("Filled numeric NaNs with column means.")
                            elif handling_method == "Fill numeric with Median":
                                for col in processed_data_mv.select_dtypes(include=np.number).columns:
                                    if processed_data_mv[col].isnull().any():
                                        processed_data_mv[col].fillna(processed_data_mv[col].median(), inplace=True)
                                st.info("Filled numeric NaNs with column medians.")
                            elif handling_method == "Fill numeric with Zero":
                                for col in processed_data_mv.select_dtypes(include=np.number).columns:
                                    if processed_data_mv[col].isnull().any():
                                        processed_data_mv[col].fillna(0, inplace=True)
                                st.info("Filled numeric NaNs with zero.")
                            elif handling_method == "Fill categorical with Mode":
                                for col in processed_data_mv.select_dtypes(include='object').columns:
                                    if processed_data_mv[col].isnull().any():
                                        # Ensure mode() returns a value, handle case where column is all NaN
                                        mode_val = processed_data_mv[col].mode()
                                        if not mode_val.empty:
                                            processed_data_mv[col].fillna(mode_val[0], inplace=True)
                                        else:
                                             st.warning(f"Column '{col}' seems to be all NaN, cannot fill with mode.")
                                st.info("Filled categorical NaNs with column modes (if applicable).")
                            elif handling_method == "Skip Missing Value Handling":
                                 st.info("Skipped missing value handling.")

                            st.session_state.processed_data = processed_data_mv # Update session state
                            st.success("Missing value treatment applied (or skipped). Data updated.")
                            st.rerun() # Rerun to reflect changes

                        except Exception as e:
                            st.error(f"Error during missing value treatment: {e}")
                            traceback.print_exc()


            else:
                st.success("✅ No missing values found in the current dataset!")

            # --- Outlier Detection ---
            st.markdown("---")
            st.subheader("Outlier Detection & Handling")
            numeric_cols = data_to_process.select_dtypes(include=np.number).columns.tolist()

            if numeric_cols:
                selected_col_outlier = st.selectbox("Select column to analyze for outliers:", numeric_cols, key="outlier_col_select")

                col1, col2 = st.columns(2)
                with col1:
                    try:
                        fig_box = px.box(data_to_process, y=selected_col_outlier, title=f"Box Plot for {selected_col_outlier}", points='outliers')
                        st.plotly_chart(fig_box, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate box plot: {e}")
                with col2:
                    try:
                        fig_hist = px.histogram(data_to_process[selected_col_outlier].dropna(), x=selected_col_outlier, marginal="box", title=f"Distribution of {selected_col_outlier}")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate histogram: {e}")

                st.markdown(f"#### Handle Outliers in '{selected_col_outlier}'")
                outlier_method = st.selectbox(
                    f"Select method:",
                    ["No treatment",
                     "Remove outliers (IQR method)",
                     "Cap/Winsorize outliers (IQR method)",
                     "Log Transform (handle non-positive)"],
                    index=0 # Default to no treatment
                )

                if st.button("Apply Outlier Treatment"):
                    with st.spinner(f"Applying treatment to {selected_col_outlier}..."):
                        processed_data_ot = data_to_process.copy() # Work on a copy for this step
                        rows_before = len(processed_data_ot)
                        col = selected_col_outlier
                        col_log_created = False

                        try:
                            if processed_data_ot[col].isnull().all():
                                st.warning(f"Column '{col}' is all NaN. Skipping outlier treatment.")
                            elif outlier_method == "Remove outliers (IQR method)":
                                Q1 = processed_data_ot[col].quantile(0.25)
                                Q3 = processed_data_ot[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                processed_data_ot = processed_data_ot[(processed_data_ot[col] >= lower_bound) & (processed_data_ot[col] <= upper_bound)]
                                st.info(f"Removed {rows_before - len(processed_data_ot)} outlier rows based on {col}.")
                            elif outlier_method == "Cap/Winsorize outliers (IQR method)":
                                Q1 = processed_data_ot[col].quantile(0.25)
                                Q3 = processed_data_ot[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                outliers_low = (processed_data_ot[col] < lower_bound).sum()
                                outliers_high = (processed_data_ot[col] > upper_bound).sum()
                                processed_data_ot[col] = processed_data_ot[col].clip(lower=lower_bound, upper=upper_bound)
                                st.info(f"Capped {outliers_low + outliers_high} outlier values in {col}.")
                            elif outlier_method == "Log Transform (handle non-positive)":
                                new_col_name = f"{col}_log"
                                if (processed_data_ot[col] <= 0).any():
                                    min_val = processed_data_ot[col].min()
                                    shift = 1 - min_val if min_val <= 0 else 0 # Add 1 if min is 0 or less
                                    processed_data_ot[new_col_name] = np.log(processed_data_ot[col] + shift)
                                    st.info(f"Applied log transform to {col} (added {shift:.2f} before log). New column: {new_col_name}")
                                else:
                                    processed_data_ot[new_col_name] = np.log(processed_data_ot[col])
                                    st.info(f"Applied log transform to {col}. New column: {new_col_name}")
                                col_log_created = True
                            elif outlier_method == "No treatment":
                                 st.info("No outlier treatment applied for this column.")

                            st.session_state.processed_data = processed_data_ot # Update session state
                            st.success(f"Outlier treatment for '{col}' applied (or skipped). Data updated.")
                            st.rerun() # Rerun to reflect changes

                        except Exception as e:
                            st.error(f"Error during outlier treatment for '{col}': {e}")
                            traceback.print_exc()

            else:
                st.warning("No numeric columns found for outlier analysis.")

            # --- Feature Scaling ---
            st.markdown("---")
            st.subheader("Feature Scaling")
            # Use updated numeric columns list after potential log transform
            current_numeric_cols = st.session_state.processed_data.select_dtypes(include=np.number).columns.tolist()

            if current_numeric_cols:
                scaling_method = st.selectbox(
                    "Select scaling method for ALL numeric features:",
                    ["No scaling", "StandardScaler (Z-score)", "MinMaxScaler (0-1)"],
                    index=0 if st.session_state.scaler is None else (1 if isinstance(st.session_state.scaler, StandardScaler) else 2), # Reflect current state if possible
                    key="scaling_method_select"
                )

                if st.button("Apply Scaling"):
                    with st.spinner("Scaling features..."):
                        processed_data_sc = data_to_process.copy() # Work on a copy
                        scaler_obj = None
                        try:
                            if scaling_method == "StandardScaler (Z-score)":
                                scaler = StandardScaler()
                                processed_data_sc[current_numeric_cols] = scaler.fit_transform(processed_data_sc[current_numeric_cols])
                                scaler_obj = scaler
                                st.info("Applied StandardScaler to all numeric features.")
                            elif scaling_method == "MinMaxScaler (0-1)":
                                scaler = MinMaxScaler()
                                processed_data_sc[current_numeric_cols] = scaler.fit_transform(processed_data_sc[current_numeric_cols])
                                scaler_obj = scaler
                                st.info("Applied MinMaxScaler to all numeric features.")
                            elif scaling_method == "No scaling":
                                scaler_obj = None # Explicitly set scaler to None
                                st.info("No scaling applied (or existing scaling removed).")

                            st.session_state.processed_data = processed_data_sc # Update session state
                            st.session_state.scaler = scaler_obj # Store the scaler or None
                            st.success("Scaling applied (or skipped). Data updated.")
                            st.rerun() # Rerun to reflect changes
                        except Exception as e:
                            st.error(f"Error applying scaling: {e}")
                            traceback.print_exc()

            else:
                 st.warning("No numeric columns found for scaling.")


            # --- Proceed Button ---
            st.markdown("---")
            st.subheader("Review Processed Data")
            st.dataframe(st.session_state.processed_data.head())
            st.write(f"Current Shape: {st.session_state.processed_data.shape}")
            if st.session_state.scaler:
                st.info(f"Current scaling method: {type(st.session_state.scaler).__name__}")
            else:
                 st.info("Current scaling method: None")


            if st.button("➡️ Confirm Preprocessing & Proceed to Feature Engineering"):
                st.session_state.stage = 3
                st.success("Preprocessing steps confirmed! Moving to Feature Engineering.")
                st.rerun()
        else:
            st.warning("No data available. Please load data first in Stage 1.")

        st.markdown('</div>', unsafe_allow_html=True)


# 3. Feature Engineering
elif st.session_state.stage == 3:
    st.title("3️⃣ Feature Engineering & Selection")
    with st.container(border=False):
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state.processed_data is not None:
            data = st.session_state.processed_data.copy() # Work with the latest processed data

            st.subheader("Current Data State")
            st.dataframe(data.head())
            st.write(f"Shape: {data.shape}")

            st.markdown("---")
            # --- Target Variable Selection (for Supervised Learning) ---
            st.subheader("Target Variable Selection")
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            all_cols = data.columns.tolist()
            current_target = st.session_state.get('target') # Get existing target if set

            if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                if st.session_state.model_type == "Linear Regression":
                    possible_targets = numeric_cols
                    target_prompt = "Select target variable (must be numeric):"
                    if not possible_targets:
                        st.error("Linear Regression requires numeric columns. None found in the current data.")
                        st.stop()
                else: # Logistic Regression
                    # Include low-cardinality non-numeric columns
                    possible_targets = numeric_cols + [col for col in data.columns if data[col].nunique() < 15 and col not in numeric_cols]
                    target_prompt = "Select target variable (numeric or low unique values):"
                    if not possible_targets:
                        st.error("Logistic Regression requires a target variable. No suitable columns found.")
                        st.stop()

                # Use selectbox to choose target
                selected_target = st.selectbox(
                    target_prompt,
                    possible_targets,
                    index=possible_targets.index(current_target) if current_target and current_target in possible_targets else 0,
                    key="target_select"
                )
                # Update session state if changed
                if selected_target != current_target:
                    st.session_state.target = selected_target
                    st.rerun() # Rerun to update subsequent logic based on new target

                # --- Binarization Option (for Logistic Regression) ---
                if st.session_state.model_type == "Logistic Regression":
                    target_col = st.session_state.target
                    target_dtype = data[target_col].dtype
                    target_nunique = data[target_col].nunique()

                    if pd.api.types.is_numeric_dtype(target_dtype) or target_nunique > 2:
                        with st.expander(f"Optional: Binarize Target '{target_col}'"):
                            st.warning(f"Target '{target_col}' ({target_dtype}, {target_nunique} unique values) is not binary. Binarization is recommended for Logistic Regression.")
                            if pd.api.types.is_numeric_dtype(target_dtype):
                                threshold = st.slider("Select threshold:", float(data[target_col].min()), float(data[target_col].max()), float(data[target_col].median()), key="binarize_thresh")
                                if st.button("Apply Binarization"):
                                    try:
                                        new_target_name = f"{target_col}_binary"
                                        # Use the current 'data' DataFrame
                                        data[new_target_name] = (data[target_col] > threshold).astype(int)
                                        st.session_state.processed_data = data # Save changes
                                        st.session_state.target = new_target_name # Update target in session state
                                        st.success(f"Created binary target '{st.session_state.target}' based on threshold {threshold:.2f}.")
                                        st.rerun()
                                    except Exception as e:
                                         st.error(f"Error during binarization: {e}")
                            else:
                                # Simple binarization for categorical: one vs rest (choose most frequent as 1?)
                                st.info("Binarization for multi-class categorical target not yet implemented. Model will attempt multi-class classification.")

            elif st.session_state.model_type == "K-Means Clustering":
                st.info("ℹ️ K-Means is unsupervised and does not require a target variable.")
                st.session_state.target = None # Ensure target is None

            st.markdown("---")
            # --- Feature Selection ---
            st.subheader("Feature Selection")
            target_col = st.session_state.get('target') # Get current target
            available_features = data.select_dtypes(include=np.number).columns.tolist()
            if target_col and target_col in available_features:
                try:
                    available_features.remove(target_col) # Don't use target as a feature
                except ValueError:
                    pass # Target might have been transformed (e.g., _binary)

            if not available_features:
                st.error("No numeric features available for selection (after excluding target). Cannot proceed.")
                st.stop()
            else:
                st.markdown("**Select features to use for the model:**")
                # Use multiselect for easier selection
                # Ensure default features are valid and exist in the current available features
                current_selection = st.session_state.get('features', [])
                valid_default = [f for f in current_selection if f in available_features]
                if not valid_default: # If previous selection is invalid or empty, default to all
                     valid_default = available_features

                selected_features = st.multiselect(
                     "Features:",
                     available_features,
                     default=valid_default,
                     key="feature_multiselect"
                )

            if st.button("🔍 Confirm Selected Features & Preview Importance"):
                if not selected_features:
                    st.error("Please select at least one feature.")
                else:
                    st.session_state.features = selected_features
                    st.success(f"Confirmed {len(selected_features)} features.")

                    # Calculate feature importance preview if applicable
                    if target_col and st.session_state.model_type != "K-Means Clustering":
                        try:
                            X_preview = data[selected_features].copy()
                            y_preview = data[target_col].copy()
                            # Handle potential NaNs before fitting selector
                            combined_preview = pd.concat([X_preview, y_preview], axis=1).dropna()
                            X_preview_clean = combined_preview[selected_features]
                            y_preview_clean = combined_preview[target_col]

                            if len(X_preview_clean) > 1 and len(y_preview_clean.unique()) > 1: # Need samples and variation
                                with st.spinner("Calculating feature importance preview..."):
                                     score_func = f_regression if st.session_state.model_type == "Linear Regression" else f_classif
                                     score_func_name = "F-Regression" if st.session_state.model_type == "Linear Regression" else "F-Classification (ANOVA)"
                                     selector = SelectKBest(score_func, k='all')
                                     selector.fit(X_preview_clean, y_preview_clean)
                                     scores = selector.scores_

                                     importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': scores}).sort_values('Importance', ascending=False)
                                     st.session_state.feature_importance = importance_df # Store importance
                                     fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title=f'Feature Importance Preview ({score_func_name})', color='Importance', color_continuous_scale='Viridis')
                                     st.plotly_chart(fig_imp, use_container_width=True)
                            else:
                                 st.warning("Not enough valid data or target variation to calculate feature importance preview.")
                        except Exception as e:
                            st.warning(f"Could not calculate feature importance preview: {e}")
                            traceback.print_exc()
                    # Do not rerun here, let user proceed manually after seeing importance
                    # st.rerun()

            # --- Feature Creation Options ---
            st.markdown("---")
            st.subheader("Feature Creation (Optional)")

            # Date Features
            date_cols_create = data.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
            if date_cols_create:
                with st.expander("📅 Create Date-Based Features"):
                    selected_date_col_create = st.selectbox("Select date column:", date_cols_create, key="date_feat_col")
                    col_dt1, col_dt2, col_dt3 = st.columns(3)
                    with col_dt1: create_year = st.checkbox("Year", key="dt_year")
                    with col_dt2: create_month = st.checkbox("Month", key="dt_month")
                    with col_dt3: create_day = st.checkbox("Day", key="dt_day")
                    with col_dt1: create_dayofweek = st.checkbox("Day of Week", key="dt_dow")
                    with col_dt2: create_quarter = st.checkbox("Quarter", key="dt_quarter")
                    # Add more time-based features if needed (e.g., week of year, day of year)

                    if st.button("Create Selected Date Features"):
                        with st.spinner("Creating date features..."):
                            new_data_fc = data.copy() # Work on a copy
                            base_col = new_data_fc[selected_date_col_create]
                            created_count = 0
                            if create_year: new_data_fc[f'{selected_date_col_create}_year'] = base_col.dt.year; created_count+=1
                            if create_month: new_data_fc[f'{selected_date_col_create}_month'] = base_col.dt.month; created_count+=1
                            if create_day: new_data_fc[f'{selected_date_col_create}_day'] = base_col.dt.day; created_count+=1
                            if create_dayofweek: new_data_fc[f'{selected_date_col_create}_dayofweek'] = base_col.dt.dayofweek; created_count+=1
                            if create_quarter: new_data_fc[f'{selected_date_col_create}_quarter'] = base_col.dt.quarter; created_count+=1

                            if created_count > 0:
                                st.session_state.processed_data = new_data_fc # Update data
                                st.success(f"{created_count} date features created.")
                                st.rerun()
                            else:
                                st.info("No date features selected to create.")

            # Technical Indicators (if OHLCV data exists)
            req_cols = ['Open', 'High', 'Low', 'Close'] # Volume is common but not strictly needed for all
            has_ohlc = all(col in data.columns for col in req_cols)
            has_volume = 'Volume' in data.columns
            if has_ohlc:
                with st.expander("📊 Create Technical Indicators"):
                    st.info("Requires 'Open', 'High', 'Low', 'Close' columns. 'Volume' needed for some.")
                    col_ti1, col_ti2, col_ti3 = st.columns(3)
                    with col_ti1: create_ma = st.checkbox("Moving Avg (5, 20)", key="ti_ma")
                    with col_ti2: create_rsi = st.checkbox("RSI (14)", key="ti_rsi")
                    with col_ti3: create_bb = st.checkbox("Bollinger Bands (20, 2)", key="ti_bb")
                    with col_ti1: create_macd = st.checkbox("MACD (12, 26, 9)", key="ti_macd")
                    with col_ti2: create_vol_change = st.checkbox("Volume % Change", key="ti_vol", disabled=not has_volume)


                    if st.button("Create Selected Indicators"):
                        with st.spinner("Creating indicators..."):
                            new_data_ti = data.copy() # Work on a copy
                            created_count = 0
                            if create_ma:
                                for period in [5, 20]: new_data_ti[f'MA_{period}'] = new_data_ti['Close'].rolling(window=period).mean(); created_count+=1
                            if create_rsi:
                                delta = new_data_ti['Close'].diff()
                                gain = delta.where(delta > 0, 0).fillna(0)
                                loss = -delta.where(delta < 0, 0).fillna(0)
                                avg_gain = gain.rolling(window=14).mean()
                                avg_loss = loss.rolling(window=14).mean()
                                rs = avg_gain / avg_loss
                                new_data_ti['RSI_14'] = 100 - (100 / (1 + rs)); created_count+=1
                            if create_bb:
                                ma20 = new_data_ti['Close'].rolling(window=20).mean()
                                sd20 = new_data_ti['Close'].rolling(window=20).std()
                                new_data_ti['BB_Upper'] = ma20 + (sd20 * 2)
                                new_data_ti['BB_Lower'] = ma20 - (sd20 * 2); created_count+=2
                            if create_macd:
                                ema12 = new_data_ti['Close'].ewm(span=12, adjust=False).mean()
                                ema26 = new_data_ti['Close'].ewm(span=26, adjust=False).mean()
                                new_data_ti['MACD'] = ema12 - ema26
                                new_data_ti['MACD_Signal'] = new_data_ti['MACD'].ewm(span=9, adjust=False).mean(); created_count+=2
                            if create_vol_change and has_volume:
                                 new_data_ti['Volume_PctChange'] = new_data_ti['Volume'].pct_change() * 100; created_count+=1

                            if created_count > 0:
                                # Drop NaNs introduced by rolling calculations
                                rows_before = len(new_data_ti)
                                new_data_ti.dropna(inplace=True)
                                rows_after = len(new_data_ti)
                                st.session_state.processed_data = new_data_ti # Update data
                                st.success(f"{created_count} technical indicator(s) created.")
                                if rows_before > rows_after:
                                    st.info(f"Dropped {rows_before - rows_after} rows due to NaNs from indicator calculations.")
                                st.rerun()
                            else:
                                st.info("No indicators selected to create.")

            # Interaction Features
            current_numeric_cols_int = data.select_dtypes(include=np.number).columns.tolist()
            if target_col and target_col in current_numeric_cols_int:
                 try: current_numeric_cols_int.remove(target_col)
                 except ValueError: pass # Ignore if target not in list

            if len(current_numeric_cols_int) >= 2:
                 with st.expander("🔗 Create Interaction Features"):
                    col1_int, col2_int = st.columns(2)
                    with col1_int: feat1 = st.selectbox("Select first feature:", current_numeric_cols_int, key="interact_1")
                    # Ensure feat2 list doesn't include feat1
                    feat2_options = [c for c in current_numeric_cols_int if c != feat1]
                    if not feat2_options:
                        st.warning("Need at least two distinct numeric features for interaction.")
                    else:
                        with col2_int: feat2 = st.selectbox("Select second feature:", feat2_options, key="interact_2")

                        interaction_type = st.selectbox("Select interaction type:", ["Multiplication (*)", "Division (/)", "Addition (+)", "Subtraction (-)"])

                        if st.button("Create Interaction Feature"):
                            with st.spinner("Creating interaction..."):
                                new_data_int = data.copy()
                                new_feat_name = ""
                                try:
                                    if interaction_type == "Multiplication (*)": new_feat_name = f'{feat1}_x_{feat2}'; new_data_int[new_feat_name] = new_data_int[feat1] * new_data_int[feat2]
                                    elif interaction_type == "Addition (+)": new_feat_name = f'{feat1}_plus_{feat2}'; new_data_int[new_feat_name] = new_data_int[feat1] + new_data_int[feat2]
                                    elif interaction_type == "Subtraction (-)": new_feat_name = f'{feat1}_minus_{feat2}'; new_data_int[new_feat_name] = new_data_int[feat1] - new_data_int[feat2]
                                    elif interaction_type == "Division (/)":
                                        new_feat_name = f'{feat1}_div_{feat2}';
                                        # Add small epsilon only if denominator can be zero
                                        if (new_data_int[feat2] == 0).any():
                                            new_data_int[new_feat_name] = new_data_int[feat1] / (new_data_int[feat2] + 1e-9)
                                            st.info("Added small epsilon to denominator to avoid division by zero.")
                                        else:
                                            new_data_int[new_feat_name] = new_data_int[feat1] / new_data_int[feat2]

                                    st.session_state.processed_data = new_data_int
                                    st.success(f"Created '{new_feat_name}' feature.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error creating interaction feature: {e}")


            # --- Proceed Button ---
            st.markdown("---")
            if st.session_state.get('features'): # Only enable if features have been confirmed
                st.success(f"Ready to proceed with {len(st.session_state['features'])} selected features.")
                if st.button("➡️ Confirm Features & Proceed to Train/Test Split"):
                    # Final check for NaNs in selected features/target before proceeding
                    final_data = st.session_state.processed_data
                    cols_to_check = list(st.session_state.features) # Make a copy
                    if st.session_state.target:
                        cols_to_check.append(st.session_state.target)

                    # Ensure all columns to check actually exist in the dataframe
                    cols_to_check = [col for col in cols_to_check if col in final_data.columns]

                    if not cols_to_check:
                         st.error("Error: No valid features or target columns found to check for NaNs.")
                    else:
                        nan_counts = final_data[cols_to_check].isnull().sum()
                        if nan_counts.sum() > 0:
                            st.warning(f"Warning: {nan_counts.sum()} NaN values found in selected columns. These rows will be dropped before splitting/training.")
                            st.write("NaN counts per column:")
                            st.dataframe(nan_counts[nan_counts > 0])
                            # Consider adding an explicit dropna confirmation here if needed

                        st.session_state.stage = 4
                        st.session_state.split_done = False # Reset flag for next stage
                        st.success("Feature selection/engineering confirmed! Moving to Train/Test Split.")
                        st.rerun()
            else:
                st.warning("Please confirm selected features using the 'Confirm Selected Features' button before proceeding.")

        else:
            st.warning("No processed data available. Please complete Stage 2 first.")

        st.markdown('</div>', unsafe_allow_html=True)


# 4. Train/Test Split
elif st.session_state.stage == 4:
    st.title("4️⃣ Train/Test Split")
    with st.container(border=False):
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state.processed_data is not None and st.session_state.get('features'):
            data = st.session_state.processed_data.copy()
            features = st.session_state.features
            target = st.session_state.get('target')

            # --- Supervised Learning Split ---
            if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                if not target:
                    st.error("Target variable not selected. Please go back to Stage 3.")
                    st.stop()
                elif target not in data.columns:
                     st.error(f"Target variable '{target}' not found in the current processed data. Please revisit Stage 3.")
                     st.stop()
                else:
                    st.subheader("Split Data for Supervised Learning")
                    st.markdown(f"**Selected Features:** `{', '.join(features)}`")
                    st.markdown(f"**Selected Target:** `{target}`")

                    # Prepare X and y, explicitly handling potential NaNs
                    X = data[features].copy()
                    y = data[target].copy()
                    initial_rows = len(X)

                    # Drop rows where target or any feature is NaN *before* splitting
                    combined = pd.concat([X, y], axis=1)
                    rows_before_na = len(combined)
                    combined.dropna(subset=features + [target], inplace=True)
                    rows_after_na = len(combined)

                    if rows_after_na < rows_before_na:
                        st.warning(f"Removed {rows_before_na - rows_after_na} rows due to NaN values in selected features or target before splitting.")

                    if rows_after_na < 2:
                         st.error("Not enough data remaining after handling NaNs (need at least 2 rows) to perform train/test split.")
                         st.stop()
                    else:
                        X = combined[features]
                        y = combined[target]

                        test_size = st.slider("Select test set size (%):", min_value=10, max_value=50, value=25, step=1, key="test_size_slider") / 100.0
                        random_state = st.number_input("Random state (for reproducibility):", min_value=0, value=42, step=1, key="split_random_state")
                        stratify_option = None
                        if st.session_state.model_type == "Logistic Regression" and y.nunique() > 1:
                            # Check if stratification is feasible (at least 2 samples per class in smallest set)
                            min_samples_needed = 2 # Need at least 1 for train, 1 for test per class
                            class_counts = y.value_counts()
                            smallest_class_count = class_counts.min()
                            # Estimate test set size for smallest class
                            approx_test_smallest_count = int(np.ceil(smallest_class_count * test_size))
                            if smallest_class_count >= min_samples_needed and approx_test_smallest_count >=1 :
                                stratify_option = y
                                st.info("Stratifying split by target variable to maintain class proportions.")
                            else:
                                st.warning(f"Cannot stratify: Smallest class has {smallest_class_count} samples. Need at least {min_samples_needed} total and approx 1 in test set. Performing non-stratified split.")


                        if st.button("Split Data"):
                             with st.spinner("Splitting data..."):
                                try:
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y,
                                        test_size=test_size,
                                        random_state=int(random_state),
                                        stratify=stratify_option # Use determined stratify option
                                    )

                                    st.session_state.X_train = X_train
                                    st.session_state.X_test = X_test
                                    st.session_state.y_train = y_train
                                    st.session_state.y_test = y_test
                                    # Reset scaled data from previous runs if any
                                    st.session_state.X_train_scaled = None
                                    st.session_state.X_test_scaled = None

                                    st.success(f"Data split successfully!")
                                    col_split1, col_split2 = st.columns(2)
                                    col_split1.metric("Training Samples", X_train.shape[0])
                                    col_split2.metric("Testing Samples", X_test.shape[0])


                                    # Visualize split
                                    split_df = pd.DataFrame({'Set': ['Train', 'Test'], 'Samples': [len(X_train), len(X_test)]})
                                    fig_pie = px.pie(split_df, values='Samples', names='Set', title='Train/Test Split Ratio', hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel1)
                                    st.plotly_chart(fig_pie, use_container_width=True)

                                    # Show preview of split data
                                    with st.expander("Preview Split Data"):
                                         st.write("Train Features (Head):"); st.dataframe(X_train.head(3))
                                         st.write("Test Features (Head):"); st.dataframe(X_test.head(3))
                                         st.write("Train Target (Head):"); st.dataframe(y_train.head(3))
                                         st.write("Test Target (Head):"); st.dataframe(y_test.head(3))


                                    st.session_state.split_done = True # Mark as ready to proceed

                                except ValueError as ve:
                                     st.error(f"Error during split: {ve}. This might happen if a class has too few samples for stratification or other data issues.")
                                     traceback.print_exc()
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during split: {e}")
                                    traceback.print_exc()

            # --- Unsupervised Learning Preparation ---
            elif st.session_state.model_type == "K-Means Clustering":
                st.subheader("Prepare Data for Clustering")
                st.markdown(f"**Selected Features:** `{', '.join(features)}`")

                X = data[features].copy()
                initial_rows = len(X)
                X.dropna(inplace=True) # Drop rows with any NaNs in selected features
                rows_after_na = len(X)

                if rows_after_na < initial_rows:
                     st.warning(f"Removed {initial_rows - rows_after_na} rows due to NaN values in selected features.")

                if rows_after_na < 2: # Need at least 2 samples for clustering
                     st.error("Not enough data remaining after handling NaNs for clustering.")
                     st.stop()
                else:
                    st.info("ℹ️ K-Means uses the full dataset (after NaN removal). No train/test split needed. Scaling will be applied before training.")
                    # Store the prepared data (unscaled version)
                    st.session_state.X_train = X # Using X_train key for consistency, represents full dataset
                    st.session_state.X_test = None # No test set for unsupervised
                    st.session_state.y_train = None
                    st.session_state.y_test = None
                    st.session_state.X_scaled = None # Reset scaled data

                    st.success(f"Data prepared for clustering: {X.shape[0]} samples, {X.shape[1]} features.")
                    st.session_state.split_done = True # Mark as ready to proceed

            # --- Proceed Button ---
            st.markdown("---")
            if st.session_state.split_done:
                if st.button("➡️ Proceed to Model Training"):
                    st.session_state.stage = 5
                    st.session_state.training_done = False # Reset flag for next stage
                    st.success("Data ready! Moving to Model Training.")
                    st.rerun()
            else:
                st.info("Click 'Split Data' (or confirm data preparation for K-Means) to continue.")


        elif not st.session_state.processed_data:
            st.warning("Processed data not found. Please complete Stages 2 & 3.")
        elif not st.session_state.features:
            st.warning("Features not selected. Please go back to Stage 3 and confirm features.")

        st.markdown('</div>', unsafe_allow_html=True)

# 5. Model Training
elif st.session_state.stage == 5:
    st.title("5️⃣ Model Training")
    with st.container(border=False):
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state.X_train is not None:
            model_type = st.session_state.model_type
            st.subheader(f"Train {model_type} Model")

            # --- Linear Regression Training ---
            if model_type == "Linear Regression":
                if st.session_state.y_train is None:
                     st.error("Training data (y_train) not found. Please complete previous steps.")
                     st.stop()
                else:
                    st.markdown("*(Using default scikit-learn parameters)*")
                    # Optional: Check if scaling was applied and inform user
                    if st.session_state.scaler is not None:
                         st.info("Note: Preprocessing included scaling. Applying trained scaler to training data before fitting.")

                    if st.button("Train Linear Regression Model"):
                        with st.spinner("Training Linear Regression..."):
                            try:
                                X_train = st.session_state.X_train
                                y_train = st.session_state.y_train

                                # Apply scaling if it exists from preprocessing step
                                scaler = st.session_state.get('scaler')
                                if scaler:
                                    X_train_final = scaler.transform(X_train)
                                    st.session_state.X_train_scaled = X_train_final # Store scaled version
                                else:
                                    X_train_final = X_train # Use original if no scaler

                                model = LinearRegression()
                                model.fit(X_train_final, y_train)

                                st.session_state.model = model
                                st.success("Linear Regression model trained successfully!")

                                # Display Coefficients
                                coef_df = pd.DataFrame({
                                    'Feature': st.session_state.features,
                                    'Coefficient': model.coef_
                                }).sort_values('Coefficient', key=abs, ascending=False) # Sort by absolute value
                                st.write("**Model Coefficients:**")
                                try:
                                    fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', title='Feature Coefficients (Sorted by Magnitude)', color='Coefficient', color_continuous_scale='RdBu')
                                    st.plotly_chart(fig_coef, use_container_width=True)
                                except Exception as e:
                                     st.warning(f"Could not plot coefficients: {e}")
                                     st.dataframe(coef_df)
                                st.metric("Intercept", f"{model.intercept_:.4f}")

                                st.session_state.training_done = True

                            except Exception as e:
                                st.error(f"Error training Linear Regression: {e}")
                                traceback.print_exc()


            # --- Logistic Regression Training ---
            elif model_type == "Logistic Regression":
                 if st.session_state.y_train is None:
                     st.error("Training data (y_train) not found. Please complete previous steps.")
                     st.stop()
                 else:
                     st.markdown("#### Hyperparameters")
                     col_p1, col_p2, col_p3 = st.columns(3)
                     with col_p1: solver = st.selectbox("Solver:", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"], index=0)
                     with col_p2: C = st.number_input("Regularization (C):", min_value=0.01, max_value=100.0, value=1.0, step=0.1, format="%.2f")
                     with col_p3: max_iter = st.number_input("Max Iterations:", min_value=100, max_value=5000, value=1000, step=100)

                     # Inform about scaling
                     if st.session_state.scaler is not None:
                         st.info("Note: Preprocessing included scaling. Applying trained scaler to training data before fitting.")
                     else:
                          st.warning("Note: Data was not scaled during preprocessing. Logistic Regression often performs better with scaled features.")


                     if st.button("Train Logistic Regression Model"):
                         with st.spinner("Training Logistic Regression..."):
                             try:
                                 X_train = st.session_state.X_train
                                 y_train = st.session_state.y_train

                                 # Apply scaling if it exists from preprocessing step
                                 scaler = st.session_state.get('scaler')
                                 if scaler:
                                     X_train_final = scaler.transform(X_train)
                                     st.session_state.X_train_scaled = X_train_final # Store scaled version
                                 else:
                                     # Scale here if not done before (fit ONLY on train)
                                     st.info("Applying StandardScaler before training as no scaler was found from preprocessing.")
                                     scaler = StandardScaler()
                                     X_train_final = scaler.fit_transform(X_train)
                                     st.session_state.scaler = scaler # Store the newly fitted scaler
                                     st.session_state.X_train_scaled = X_train_final

                                 # Check target data type (sklearn expects numerical usually)
                                 if not pd.api.types.is_numeric_dtype(y_train):
                                     try:
                                         y_train = pd.to_numeric(y_train)
                                     except:
                                         st.error("Could not convert target variable to numeric type for Logistic Regression. Please check data.")
                                         st.stop()


                                 model = LogisticRegression(solver=solver, max_iter=max_iter, C=C, random_state=42)
                                 model.fit(X_train_final, y_train)

                                 st.session_state.model = model
                                 st.success("Logistic Regression model trained successfully!")

                                 # Display Coefficients (handle multi-class if necessary)
                                 if len(model.classes_) > 2:
                                      st.info(f"Multi-class classification ({len(model.classes_)} classes) detected.")
                                      # Provide option to view coefficients per class
                                      class_options = [f"Class {c}" for c in model.classes_]
                                      selected_class_view = st.selectbox("View coefficients for class:", class_options)
                                      class_index = class_options.index(selected_class_view)
                                      coefs = model.coef_[class_index]
                                      intercept_val = model.intercept_[class_index]
                                 else: # Binary or only one class detected (latter shouldn't happen with split)
                                      coefs = model.coef_[0]
                                      intercept_val = model.intercept_[0]


                                 coef_df = pd.DataFrame({
                                     'Feature': st.session_state.features,
                                     'Coefficient': coefs
                                 }).sort_values('Coefficient', key=abs, ascending=False) # Sort by absolute value
                                 st.write("**Model Coefficients:**")
                                 try:
                                     fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', title=f'Feature Coefficients (Sorted by Magnitude) for {selected_class_view if len(model.classes_) > 2 else "Class " + str(model.classes_[1])}', color='Coefficient', color_continuous_scale='RdBu')
                                     st.plotly_chart(fig_coef, use_container_width=True)
                                 except Exception as e:
                                     st.warning(f"Could not plot coefficients: {e}")
                                     st.dataframe(coef_df)

                                 st.metric("Intercept", f"{intercept_val:.4f}")
                                 st.session_state.training_done = True

                             except Exception as e:
                                 st.error(f"Error training Logistic Regression: {e}")
                                 traceback.print_exc()


            # --- K-Means Clustering Training ---
            elif model_type == "K-Means Clustering":
                 st.markdown("#### Hyperparameters")
                 col_k1, col_k2, col_k3 = st.columns(3)
                 with col_k1: n_clusters = st.number_input("Number of clusters (k):", min_value=2, max_value=20, value=st.session_state.get('kmeans_k', 3), step=1, key="kmeans_k")
                 with col_k2: init_method = st.selectbox("Initialization method:", ["k-means++", "random"], index=0, key="kmeans_init")
                 with col_k3: n_init = st.number_input("Num Initializations (n_init):", min_value=1, max_value=50, value=10, step=1, key="kmeans_ninit")
                 max_iter_kmeans = st.number_input("Maximum iterations:", min_value=100, max_value=1000, value=300, step=50, key="kmeans_maxiter")

                 # Store chosen k for potential later use (e.g., elbow plot suggestion)
                 st.session_state.kmeans_k = n_clusters

                 st.info("ℹ️ Data will be scaled using StandardScaler before applying K-Means.")

                 if st.button("Train K-Means Clustering Model"):
                     with st.spinner("Running K-Means Clustering..."):
                         try:
                             X = st.session_state.X_train # This is the full dataset prepared earlier

                             # Scale data using a new scaler or the one from preprocessing
                             scaler_kmeans = st.session_state.get('scaler') # Check if scaler exists from preprocessing
                             if scaler_kmeans is None:
                                 st.info("Applying new StandardScaler specifically for K-Means.")
                                 scaler_kmeans = StandardScaler()
                                 X_scaled = scaler_kmeans.fit_transform(X)
                                 st.session_state.scaler = scaler_kmeans # Store this scaler
                             else:
                                 st.info("Using existing scaler from preprocessing stage.")
                                 X_scaled = scaler_kmeans.transform(X)


                             st.session_state.X_scaled = X_scaled # Save scaled data used for clustering

                             model = KMeans(
                                 n_clusters=n_clusters,
                                 init=init_method,
                                 n_init=n_init,
                                 max_iter=max_iter_kmeans,
                                 random_state=42
                             )
                             model.fit(X_scaled)

                             st.session_state.model = model
                             st.session_state.clusters = model.labels_ # Store cluster labels
                             st.success(f"K-Means clustering completed with {n_clusters} clusters!")

                             # Display Cluster Centers (original scale if possible)
                             st.subheader("Cluster Centers")
                             centers_scaled = model.cluster_centers_
                             try:
                                centers_original = st.session_state.scaler.inverse_transform(centers_scaled)
                                centers_df = pd.DataFrame(centers_original, columns=st.session_state.features)
                                centers_df.index.name = 'Cluster'
                                st.write("Cluster Centers (Original Scale):")
                                st.dataframe(centers_df.style.format("{:.2f}").background_gradient(cmap='viridis'))
                             except Exception as e:
                                 st.warning(f"Could not inverse transform centers (using scaled values): {e}")
                                 centers_df_scaled = pd.DataFrame(centers_scaled, columns=st.session_state.features)
                                 centers_df_scaled.index.name = 'Cluster'
                                 st.write("Cluster Centers (Scaled):")
                                 st.dataframe(centers_df_scaled.style.format("{:.2f}").background_gradient(cmap='viridis'))


                             # Display Inertia & Silhouette Score
                             col_m1, col_m2 = st.columns(2)
                             inertia = model.inertia_
                             col_m1.metric("Inertia (WSS)", f"{inertia:.2f}")
                             try:
                                 silhouette_avg = silhouette_score(X_scaled, model.labels_)
                                 col_m2.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                                 st.session_state.metrics = {'Inertia': inertia, 'Silhouette Score': silhouette_avg}
                             except Exception as sil_e:
                                 col_m2.metric("Silhouette Score", "N/A")
                                 st.warning(f"Could not calculate silhouette score: {sil_e}")
                                 st.session_state.metrics = {'Inertia': inertia}


                             # Display Cluster Sizes
                             cluster_sizes = pd.Series(model.labels_).value_counts().sort_index().reset_index()
                             cluster_sizes.columns = ['Cluster', 'Number of Samples']
                             fig_sizes = px.bar(cluster_sizes, x='Cluster', y='Number of Samples', title='Cluster Sizes', text_auto=True)
                             fig_sizes.update_layout(xaxis_title='Cluster ID')
                             st.plotly_chart(fig_sizes, use_container_width=True)

                             st.session_state.training_done = True

                         except Exception as e:
                             st.error(f"Error running K-Means: {e}")
                             traceback.print_exc()

            # --- Proceed Button ---
            st.markdown("---")
            if st.session_state.training_done:
                if st.button("➡️ Proceed to Evaluation"):
                    st.session_state.stage = 6
                    st.session_state.evaluation_done = False # Reset flag for next stage
                    st.success("Model training complete! Moving to Evaluation.")
                    st.rerun()
            else:
                st.info("Click 'Train Model' to continue.")


        else:
            st.warning("Training data not found. Please complete the Train/Test Split stage first.")

        st.markdown('</div>', unsafe_allow_html=True)

# 6. Evaluation
elif st.session_state.stage == 6:
    st.title("6️⃣ Model Evaluation")
    with st.container(border=False):
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state.model:
            model = st.session_state.model
            model_type = st.session_state.model_type
            st.subheader(f"Evaluation Metrics for {model_type}")

            # --- Linear Regression Evaluation ---
            if model_type == "Linear Regression":
                 if st.session_state.X_test is not None and st.session_state.y_test is not None:
                     X_test = st.session_state.X_test
                     y_test = st.session_state.y_test
                     scaler = st.session_state.get('scaler')

                     try:
                         # Scale X_test if a scaler was used during training/preprocessing
                         if scaler:
                             X_test_final = scaler.transform(X_test)
                             st.session_state.X_test_scaled = X_test_final # Store for potential reuse
                         else:
                             X_test_final = X_test

                         # Check if predictions already exist for this model instance, otherwise predict
                         # This avoids re-predicting if user just navigates back and forth
                         if 'predictions' not in st.session_state or st.session_state.predictions is None:
                             with st.spinner("Generating predictions on test set..."):
                                 st.session_state.predictions = model.predict(X_test_final)
                         y_pred = st.session_state.predictions

                         # Calculate Metrics
                         mse = mean_squared_error(y_test, y_pred)
                         r2 = r2_score(y_test, y_pred)
                         rmse = np.sqrt(mse)
                         st.session_state.metrics = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

                         st.markdown("#### Performance Metrics")
                         col_m1, col_m2, col_m3 = st.columns(3)
                         col_m1.metric("R² Score", f"{r2:.4f}")
                         col_m2.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                         col_m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
                         st.markdown("---")

                         # Actual vs Predicted Plot
                         st.subheader("Actual vs. Predicted Values")
                         fig_pred = go.Figure()
                         fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions', marker=dict(color='#ff6e40', opacity=0.7)))
                         fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal (y=x)', line=dict(color='#1e3d59', dash='dash')))
                         fig_pred.update_layout(title='Actual vs. Predicted Values on Test Set', xaxis_title='Actual Values', yaxis_title='Predicted Values', legend_title="Legend")
                         st.plotly_chart(fig_pred, use_container_width=True)

                         # Residuals Plot
                         st.subheader("Residuals Analysis")
                         residuals = y_test - y_pred
                         fig_res = px.scatter(x=y_pred, y=residuals, title='Residuals vs. Predicted Values', labels={'x': 'Predicted Values', 'y': 'Residuals'}, opacity=0.7, trendline="lowess", trendline_color_override="red") # Add lowess trendline
                         fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
                         st.plotly_chart(fig_res, use_container_width=True)
                         st.caption("Residuals (errors) should ideally be randomly scattered around zero.")

                         fig_res_hist = px.histogram(residuals, nbins=50, title='Distribution of Residuals', labels={'value': 'Residual Value'})
                         fig_res_hist.update_layout(yaxis_title='Count')
                         st.plotly_chart(fig_res_hist, use_container_width=True)
                         st.caption("Residuals should ideally follow a normal distribution centered around zero.")

                         st.session_state.evaluation_done = True

                     except Exception as e:
                          st.error(f"Error during Linear Regression evaluation: {e}")
                          traceback.print_exc()

                 else:
                     st.warning("Test data (X_test, y_test) not found. Cannot evaluate.")


            # --- Logistic Regression Evaluation ---
            elif model_type == "Logistic Regression":
                 if st.session_state.X_test is not None and st.session_state.y_test is not None:
                     X_test = st.session_state.X_test
                     y_test = st.session_state.y_test
                     scaler = st.session_state.get('scaler')

                     try:
                         # Scale X_test using the *same* scaler fitted on training data
                         if scaler:
                             X_test_final = scaler.transform(X_test)
                             st.session_state.X_test_scaled = X_test_final # Store for potential reuse
                         else:
                             st.error("Scaler not found, but Logistic Regression requires scaled data for reliable evaluation. Please re-run preprocessing with scaling.")
                             st.stop() # Stop evaluation if scaling is missing

                         # Predict and get probabilities
                         if 'predictions' not in st.session_state or st.session_state.predictions is None:
                             with st.spinner("Generating predictions on test set..."):
                                 st.session_state.predictions = model.predict(X_test_final)
                         y_pred = st.session_state.predictions

                         if 'probabilities' not in st.session_state or st.session_state.probabilities is None:
                             if hasattr(model, "predict_proba"):
                                 with st.spinner("Calculating prediction probabilities..."):
                                     st.session_state.probabilities = model.predict_proba(X_test_final)
                             else:
                                 st.warning("Model does not support probability predictions.")
                                 st.session_state.probabilities = None
                         y_prob = st.session_state.probabilities


                         # Calculate Metrics
                         accuracy = accuracy_score(y_test, y_pred)
                         conf_matrix = confusion_matrix(y_test, y_pred)
                         class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                         st.session_state.metrics = {'Accuracy': accuracy, 'Confusion Matrix': conf_matrix, 'Classification Report': class_report}

                         st.markdown("#### Performance Metrics")
                         st.metric("Accuracy", f"{accuracy:.4f}")
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
                              st.write("Confusion Matrix Data:")
                              st.dataframe(pd.DataFrame(conf_matrix, index=labels, columns=labels))


                         # Classification Report
                         st.subheader("Classification Report")
                         try:
                             report_df = pd.DataFrame(class_report).transpose()
                             st.dataframe(report_df.style.format("{:.3f}").highlight_max(axis=0, subset=pd.IndexSlice[['precision', 'recall', 'f1-score'],:], color='lightgreen'))
                         except Exception as e:
                             st.error(f"Could not display classification report: {e}")
                             st.text(classification_report(y_test, y_pred, zero_division=0)) # Fallback to text
                         st.markdown("---")

                         # ROC Curve (if binary and probabilities available)
                         if y_prob is not None and len(model.classes_) == 2:
                             st.subheader("ROC Curve & AUC")
                             fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                             roc_auc = auc(fpr, tpr)
                             st.session_state.metrics['AUC'] = roc_auc # Store AUC

                             fig_roc = go.Figure()
                             fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.3f})', line=dict(color='#ff6e40', width=3)))
                             fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No Skill (AUC = 0.5)', line=dict(color='#1e3d59', dash='dash')))
                             fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate (1 - Specificity)', yaxis_title='True Positive Rate (Sensitivity/Recall)', legend=dict(x=0.6, y=0.1))
                             st.plotly_chart(fig_roc, use_container_width=True)

                         # Precision-Recall Curve (if binary and probabilities available)
                         if y_prob is not None and len(model.classes_) == 2:
                             st.subheader("Precision-Recall Curve")
                             precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
                             avg_precision = average_precision_score(y_test, y_prob[:, 1])
                             st.session_state.metrics['Average Precision'] = avg_precision # Store AP

                             # Baseline precision is the proportion of the positive class
                             no_skill = len(y_test[y_test==1]) / len(y_test) if len(y_test) > 0 else 0

                             fig_pr = go.Figure()
                             fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR Curve (AP = {avg_precision:.3f})', line=dict(color='#ff6e40', width=3)))
                             fig_pr.add_trace(go.Scatter(x=[0, 1], y=[no_skill, no_skill], mode='lines', name=f'No Skill (AP ≈ {no_skill:.3f})', line=dict(color='#1e3d59', dash='dash')))
                             fig_pr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall (True Positive Rate)', yaxis_title='Precision', legend=dict(x=0.5, y=0.1))
                             fig_pr.update_yaxes(range=[0.0, 1.05])
                             fig_pr.update_xaxes(range=[0.0, 1.0])
                             st.plotly_chart(fig_pr, use_container_width=True)

                         st.session_state.evaluation_done = True
                     except Exception as e:
                          st.error(f"Error during Logistic Regression evaluation: {e}")
                          traceback.print_exc()

                 else:
                     st.warning("Test data (X_test, y_test) not found. Cannot evaluate.")


            # --- K-Means Clustering Evaluation ---
            elif model_type == "K-Means Clustering":
                st.info("K-Means evaluation focuses on cluster quality (Inertia, Silhouette Score) and characteristics.")

                if 'clusters' in st.session_state and st.session_state.clusters is not None and 'X_scaled' in st.session_state:
                     st.markdown("#### Cluster Quality Metrics")
                     col_m1, col_m2 = st.columns(2)
                     col_m1.metric("Number of Clusters (k)", model.n_clusters)
                     col_m2.metric("Inertia (WSS)", f"{st.session_state.metrics.get('Inertia', 'N/A'):.2f}")

                     if 'Silhouette Score' in st.session_state.metrics:
                          st.metric("Silhouette Score", f"{st.session_state.metrics.get('Silhouette Score', 'N/A'):.3f}", help="Score between -1 (incorrect clustering) and +1 (dense, well-separated clusters). 0 indicates overlapping clusters.")
                     else:
                          # Try calculating here if not done during training
                          try:
                              silhouette_avg = silhouette_score(st.session_state.X_scaled, st.session_state.clusters)
                              st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                              st.session_state.metrics['Silhouette Score'] = silhouette_avg
                          except Exception as sil_e:
                              st.metric("Silhouette Score", "N/A")
                              st.warning(f"Could not calculate silhouette score: {sil_e}")
                     st.markdown("---")

                     # Elbow Method Plot (Optional Calculation - Reuse if calculated in training)
                     st.subheader("Elbow Method for Optimal k")
                     if st.button("Calculate Elbow Curve (may take time)"):
                         with st.spinner("Calculating inertia for different k values..."):
                              if 'X_scaled' in st.session_state and st.session_state.X_scaled is not None:
                                  X_scaled = st.session_state.X_scaled
                                  inertia_values = []
                                  k_range = range(2, 16) # Check k from 2 to 15
                                  try:
                                      for k_val in k_range:
                                          kmeans_elbow = KMeans(n_clusters=k_val, init='k-means++', n_init=10, max_iter=300, random_state=42)
                                          kmeans_elbow.fit(X_scaled)
                                          inertia_values.append(kmeans_elbow.inertia_)

                                      elbow_df = pd.DataFrame({'k': k_range, 'Inertia': inertia_values})
                                      fig_elbow = px.line(elbow_df, x='k', y='Inertia', title='Elbow Method for Optimal k', markers=True)
                                      fig_elbow.update_layout(xaxis_title='Number of Clusters (k)', yaxis_title='Inertia (Within-Cluster Sum of Squares)')
                                      # Highlight the chosen k
                                      chosen_k = st.session_state.get('kmeans_k', model.n_clusters)
                                      fig_elbow.add_vline(x=chosen_k, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Chosen k={chosen_k}")
                                      st.plotly_chart(fig_elbow, use_container_width=True)
                                      st.info("Look for the 'elbow' point where the rate of decrease in inertia slows down significantly. This suggests a reasonable trade-off between number of clusters and variance explained.")
                                  except Exception as e:
                                       st.error(f"Error calculating elbow curve: {e}")
                                       traceback.print_exc()

                              else:
                                   st.warning("Scaled data not found. Cannot calculate elbow curve.")
                     else:
                          st.info("Click the button above to calculate and visualize the Elbow curve to help assess the chosen number of clusters (k).")


                     st.session_state.evaluation_done = True
                else:
                     st.warning("Clustering results not found. Please train the K-Means model first.")

            # --- Proceed Button ---
            st.markdown("---")
            if st.session_state.evaluation_done:
                if st.button("➡️ Proceed to Results Visualization"):
                    st.session_state.stage = 7
                    st.success("Evaluation complete! Moving to Results Visualization.")
                    st.rerun()
            else:
                 st.info("Evaluation metrics and plots will appear here after calculation.")

        else:
            st.warning("Model not found. Please train a model in Stage 5 first.")

        st.markdown('</div>', unsafe_allow_html=True)


# 7. Results Visualization
elif st.session_state.stage == 7:
    st.title("7️⃣ Results Visualization & Interpretation")
    with st.container(border=False):
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        if st.session_state.model:
            model_type = st.session_state.model_type
            st.subheader(f"Final Results for {model_type}")

            # Display key metrics again
            st.markdown("#### Summary Metrics")
            metrics = st.session_state.metrics
            if metrics:
                 cols = st.columns(len(metrics))
                 i = 0
                 for name, value in metrics.items():
                     if isinstance(value, (int, float)):
                         cols[i].metric(name, f"{value:.3f}")
                         i = (i + 1) % len(metrics)
                     # Add specific handling if needed (e.g., conf matrix display)
            else:
                 st.info("No metrics were calculated or stored from the Evaluation stage.")
            st.markdown("---")

            # --- Linear Regression Visualization ---
            if model_type == "Linear Regression":
                if 'predictions' in st.session_state and st.session_state.predictions is not None and 'y_test' in st.session_state:
                    st.subheader("Key Visualizations")
                    # Re-display Actual vs Predicted
                    y_test = st.session_state.y_test
                    y_pred = st.session_state.predictions
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions', marker=dict(color='#ff6e40', opacity=0.7)))
                    fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal (y=x)', line=dict(color='#1e3d59', dash='dash')))
                    fig_pred.update_layout(title='Final: Actual vs. Predicted Values', xaxis_title='Actual Values', yaxis_title='Predicted Values', legend_title="Legend")
                    st.plotly_chart(fig_pred, use_container_width=True)

                    # Feature Importance / Coefficients
                    model = st.session_state.model
                    coef_df = pd.DataFrame({
                           'Feature': st.session_state.features,
                           'Coefficient': model.coef_
                       }).sort_values('Coefficient', key=abs, ascending=True) # Horizontal bar looks better ascending
                    st.subheader("Feature Importance (Coefficient Magnitude)")
                    fig_coef_imp = px.bar(coef_df, y='Feature', x='Coefficient', orientation='h', title='Feature Coefficients (Sorted by Absolute Value)', color='Coefficient', color_continuous_scale='RdBu')
                    st.plotly_chart(fig_coef_imp, use_container_width=True)
                    st.caption("Features with larger absolute coefficients have a stronger linear influence on the target variable in this model.")

                    # Download Predictions
                    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                    st.markdown(get_download_link(pred_df, 'linear_regression_predictions.csv', 'Download Test Predictions'), unsafe_allow_html=True)

                else:
                     st.warning("Prediction results or test data not found.")

            # --- Logistic Regression Visualization ---
            elif model_type == "Logistic Regression":
                 if 'metrics' in st.session_state and st.session_state.metrics:
                    st.subheader("Key Visualizations")
                    # Re-display Confusion Matrix
                    conf_matrix = st.session_state.metrics.get('Confusion Matrix')
                    if conf_matrix is not None:
                        labels = st.session_state.model.classes_
                        fig_conf = px.imshow(conf_matrix, text_auto=True, labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                             x=labels, y=labels, color_continuous_scale='Blues', title="Final: Confusion Matrix")
                        fig_conf.update_xaxes(side="bottom")
                        st.plotly_chart(fig_conf, use_container_width=True)

                    # Re-display ROC Curve (if available)
                    if 'probabilities' in st.session_state and st.session_state.probabilities is not None and len(st.session_state.model.classes_) == 2:
                         y_test = st.session_state.y_test
                         y_prob = st.session_state.probabilities
                         fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                         roc_auc = st.session_state.metrics.get('AUC', auc(fpr, tpr)) # Use stored or recalc
                         fig_roc = go.Figure()
                         fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.3f})', line=dict(color='#ff6e40', width=3)))
                         fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No Skill (AUC = 0.5)', line=dict(color='#1e3d59', dash='dash')))
                         fig_roc.update_layout(title='Final: Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                         st.plotly_chart(fig_roc, use_container_width=True)

                    # Feature Importance / Coefficients
                    model = st.session_state.model
                    if len(model.classes_) > 2: coefs = model.coef_[0]; title_suffix = f" (Class {model.classes_[0]} vs Rest)" # Default view
                    else: coefs = model.coef_[0]; title_suffix = f" (Class {model.classes_[1]} vs {model.classes_[0]})"
                    coef_df = pd.DataFrame({
                           'Feature': st.session_state.features,
                           'Coefficient': coefs
                       }).sort_values('Coefficient', key=abs, ascending=True) # Horizontal bar looks better ascending
                    st.subheader("Feature Importance (Coefficient Magnitude)")
                    fig_coef_imp = px.bar(coef_df, y='Feature', x='Coefficient', orientation='h', title=f'Feature Coefficients{title_suffix}', color='Coefficient', color_continuous_scale='RdBu')
                    st.plotly_chart(fig_coef_imp, use_container_width=True)
                    st.caption("Positive coefficients increase the log-odds of the positive class, negative coefficients decrease it.")

                    # Download Predictions
                    pred_df = pd.DataFrame({'Actual': st.session_state.y_test, 'Predicted': st.session_state.predictions})
                    if st.session_state.probabilities is not None:
                         for i, cls_label in enumerate(model.classes_):
                              pred_df[f'Probability_Class_{cls_label}'] = st.session_state.probabilities[:, i]
                    st.markdown(get_download_link(pred_df, 'logistic_regression_predictions.csv', 'Download Test Predictions & Probabilities'), unsafe_allow_html=True)


                 else:
                      st.warning("Evaluation metrics not found.")


            # --- K-Means Clustering Visualization ---
            elif model_type == "K-Means Clustering":
                 if 'clusters' in st.session_state and st.session_state.clusters is not None and 'X_scaled' in st.session_state:
                     X_scaled = st.session_state.X_scaled
                     clusters = st.session_state.clusters
                     features = st.session_state.features
                     n_features = X_scaled.shape[1]

                     st.subheader("Cluster Visualization (using PCA)")

                     if n_features < 2:
                         st.warning("Need at least 2 features for PCA visualization.")
                     else:
                         pca = PCA(n_components=2, random_state=42)
                         X_pca = pca.fit_transform(X_scaled)
                         pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                         pca_df['Cluster'] = clusters.astype(str) # Categorical for colors
                         pca_explained_variance = pca.explained_variance_ratio_.sum()

                         fig_clusters_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title=f"Clusters in PCA Space (Explains {pca_explained_variance:.1%} of Variance)",
                                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                         fig_clusters_pca.update_traces(marker=dict(size=8, opacity=0.8))
                         fig_clusters_pca.update_layout(legend_title_text='Cluster ID')
                         st.plotly_chart(fig_clusters_pca, use_container_width=True)

                     # Parallel Coordinates Plot (if enough features)
                     if n_features > 1:
                         st.subheader("Parallel Coordinates Plot of Clusters")
                         try:
                             parallel_df = pd.DataFrame(X_scaled, columns=features)
                             parallel_df['Cluster'] = clusters.astype(str)
                             # Sample if large
                             sample_size_pc = 5000
                             if len(parallel_df) > sample_size_pc:
                                  parallel_df_sampled = parallel_df.sample(sample_size_pc, random_state=42)
                                  st.info(f"Sampled {sample_size_pc} points for parallel coordinates plot.")
                             else:
                                 parallel_df_sampled = parallel_df

                             fig_parallel = px.parallel_coordinates(parallel_df_sampled, dimensions=features, color='Cluster',
                                                                  title="Feature Values Across Clusters (Scaled Data)",
                                                                  color_continuous_scale=px.colors.sequential.Viridis) # Use Viridis for better color mapping on numbers
                             st.plotly_chart(fig_parallel, use_container_width=True)
                             st.caption("This plot helps visualize how different features contribute to cluster separation. Lines represent individual data points.")
                         except Exception as e:
                              st.warning(f"Could not generate parallel coordinates plot: {e}")

                     # Feature Distributions per Cluster
                     st.subheader("Feature Distributions by Cluster")
                     feat_dist_select = st.selectbox("Select feature to compare distributions:", features, key="dist_feat_cluster_final")
                     # Get original data before scaling
                     X_original = st.session_state.X_train # Assumes X_train holds the unscaled data used for clustering
                     if X_original is not None and feat_dist_select in X_original.columns:
                         dist_df = X_original[[feat_dist_select]].copy()
                         dist_df['Cluster'] = clusters.astype(str)
                         y_label = f"{feat_dist_select} (Original Scale)"

                         fig_dist = px.box(dist_df, x='Cluster', y=feat_dist_select, color='Cluster', title=f"Distribution of {feat_dist_select} by Cluster",
                                           labels={'Cluster': 'Cluster ID', feat_dist_select: y_label}, color_discrete_sequence=px.colors.qualitative.Pastel)
                         st.plotly_chart(fig_dist, use_container_width=True)
                     else:
                          st.warning(f"Could not retrieve original data for '{feat_dist_select}'. Please check previous steps.")


                     # Download Clustered Data
                     try:
                         clustered_df = st.session_state.X_train.copy() # Start with original features before scaling
                         clustered_df['ClusterAssignment'] = clusters
                         st.markdown(get_download_link(clustered_df, 'kmeans_clustered_data.csv', 'Download Clustered Data'), unsafe_allow_html=True)
                     except Exception as e:
                          st.error(f"Could not prepare clustered data for download: {e}")

                 else:
                      st.warning("Clustering results or data not found.")

            # --- End of Analysis ---
            st.markdown("---")
            st.success("🏁 Analysis Complete!")
            st.balloons()
            if st.button("✨ Start New Analysis (Reset)"):
                reset_app()
                st.rerun()

        else:
            st.warning("No model results found. Please complete the previous stages.")

        st.markdown('</div>', unsafe_allow_html=True)

# Fallback for unexpected stage values
elif st.session_state.stage != 0: # Should not happen if logic is correct
    st.error("An unexpected application state was reached. Resetting.")
    reset_app()
    st.rerun()
