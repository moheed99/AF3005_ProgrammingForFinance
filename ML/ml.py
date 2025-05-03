import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA # Needed for K-Means visualization
import base64
# from PIL import Image # PIL is imported but not used directly, can be removed if not needed later
from datetime import datetime, timedelta
import io
import time
import traceback # For better error reporting

# ---- Harry Potter Theme Configuration ----

# Custom CSS for Harry Potter theme
def set_harry_potter_theme():
    # Harry Potter color scheme
    # Gryffindor: #740001 (dark red), #D3A625 (gold)
    # Slytherin: #1A472A (dark green), #5D5D5D (silver)
    # Ravenclaw: #0E1A40 (navy blue), #946B2D (bronze)
    # Hufflepuff: #ECB939 (yellow), #000000 (black)

    # Font URL - Using jsDelivr CDN for better reliability
    font_url = "https://cdn.jsdelivr.net/gh/DataGoblinz/harry_potter_font@main/lumos/Lumos.ttf"

    # Background image styling (using a more reliable hosting for the background image)
    hp_background = f"""
    <style>
    @font-face {{
        font-family: 'Lumos';
        src: url('{font_url}') format('truetype');
        font-weight: normal;
        font-style: normal;
    }}

    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1618639943295-1e371a1106a9?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80"); /* Hogwarts-like background */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main .block-container {{
        background-color: rgba(0, 0, 0, 0.75); /* Slightly darker overlay */
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        color: #D3A625; /* Gold text */
        border: 1px solid #D3A625; /* Gold border */
    }}
    .sidebar .sidebar-content {{
        background-color: rgba(116, 0, 1, 0.9); /* Gryffindor dark red, more opaque */
        border-radius: 15px;
        color: #D3A625; /* Gold text in sidebar */
        padding: 1rem;
        border: 1px solid #D3A625; /* Gold border */
    }}
    /* Make sidebar text gold */
     .sidebar .sidebar-content label, .sidebar .sidebar-content div, .sidebar .sidebar-content p {{
         color: #D3A625 !important;
         font-family: 'Lumos', sans-serif;
     }}
     /* Ensure radio button labels are themed */
    .stRadio label span{{
         color: #D3A625 !important;
         font-family: 'Lumos', sans-serif;
    }}

    h1, h2, h3 {{
        color: #D3A625 !important;
        font-family: 'Lumos', sans-serif !important;
        text-shadow: 2px 2px 4px #000000; /* Add shadow for better readability */
    }}
    .stButton>button {{
        background-color: #740001; /* Gryffindor Red */
        color: #D3A625; /* Gold */
        border: 2px solid #D3A625;
        border-radius: 10px;
        padding: 10px 20px;
        font-family: 'Lumos', sans-serif;
        font-size: 1.1em;
        transition: all 0.3s ease;
        /* Wand cursor might be annoying, using default for buttons */
        /* cursor: url("https://i.ibb.co/nQ0p5TQ/wand-cursor.png"), auto; */
    }}
    .stButton>button:hover {{
        background-color: #D3A625; /* Gold */
        color: #740001; /* Gryffindor Red */
        border: 2px solid #740001;
        transform: scale(1.05);
    }}
     /* Styling for text elements */
    div[data-testid="stText"], p, li {{
        color: #FFF8DC; /* Lighter text color (Cornsilk) for readability */
        font-family: sans-serif; /* Use readable font for paragraphs */
        font-size: 1.1em;
    }}
    div[data-testid="stMarkdownContainer"] p {{
         color: #FFF8DC; /* Lighter text color (Cornsilk) for readability */
         font-family: sans-serif; /* Use readable font for paragraphs */
    }}
     /* Make titles inside markdown stand out */
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {{
       font-family: 'Lumos', sans-serif !important;
       color: #D3A625 !important;
    }}

    /* Custom cursor - Optional, can be annoying - COMMENTED OUT TO FIX NameError */
    /* * {{
        cursor: url("https://cur.cursors-4u.net/symbols/sym-1/sym51.cur"), auto !important;
    }} */

    /* Success/Info/Error message styling */
    div[data-baseweb="notification"].stNotification {{
        background-color: rgba(26, 71, 42, 0.95) !important; /* Slytherin Green background */
        border: 2px solid #D3A625 !important;
        color: #D3A625 !important;
        border-radius: 10px;
        font-family: 'Lumos', sans-serif;
    }}
    div[data-baseweb="notification"] span {{
         color: #D3A625 !important;
    }}
     div[data-baseweb="notification"].stNotification > div > div > div {{
        color: #FFF8DC !important; /* Message text color */
         font-family: sans-serif;
    }}


    /* Table styling */
    .stDataFrame {{
        background-color: rgba(14, 26, 64, 0.8) !important; /* Ravenclaw Blue */
        border: 1px solid #D3A625;
    }}
    .stDataFrame table {{
        color: #D3A625 !important;
    }}
    .stDataFrame thead th {{
        color: #ECB939 !important; /* Hufflepuff Yellow for headers */
        background-color: rgba(0, 0, 0, 0.6) !important;
        font-family: 'Lumos', sans-serif;
    }}
     .stDataFrame tbody td {{
        color: #FFF8DC !important; /* Lighter cell text */
    }}

    /* Dropdown/Selectbox styling */
    div[data-baseweb="select"] > div:first-child {{
        background-color: #0E1A40 !important; /* Ravenclaw Navy */
        border: 1px solid #D3A625 !important;
        color: #D3A625 !important;
    }}
     div[data-baseweb="select"] span {{
        color: #D3A625 !important; /* Ensure text inside dropdown is gold */
    }}
     /* Input field styling */
    .stTextInput input {{
        background-color: rgba(0, 0, 0, 0.5) !important;
        color: #D3A625 !important;
        border: 1px solid #D3A625 !important;
    }}
    /* Slider styling */
    div[data-testid="stSlider"] {{
         color: #D3A625 !important; /* Slider label color */
    }}
    div[data-testid="stSlider"] label {{
         font-family: 'Lumos', sans-serif !important;
         color: #D3A625 !important;
    }}

    </style>
    """
    st.markdown(hp_background, unsafe_allow_html=True)

# Function to load Harry Potter themed images (using more reliable URLs)
def load_hp_image(image_type):
    # Using imgbb links provided by user, consider hosting images reliably if deploying
    images = {
        "welcome": "https://i.ibb.co/wQypbzx/hp-finance-welcome.gif",
        "sorting_hat": "https://i.ibb.co/vLGv5XS/sorting-hat.gif",
        "hogwarts_logo": "https://i.ibb.co/GWRyMCW/hogwarts-logo.png",
        "gryffindor": "https://i.ibb.co/9hFCpH8/gryffindor-crest.png",
        "slytherin": "https://i.ibb.co/zZLq209/slytherin-crest.png",
        "ravenclaw": "https://i.ibb.co/0jyPGhY/ravenclaw-crest.png",
        "hufflepuff": "https://i.ibb.co/4KNMMZB/hufflepuff-crest.png",
        "data_load": "https://i.ibb.co/yQSFmVD/data-load.gif",
        "preprocessing": "https://i.ibb.co/cLQx8zJ/preprocessing.gif",
        "feature_engineering": "https://i.ibb.co/4ZL0qvn/feature.gif",
        "train_test": "https://i.ibb.co/WkrCbjf/train-test.gif",
        "model_training": "https://i.ibb.co/GQs9TyJ/training.gif",
        "evaluation": "https://i.ibb.co/xFcpcLQ/evaluation.gif",
        "results": "https://i.ibb.co/NVbkzSg/results.gif",
        "success": "https://i.ibb.co/yByZvJW/success.gif"
    }
    # Fallback image
    default_image = "https://i.ibb.co/GWRyMCW/hogwarts-logo.png"
    return images.get(image_type, default_image)

# Harry Potter themed notification
def hp_notification(notification_type, message):
    spell_sounds = {
        "success": "🔮 Lumos Maxima! ", # Changed spell
        "info": "📜 Revelio! ",       # Changed spell
        "warning": "⚡ Impedimenta! ",   # Changed spell
        "error": "🔥 Incendio! "        # Changed spell
    }

    prefix = spell_sounds.get(notification_type, "📝 ")

    if notification_type == "success":
        st.success(f"{prefix}{message}")
    elif notification_type == "info":
        st.info(f"{prefix}{message}")
    elif notification_type == "warning":
        st.warning(f"{prefix}{message}")
    elif notification_type == "error":
        st.error(f"{prefix}{message}")
    else:
        st.write(f"{prefix}{message}") # Default write for other types

# Display Harry Potter themed GIF and title
def display_hp_title(title, image_type=None, image_width=400):
    cols = st.columns([1, 3, 1]) # Center the image
    with cols[1]:
        if image_type:
            st.image(load_hp_image(image_type), width=image_width)

    st.markdown(f"<h1 style='text-align: center; font-family: Lumos, sans-serif;'>{title}</h1>",
                unsafe_allow_html=True)

# Harry Potter themed divider
def hp_divider():
    divider_image = load_hp_image("hogwarts_logo") # Use the loaded logo
    divider = f"""
    <div style="display: flex; align-items: center; margin: 2rem 0;">
        <div style="flex-grow: 1; background: linear-gradient(to right, transparent, #D3A625); height: 2px;"></div>
        <div style="padding: 0 10px;">
            <img src="{divider_image}" height="40">
        </div>
        <div style="flex-grow: 1; background: linear-gradient(to left, transparent, #D3A625); height: 2px;"></div>
    </div>
    """
    st.markdown(divider, unsafe_allow_html=True)

# ---- Helper Functions ----

# Function to download results
def get_download_link(df, filename, text):
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        # Improved styling for the download button
        href = f"""
        <a href="data:file/csv;base64,{b64}" download="{filename}"
           style="display: inline-block;
                  padding: 10px 20px;
                  background-color: #740001;
                  color: #D3A625;
                  border: 2px solid #D3A625;
                  border-radius: 10px;
                  text-decoration: none;
                  font-family: 'Lumos', sans-serif;
                  font-size: 1.1em;
                  transition: all 0.3s ease;"
           onmouseover="this.style.backgroundColor='#D3A625'; this.style.color='#740001';"
           onmouseout="this.style.backgroundColor='#740001'; this.style.color='#D3A625';">
           📥 {text}
        </a>
        """
        return href
    except Exception as e:
        st.error(f"Error creating download link: {e}")
        return ""

# Function to preprocess data
def preprocess_data(df):
    """Handles missing values and basic type conversion."""
    try:
        # Handle missing values (dropping rows with any NaNs)
        # Consider more sophisticated imputation if needed
        df_clean = df.dropna()

        # Convert potential numeric columns stored as objects
        for col in df_clean.select_dtypes(include=['object']).columns:
            try:
                # Attempt conversion, ignore errors for truly categorical cols
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except ValueError:
                 hp_notification("warning", f"Could not convert column '{col}' to numeric. Keeping as object.")


        # Convert Date/Time columns if they exist
        for col in df_clean.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                 try:
                     df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                     # Optionally drop rows where date conversion failed
                     df_clean = df_clean.dropna(subset=[col])
                 except Exception:
                     hp_notification("warning", f"Could not convert column '{col}' to datetime. Skipping.")

        return df_clean
    except Exception as e:
        hp_notification("error", f"Error during preprocessing: {e}")
        return df # Return original df if error occurs

# Feature engineering function
def feature_engineering(df, target=None):
    """Encodes categorical features and separates features/target."""
    try:
        X = df.copy()
        y = None

        if target and target in X.columns:
            y = X.pop(target) # Use pop to remove target from X

        # Handle categorical features using Label Encoding
        # Note: OneHotEncoding might be better for non-ordinal features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        encoders = {} # Store encoders if needed later
        for col in categorical_cols:
            le = LabelEncoder()
            # Use .loc to avoid SettingWithCopyWarning
            X.loc[:, col] = le.fit_transform(X.loc[:, col])
            encoders[col] = le

        # Drop remaining non-numeric columns (like unconverted dates)
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns
        if len(non_numeric_cols) > 0:
            hp_notification("info", f"Dropping non-numeric columns: {', '.join(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols)


        # Optional: Scaling features (important for K-Means and sometimes Logistic Regression)
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        # X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        # st.session_state.scaler = scaler # Store scaler if used

        return X, y
    except Exception as e:
        hp_notification("error", f"Error during feature engineering: {e}")
        traceback.print_exc() # Print detailed error
        return None, None

# Function to train model
def train_model(X_train, y_train, model_type):
    """Trains the selected machine learning model."""
    try:
        model = None # Initialize model
        if model_type == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
        elif model_type == "Logistic Regression":
            # Scale features for Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            st.session_state.scaler = scaler # Store scaler to transform test set later
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
        elif model_type == "K-Means Clustering":
             # Scale features for K-Means
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            st.session_state.scaler = scaler # Store scaler to transform test set later
            # Let user choose number of clusters? For now, fixed at 3
            n_clusters = st.session_state.get('n_clusters', 3) # Get from session state or default to 3
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set n_init explicitly
            model.fit(X_train_scaled)

        return model
    except Exception as e:
        hp_notification("error", f"Error during model training for {model_type}: {e}")
        traceback.print_exc()
        return None

# Function to evaluate model
def evaluate_model(model, X_test, y_test=None, model_type=None):
    """Evaluates the trained model on the test set."""
    results = {}
    try:
        if model_type in ["Linear Regression", "Logistic Regression", "K-Means Clustering"]:
             # Apply scaling if it was used during training
            if 'scaler' in st.session_state:
                scaler = st.session_state.scaler
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test # No scaling was applied

        if model_type == "Linear Regression":
            y_pred = model.predict(X_test_scaled)
            results["Mean Squared Error"] = mean_squared_error(y_test, y_pred)
            results["R² Score"] = r2_score(y_test, y_pred)
            results["Predictions"] = y_pred
            results["Actuals"] = y_test # Include actuals for comparison plot

        elif model_type == "Logistic Regression":
            y_pred = model.predict(X_test_scaled)
            results["Accuracy"] = accuracy_score(y_test, y_pred)
            results["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
            results["Predictions"] = y_pred
            results["Actuals"] = y_test
            results["Classes"] = model.classes_ # Store class labels for confusion matrix

        elif model_type == "K-Means Clustering":
            cluster_labels = model.predict(X_test_scaled)
            results["Cluster Labels"] = cluster_labels
            results["Cluster Centers"] = model.cluster_centers_
             # Calculate Silhouette Score (needs scaled data and labels)
            if len(np.unique(cluster_labels)) > 1: # Silhouette score requires at least 2 clusters
                 try:
                     results["Silhouette Score"] = silhouette_score(X_test_scaled, cluster_labels)
                 except Exception as sil_e:
                     hp_notification("warning", f"Could not calculate Silhouette Score: {sil_e}")
                     results["Silhouette Score"] = "N/A"
            else:
                 results["Silhouette Score"] = "N/A (Only one cluster found)"
            results["Data"] = X_test # Include original test data for plotting

    except Exception as e:
        hp_notification("error", f"Error during model evaluation for {model_type}: {e}")
        traceback.print_exc()
        # Add dummy results to avoid breaking later steps
        results["Error"] = str(e)
        if model_type == "Linear Regression":
            results.setdefault("Mean Squared Error", "Error")
            results.setdefault("R² Score", "Error")
        elif model_type == "Logistic Regression":
            results.setdefault("Accuracy", "Error")
            results.setdefault("Confusion Matrix", "Error")
        elif model_type == "K-Means Clustering":
            results.setdefault("Cluster Labels", "Error")
            results.setdefault("Silhouette Score", "Error")

    return results

# ---- Main Application ----

def main():
    st.set_page_config(page_title="Financial Sorcerer", layout="wide")
    # Set Harry Potter theme
    set_harry_potter_theme()

    # Session state initialization
    if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
    if 'preprocessed' not in st.session_state: st.session_state.preprocessed = False
    if 'features_engineered' not in st.session_state: st.session_state.features_engineered = False
    if 'data_split' not in st.session_state: st.session_state.data_split = False
    if 'model_trained' not in st.session_state: st.session_state.model_trained = False
    if 'evaluation_done' not in st.session_state: st.session_state.evaluation_done = False
    if 'step' not in st.session_state: st.session_state.step = 0
    if 'df' not in st.session_state: st.session_state.df = None
    if 'df_clean' not in st.session_state: st.session_state.df_clean = None
    if 'X' not in st.session_state: st.session_state.X = None
    if 'y' not in st.session_state: st.session_state.y = None
    if 'X_train' not in st.session_state: st.session_state.X_train = None
    if 'X_test' not in st.session_state: st.session_state.X_test = None
    if 'y_train' not in st.session_state: st.session_state.y_train = None
    if 'y_test' not in st.session_state: st.session_state.y_test = None
    if 'model' not in st.session_state: st.session_state.model = None
    if 'model_type' not in st.session_state: st.session_state.model_type = "Linear Regression" # Default
    if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = {}


    # Sidebar setup
    with st.sidebar:
        # Using columns for better logo placement
        logo_cols = st.columns([1, 2, 1])
        with logo_cols[1]:
            st.image(load_hp_image("hogwarts_logo"), width=120)

        st.markdown("<h2 style='text-align: center; font-family: Lumos, sans-serif;'>Financial Wizard's Chamber</h2>", unsafe_allow_html=True)
        hp_divider()

        # Data source selection
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Choose Your Magical Source</h3>", unsafe_allow_html=True)
        data_source = st.radio("Select data origin:", ("Upload Kragle Dataset 📚", "Fetch from Yahoo Finance 🧙‍♂️"), label_visibility="collapsed")

        if data_source == "Upload Kragle Dataset 📚":
            uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'], label_visibility="collapsed")
            if uploaded_file is not None:
                 if st.button("🪄 Cast Loading Spell", key="upload_load"):
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.preprocessed = False # Reset subsequent steps
                        st.session_state.features_engineered = False
                        st.session_state.data_split = False
                        st.session_state.model_trained = False
                        st.session_state.evaluation_done = False
                        st.session_state.step = 1
                        hp_notification("success", "Dataset loaded successfully!")
                    except Exception as e:
                         hp_notification("error", f"Failed to load CSV: {e}")
                         st.session_state.data_loaded = False


        else:  # Yahoo Finance option
            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Choose Stock Ticker</h4>", unsafe_allow_html=True)
            ticker = st.text_input("Enter Stock Symbol", "AAPL", label_visibility="collapsed")

            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Select Time Period</h4>", unsafe_allow_html=True)
            period = st.selectbox("Select period:", ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"), label_visibility="collapsed")

            if st.button("🪄 Accio Stock Data!", key="yahoo_load"):
                with st.spinner("Summoning data from the financial realm..."):
                    try:
                        stock = yf.Ticker(ticker)
                        df = stock.history(period=period)
                        if df.empty:
                             hp_notification("error", f"No data found for ticker {ticker} and period {period}.")
                             st.session_state.data_loaded = False
                        else:
                            df.reset_index(inplace=True)
                            # Convert Date column to datetime explicitly if needed
                            if 'Date' in df.columns:
                                df['Date'] = pd.to_datetime(df['Date']).dt.date # Keep only date part for simplicity
                            st.session_state.df = df
                            st.session_state.data_loaded = True
                            st.session_state.preprocessed = False # Reset subsequent steps
                            st.session_state.features_engineered = False
                            st.session_state.data_split = False
                            st.session_state.model_trained = False
                            st.session_state.evaluation_done = False
                            st.session_state.step = 1
                            hp_notification("success", f"Successfully summoned {ticker} data!")
                    except Exception as e:
                        hp_notification("error", f"Failed to summon data: {e}")
                        st.session_state.data_loaded = False

        hp_divider()

        # Model selection (only if data is loaded)
        if st.session_state.data_loaded:
            st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Choose Your Magical Model</h3>", unsafe_allow_html=True)
            model_options = ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
            # Check if data seems suitable for classification (e.g., low cardinality target)
            # This is a heuristic and might need refinement
            is_classification_suitable = False
            if st.session_state.get('y') is not None:
                 if st.session_state.y.nunique() < 20 and st.session_state.y.dtype in ['int', 'object', 'category']:
                      is_classification_suitable = True

            default_model_index = 0 # Linear Regression default
            if st.session_state.model_type == "Logistic Regression": default_model_index = 1
            elif st.session_state.model_type == "K-Means Clustering": default_model_index = 2


            model_type = st.selectbox("Select model:", model_options, index=default_model_index, label_visibility="collapsed")

            # Update model type only if it changes
            if model_type != st.session_state.model_type:
                 st.session_state.model_type = model_type
                 # Reset steps from feature engineering onwards if model type changes
                 st.session_state.features_engineered = False
                 st.session_state.data_split = False
                 st.session_state.model_trained = False
                 st.session_state.evaluation_done = False
                 st.session_state.step = max(st.session_state.step, 2) # Go back to step 2 or stay
                 st.rerun() # Rerun to update the UI based on new model type

            if model_type == "K-Means Clustering":
                 st.session_state.n_clusters = st.number_input("Number of Clusters (K)", min_value=2, max_value=10, value=st.session_state.get('n_clusters', 3))

        # Reset button
        hp_divider()
        if st.button("🧹 Start New Journey", key="reset_all"):
            # Clear most session state variables
            keys_to_clear = ['data_loaded', 'preprocessed', 'features_engineered', 'data_split',
                             'model_trained', 'evaluation_done', 'df', 'df_clean', 'X', 'y',
                             'X_train', 'X_test', 'y_train', 'y_test', 'model', 'evaluation_results', 'scaler', 'n_clusters']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 0
            st.session_state.model_type = "Linear Regression" # Reset to default model
            hp_notification("info", "Restarting the magical journey!")
            time.sleep(1) # Pause for effect
            st.rerun()


    # Main content based on step
    try:
        if st.session_state.step == 0:
            # Welcome page
            display_hp_title("The Financial Sorcerer's Apprentice", "welcome")

            st.markdown("""
            <div style="text-align:center; margin: 2rem 0;">
                <p style="font-size: 1.3em; font-family: 'Lumos', sans-serif; color: #ECB939;">Welcome to Hogwarts School of Financial Wizardry!</p>
                <p style="font-size: 1.1em;">Where data transforms into insight and predictions become potent spells.</p>
                <p style="font-size: 1.1em;">Begin your quest by summoning data using the options in the sidebar.</p>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns([1,2,1])
            with cols[1]:
                st.image(load_hp_image("sorting_hat"), width=300)

            st.markdown("""
            <div style="text-align:center; font-family: sans-serif; margin: 2rem 0;">
                <p style="font-size: 1.1em;">The Sorting Hat awaits to guide your machine learning adventure!</p>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state.step == 1 and st.session_state.data_loaded and st.session_state.df is not None:
            # Step 1: Data Loading Display
            display_hp_title("The Magical Dataset Chamber", "data_load", image_width=350)

            st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Your Summoned Data Appears!</h3>", unsafe_allow_html=True)

            st.dataframe(st.session_state.df.head())

            st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Data Parchment Details</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            df_info = io.StringIO()
            st.session_state.df.info(buf=df_info)
            info_string = df_info.getvalue()

            with col1:
                st.write(f"📊 Rows: {st.session_state.df.shape[0]}")
                st.write(f"📜 Columns: {st.session_state.df.shape[1]}")
            with col2:
                st.write(f"✨ Missing Values: {st.session_state.df.isna().sum().sum()} entries")
                st.write(f"🧪 Column Types:")
                st.text(info_string) # Display concise info

            if st.button("🪄 Proceed to Cleansing Chamber"):
                hp_notification("info", "Entering the Cleansing Chamber...")
                st.session_state.step = 2
                st.rerun()


        elif st.session_state.step == 2 and st.session_state.data_loaded:
            # Step 2: Preprocessing
            display_hp_title("The Scourgify Charm Chamber", "preprocessing", image_width=300)

            st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Purify Your Data with Magical Cleansing</h3>", unsafe_allow_html=True)

            if st.session_state.df is not None:
                # Display missing values before preprocessing
                st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Dark Marks (Missing Values) Before Cleansing:</h4>", unsafe_allow_html=True)
                missing_before = st.session_state.df.isna().sum()
                missing_before_df = missing_before[missing_before > 0].reset_index()
                missing_before_df.columns = ['Column', 'Missing Count']
                if not missing_before_df.empty:
                     st.dataframe(missing_before_df)
                else:
                    st.write("No missing values detected! Excellent scroll-keeping!")

                # Display data types before preprocessing
                st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Initial Data Type Enchantments:</h4>", unsafe_allow_html=True)
                st.dataframe(st.session_state.df.dtypes.reset_index().rename(columns={'index':'Column', 0:'DataType'}))


                if st.button("🪄 Cast Scourgify Charm"):
                    with st.spinner("Performing magical cleansing... Scourgify!"):
                        # Preprocess data
                        st.session_state.df_clean = preprocess_data(st.session_state.df.copy()) # Use copy

                        if st.session_state.df_clean is not None:
                            st.session_state.preprocessed = True

                            # Display statistics after preprocessing
                            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>After Cleansing:</h4>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)

                            rows_removed = st.session_state.df.shape[0] - st.session_state.df_clean.shape[0]
                            missing_after = st.session_state.df_clean.isna().sum().sum()

                            with col1:
                                st.write(f"🧹 Rows remaining: {st.session_state.df_clean.shape[0]}")
                                st.write(f"🗑️ Rows removed (due to NaN): {rows_removed}")

                            with col2:
                                st.write(f"✨ Missing values after: {missing_after}")
                                st.write(f"📜 Columns remaining: {st.session_state.df_clean.shape[1]}")

                            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Final Data Type Enchantments:</h4>", unsafe_allow_html=True)
                            st.dataframe(st.session_state.df_clean.dtypes.reset_index().rename(columns={'index':'Column', 0:'DataType'}))


                            hp_notification("success", "Scourgify successful! Your data is now sparkling clean.")
                            st.session_state.step = 3
                            st.rerun() # Move to next step
                        else:
                             hp_notification("error", "Cleansing charm failed. Check previous steps.")
            else:
                 st.warning("Cannot perform cleansing without loaded data. Please load data first.")


        elif st.session_state.step == 3 and st.session_state.preprocessed:
            # Step 3: Feature Engineering
            display_hp_title("The Transfiguration Chamber", "feature_engineering", image_width=350)

            st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Transfigure Your Data into Features</h3>", unsafe_allow_html=True)

            if st.session_state.df_clean is not None:
                # Display available columns
                st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Available Magical Components (Columns):</h4>", unsafe_allow_html=True)
                st.write(", ".join(st.session_state.df_clean.columns.tolist()))

                target_col = None
                # Select target variable if using supervised learning
                if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                     # Filter for potential target columns (numeric for LR, potentially numeric/categorical for LogR)
                     potential_targets = st.session_state.df_clean.columns.tolist()
                     if st.session_state.model_type == "Linear Regression":
                         potential_targets = st.session_state.df_clean.select_dtypes(include=np.number).columns.tolist()

                     if potential_targets:
                        target_col = st.selectbox("🎯 Select Target Variable (The prophecy you wish to predict)",
                                                  potential_targets, index=max(0, len(potential_targets)-1)) # Default to last column
                     else:
                         st.warning("No suitable target columns found for the selected model type after preprocessing.")

                if st.button("🪄 Cast Transfiguration Spell"):
                    with st.spinner("Transfiguring data with advanced charms..."):
                        # Feature engineering
                        X, y = feature_engineering(st.session_state.df_clean, target_col)

                        if X is not None:
                            st.session_state.X = X
                            st.session_state.y = y # Will be None for K-Means

                            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Transfiguration Results:</h4>", unsafe_allow_html=True)
                            if target_col:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"🧙‍♂️ Features (X): {X.shape[1]} columns")
                                    st.dataframe(X.head())
                                with col2:
                                    st.write(f"🔮 Target (y): {target_col}")
                                    if y is not None:
                                        # Display target distribution
                                        unique_vals = y.nunique()
                                        if unique_vals < 2:
                                             st.warning(f"Target variable '{target_col}' has only {unique_vals} unique value. This may cause issues for model training.")
                                        elif unique_vals <= 15 and st.session_state.model_type == "Logistic Regression": # Show pie for classification with few classes
                                            fig = px.pie(names=y.astype(str).value_counts().index,
                                                        values=y.value_counts().values,
                                                        title=f"Distribution of Target: {target_col}",
                                                        color_discrete_sequence=px.colors.sequential.YlOrRd)
                                            fig.update_layout(legend_title_text='Classes')
                                            st.plotly_chart(fig, use_container_width=True)
                                        else: # Show histogram for regression or many classes
                                            fig = px.histogram(y, title=f"Distribution of Target: {target_col}",
                                                               color_discrete_sequence=['#D3A625'])
                                            st.plotly_chart(fig, use_container_width=True)

                            else: # For clustering
                                st.write(f"🧙‍♂️ Features for Clustering (X): {X.shape[1]} columns")
                                st.dataframe(X.head())


                            st.session_state.features_engineered = True
                            hp_notification("success", "Transfiguration complete! Your data is ready for Divination.")
                            st.session_state.step = 4
                            st.rerun()
                        else:
                             hp_notification("error", "Transfiguration spell fizzled. Check data and target selection.")

            else:
                 st.warning("Cannot perform transfiguration without cleansed data. Please complete the previous step.")


        elif st.session_state.step == 4 and st.session_state.features_engineered:
            # Step 4: Train/Test Split
            display_hp_title("The Divination Chamber", "train_test", image_width=350)

            st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Gaze into the Crystal Ball: Splitting the Data</h3>", unsafe_allow_html=True)
            st.markdown("We must divide our knowledge (data) to train our spells and then test their foresight.")

            if st.session_state.X is not None:
                # Test size slider
                test_size = st.slider("🔮 Select Test Size Potion Strength (Proportion for testing)",
                                      0.1, 0.5, 0.2, 0.05,
                                      help="How much data should be held back to test the model's predictions?")

                random_state = st.number_input("✨ Enter a Lucky Number (Random Seed)", value=42, step=1,
                                               help="Ensures the split is the same each time, like a fixed star alignment.")


                if st.button("🪄 Cast Divination Spell"):
                    with st.spinner("Dividing data with mystical precision..."):
                         try:
                            # Split data
                            if st.session_state.y is not None: # Supervised learning
                                if len(st.session_state.X) != len(st.session_state.y):
                                     raise ValueError("Features (X) and Target (y) have different numbers of samples!")
                                # Stratify for classification tasks if target is suitable
                                stratify_opt = None
                                if st.session_state.model_type == "Logistic Regression" and st.session_state.y.nunique() < len(st.session_state.y) * test_size:
                                     try:
                                         # Check if stratification is possible
                                         value_counts = st.session_state.y.value_counts()
                                         if (value_counts < 2).any():
                                              hp_notification("warning", "Cannot stratify split due to classes with only 1 sample. Performing regular split.")
                                         else:
                                              stratify_opt = st.session_state.y
                                              hp_notification("info", "Stratifying split by target variable.")
                                     except Exception as strat_err:
                                          hp_notification("warning", f"Could not stratify: {strat_err}. Performing regular split.")


                                X_train, X_test, y_train, y_test = train_test_split(
                                    st.session_state.X, st.session_state.y, test_size=test_size, random_state=random_state, stratify=stratify_opt)

                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = y_train
                                st.session_state.y_test = y_test

                            else:  # Unsupervised learning (K-Means)
                                X_train, X_test = train_test_split(
                                    st.session_state.X, test_size=test_size, random_state=random_state)

                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = None # Explicitly set to None
                                st.session_state.y_test = None # Explicitly set to None

                            # Visualize the split
                            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Data Split Foreseen:</h4>", unsafe_allow_html=True)

                            split_data = {
                                'Set': ['Training Data', 'Testing Data'],
                                'Size': [st.session_state.X_train.shape[0], st.session_state.X_test.shape[0]]
                            }
                            split_df = pd.DataFrame(split_data)

                            fig = px.pie(split_df, values='Size', names='Set',
                                        title='Training vs. Testing Data Split',
                                        color_discrete_sequence=['#740001', '#D3A625']) # Gryffindor colors
                            fig.update_layout(legend_title_text='Dataset Portion')
                            st.plotly_chart(fig, use_container_width=True)

                            st.session_state.data_split = True
                            hp_notification("success", "Divination complete! Your data awaits the Spell Casting Chamber.")
                            st.session_state.step = 5
                            st.rerun()

                         except ValueError as ve:
                              hp_notification("error", f"Divination Error: {ve}")
                         except Exception as e:
                             hp_notification("error", f"An unexpected error occurred during Divination: {e}")
                             traceback.print_exc()

            else:
                 st.warning("Cannot perform Divination without transfigured features. Please complete the previous step.")


        elif st.session_state.step == 5 and st.session_state.data_split:
            # Step 5: Model Training
            display_hp_title("The Spell Casting Chamber", "model_training", image_width=300)

            st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Incantations Ready: Training the Model</h3>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-family: sans-serif; margin: 1rem 0; padding: 1rem; background-color: rgba(14, 26, 64, 0.5); border-radius: 10px; border: 1px solid #946B2D;">
                <p>You have chosen the <span style="color: #D3A625; font-weight: bold; font-family: 'Lumos', sans-serif;">{st.session_state.model_type}</span> spell.</p>
                <p>This powerful magic will now learn from the training data to {
                'predict numerical values (like future gold galleons!)' if st.session_state.model_type == "Linear Regression"
                else 'classify outcomes into distinct categories (like House sorting!)' if st.session_state.model_type == "Logistic Regression"
                else 'discover hidden groups within your data (like secret societies!)'}.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.X_train is not None:
                if st.button("🪄 Cast Training Spell"):
                    with st.spinner(f"Casting the {st.session_state.model_type} training spell... Alohomora Patterns!"):
                        # Train model
                        model = train_model(st.session_state.X_train, st.session_state.y_train, st.session_state.model_type)

                        if model is not None:
                            st.session_state.model = model
                            st.session_state.model_trained = True

                            # Display training information
                            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Training Incantation Complete!</h4>", unsafe_allow_html=True)

                            if st.session_state.model_type == "Linear Regression":
                                st.write("📊 Learned Coefficients (Feature Influence):")
                                try:
                                    coef_df = pd.DataFrame({
                                        'Feature': st.session_state.X_train.columns,
                                        'Coefficient': model.coef_
                                    }).sort_values(by='Coefficient', key=abs, ascending=False) # Sort by absolute value

                                    fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                                                title='Feature Importance (Coefficients)',
                                                color='Coefficient', color_continuous_scale=px.colors.sequential.YlOrRd)
                                    fig.update_layout(yaxis_title="Feature", xaxis_title="Coefficient Value")
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.write(f"Intercept (Base Prediction): {model.intercept_:.4f}")
                                except Exception as plot_e:
                                     hp_notification("warning", f"Could not plot coefficients: {plot_e}")
                                     st.dataframe(coef_df)


                            elif st.session_state.model_type == "Logistic Regression":
                                st.write("📊 Learned Coefficients (Odds Influence):")
                                try:
                                    # Handle multi-class vs binary
                                    if model.coef_.shape[0] == 1: # Binary classification
                                        coef_df = pd.DataFrame({
                                            'Feature': st.session_state.X_train.columns,
                                            'Coefficient': model.coef_[0]
                                        }).sort_values(by='Coefficient', key=abs, ascending=False)

                                        fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                                                    title=f'Feature Importance for Class {model.classes_[1]} (Log-Odds)',
                                                    color='Coefficient', color_continuous_scale=px.colors.diverging.RdBu)
                                        fig.update_layout(yaxis_title="Feature", xaxis_title="Log-Odds Coefficient")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else: # Multi-class classification
                                        st.write("Multi-class model coefficients (Log-Odds for each class vs rest):")
                                        coef_dfs = []
                                        for i, class_label in enumerate(model.classes_):
                                             coef_dfs.append(pd.DataFrame({
                                                 'Feature': st.session_state.X_train.columns,
                                                 f'Coefficient_Class_{class_label}': model.coef_[i]
                                             }))
                                        # Combine dataframes for display
                                        all_coef_df = coef_dfs[0]
                                        for i in range(1, len(coef_dfs)):
                                             all_coef_df = pd.merge(all_coef_df, coef_dfs[i], on='Feature')
                                        st.dataframe(all_coef_df.set_index('Feature'))
                                        hp_notification("info", "Bar chart skipped for multi-class coefficients.")

                                except Exception as plot_e:
                                     hp_notification("warning", f"Could not display coefficients: {plot_e}")
                                     # Fallback display if plotting fails
                                     st.write(model.coef_)


                            elif st.session_state.model_type == "K-Means Clustering":
                                st.write(f"🧙‍♂️ Discovered Cluster Centers ({model.n_clusters} clusters):")
                                try:
                                     # Use scaled centers if scaler exists
                                    centers_to_display = model.cluster_centers_
                                    if 'scaler' in st.session_state:
                                         centers_to_display = st.session_state.scaler.inverse_transform(model.cluster_centers_) # Show in original scale

                                    centers_df = pd.DataFrame(centers_to_display,
                                                            columns=st.session_state.X_train.columns,
                                                            index=[f"Cluster {i}" for i in range(model.n_clusters)])
                                    st.dataframe(centers_df.style.background_gradient(cmap='YlOrRd')) # Add heatmap style
                                except Exception as plot_e:
                                     hp_notification("warning", f"Could not display cluster centers: {plot_e}")


                            hp_notification("success", f"{st.session_state.model_type} model trained successfully! Proceed to the Evaluation Chamber.")
                            st.session_state.step = 6
                            st.rerun()
                        else:
                             hp_notification("error", "Model training spell failed. Check logs or data.")
            else:
                 st.warning("Cannot cast training spell without prepared training data. Please complete previous steps.")


        elif st.session_state.step == 6 and st.session_state.model_trained:
            # Step 6: Model Evaluation
            display_hp_title("The Pensieve of Evaluation", "evaluation", image_width=300)
            st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Gazing into the Past: Evaluating Model Performance</h3>", unsafe_allow_html=True)
            st.markdown("Let's use the Pensieve (test data) to see how well our magical model performs on unseen memories.")


            if st.session_state.model is not None and st.session_state.X_test is not None:
                 if st.button("🪄 Cast Evaluation Spell"):
                     with st.spinner("Consulting the Pensieve for model insights..."):
                        # Evaluate model
                        # Pass y_test only if it exists (i.e., not K-Means)
                        y_test_arg = st.session_state.y_test if hasattr(st.session_state, 'y_test') else None

                        evaluation_results = evaluate_model(st.session_state.model,
                                                            st.session_state.X_test,
                                                            y_test_arg,
                                                            st.session_state.model_type)

                        if evaluation_results and "Error" not in evaluation_results:
                            st.session_state.evaluation_results = evaluation_results
                            st.session_state.evaluation_done = True

                            # Display quick summary of results
                            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Pensieve Reflections (Evaluation Metrics):</h4>", unsafe_allow_html=True)
                            if st.session_state.model_type == "Linear Regression":
                                st.metric("Mean Squared Error", f"{evaluation_results.get('Mean Squared Error', 'N/A'):.4f}")
                                st.metric("R² Score", f"{evaluation_results.get('R² Score', 'N/A'):.4f}", help="Closer to 1 is better")
                            elif st.session_state.model_type == "Logistic Regression":
                                st.metric("Accuracy", f"{evaluation_results.get('Accuracy', 'N/A'):.4f}")
                                st.write("Confusion Matrix:")
                                if "Confusion Matrix" in evaluation_results:
                                     st.dataframe(pd.DataFrame(evaluation_results["Confusion Matrix"],
                                                               index=evaluation_results.get("Classes", None),
                                                               columns=evaluation_results.get("Classes", None)))
                                else:
                                     st.write("Not available.")
                            elif st.session_state.model_type == "K-Means Clustering":
                                 st.metric("Silhouette Score", f"{evaluation_results.get('Silhouette Score', 'N/A')}", help="Score between -1 and 1. Closer to 1 indicates better-defined clusters.")
                                 st.write(f"Data points assigned to {st.session_state.model.n_clusters} clusters.")


                            hp_notification("success", "Evaluation complete! Proceed to the Results Chamber to visualize the findings.")
                            st.session_state.step = 7
                            st.rerun()
                        else:
                            hp_notification("error", f"Evaluation spell failed. {evaluation_results.get('Error', 'Check logs.')}")
                 else:
                     st.info("Click the button above to cast the evaluation spell on the test data.")
            else:
                 st.warning("Cannot evaluate without a trained model and test data. Please complete previous steps.")


        elif st.session_state.step == 7 and st.session_state.evaluation_done:
             # Step 7: Results and Visualization
            display_hp_title("The Great Hall of Results", "results", image_width=400)
            st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Unveiling the Prophecy: Results & Visualizations</h3>", unsafe_allow_html=True)

            results = st.session_state.evaluation_results

            if not results or "Error" in results:
                 hp_notification("error", f"Cannot display results due to an earlier error: {results.get('Error', 'Unknown error')}")
                 st.stop() # Stop execution for this step if results are invalid

            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Performance Metrics:</h4>", unsafe_allow_html=True)
            cols_metrics = st.columns(2)
            if st.session_state.model_type == "Linear Regression":
                with cols_metrics[0]:
                    st.metric("Mean Squared Error", f"{results.get('Mean Squared Error', 'N/A'):.4f}", delta=None, help="Lower is better")
                with cols_metrics[1]:
                    st.metric("R² Score", f"{results.get('R² Score', 'N/A'):.4f}", delta=None, help="Proportion of variance explained (closer to 1 is better)")

                # Visualization: Actual vs Predicted
                st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Divination Accuracy: Actual vs. Predicted Values</h4>", unsafe_allow_html=True)
                if "Actuals" in results and "Predictions" in results:
                     pred_df = pd.DataFrame({'Actual': results['Actuals'], 'Predicted': results['Predictions']})
                     fig = px.scatter(pred_df, x='Actual', y='Predicted',
                                      title='Actual vs. Predicted Values',
                                      labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
                                      trendline='ols', # Add regression line
                                      color_discrete_sequence=['#ECB939']) # Hufflepuff Yellow
                     fig.add_shape(type='line', line=dict(dash='dash', color='#740001'), x0=pred_df['Actual'].min(), y0=pred_df['Actual'].min(), x1=pred_df['Actual'].max(), y1=pred_df['Actual'].max()) # Add y=x line
                     st.plotly_chart(fig, use_container_width=True)

                     # Download Predictions
                     st.markdown(get_download_link(pred_df, 'linear_regression_predictions.csv', 'Download Predictions'), unsafe_allow_html=True)

            elif st.session_state.model_type == "Logistic Regression":
                with cols_metrics[0]:
                    st.metric("Accuracy", f"{results.get('Accuracy', 'N/A'):.4f}", delta=None, help="Overall correctness")
                with cols_metrics[1]:
                     st.write(" ") # Placeholder for alignment

                # Visualization: Confusion Matrix Heatmap
                st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Sorting Hat's Decisions: Confusion Matrix</h4>", unsafe_allow_html=True)
                if "Confusion Matrix" in results:
                     cm = results["Confusion Matrix"]
                     classes = results.get("Classes", [f"Class {i}" for i in range(cm.shape[0])])
                     fig = px.imshow(cm, text_auto=True,
                                     labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                                     x=classes, y=classes,
                                     title='Confusion Matrix Heatmap',
                                     color_continuous_scale=px.colors.sequential.Greens) # Slytherin Green scale
                     st.plotly_chart(fig, use_container_width=True)

                     # Download Predictions
                     pred_df = pd.DataFrame({'Actual': results['Actuals'], 'Predicted': results['Predictions']})
                     st.markdown(get_download_link(pred_df, 'logistic_regression_predictions.csv', 'Download Predictions'), unsafe_allow_html=True)


            elif st.session_state.model_type == "K-Means Clustering":
                 with cols_metrics[0]:
                    st.metric("Silhouette Score", f"{results.get('Silhouette Score', 'N/A')}", delta=None, help="Cluster separation quality (closer to 1 is better)")
                 with cols_metrics[1]:
                      st.metric("Number of Clusters (K)", f"{st.session_state.model.n_clusters}")

                 # Visualization: Cluster Plot (using PCA for 2D)
                 st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Mapping the Secret Societies: Cluster Visualization (PCA 2D)</h4>", unsafe_allow_html=True)
                 if "Cluster Labels" in results and "Data" in results:
                    X_test_orig = results["Data"] # Original (unscaled) test data
                    labels = results["Cluster Labels"]
                    n_components = min(2, X_test_orig.shape[1]) # Use 2 components if possible

                    if n_components < 2:
                        st.warning("Need at least 2 features to create a 2D cluster plot.")
                    else:
                        pca = PCA(n_components=2)
                        try:
                             # Scale data before PCA if not already scaled (or use scaled data directly if stored)
                            if 'scaler' in st.session_state:
                                X_test_scaled = st.session_state.scaler.transform(X_test_orig)
                                X_pca = pca.fit_transform(X_test_scaled)
                            else:
                                # Scale just for PCA if no global scaler was used
                                temp_scaler = StandardScaler()
                                X_pca = pca.fit_transform(temp_scaler.fit_transform(X_test_orig))

                            pca_df = pd.DataFrame(X_pca, columns=['PCA Component 1', 'PCA Component 2'], index=X_test_orig.index)
                            pca_df['Cluster'] = labels.astype(str) # Convert labels to string for discrete colors

                            fig = px.scatter(pca_df, x='PCA Component 1', y='PCA Component 2', color='Cluster',
                                             title='Clusters Visualized using PCA (2 Components)',
                                             color_discrete_sequence=px.colors.qualitative.Set1) # Use distinct colors
                            fig.update_layout(legend_title_text='Assigned Cluster')
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as pca_e:
                             hp_notification("error", f"Could not create PCA plot: {pca_e}")


                    # Download Clustered Data
                    clustered_df = X_test_orig.copy()
                    clustered_df['ClusterAssignment'] = labels
                    st.markdown(get_download_link(clustered_df, 'kmeans_clusters.csv', 'Download Clustered Data'), unsafe_allow_html=True)

            hp_divider()
            st.image(load_hp_image("success"), width=300)
            hp_notification("success", "Your financial magic has been revealed! Cast 'Start New Journey' in the sidebar to begin again.")


        # Fallback for invalid state
        elif st.session_state.step > 0 and st.session_state.df is None:
             st.warning("It seems the data has vanished! Please restart your journey by loading data from the sidebar.")
             st.session_state.step = 0 # Reset step
             # Add a button to explicitly reset if needed
             if st.button("Return to Start"):
                 st.session_state.step = 0
                 st.rerun()

    except KeyError as ke:
        hp_notification("error", f"A required magical ingredient (session key: {ke}) is missing! Restarting journey might help.")
        st.exception(ke)
        # Optionally reset state here
        # st.session_state.step = 0
        # st.rerun()
    except Exception as e:
        hp_notification("error", f"An unexpected dark force interrupted the process: {e}")
        st.exception(e) # Show full traceback for debugging
        traceback.print_exc()


if __name__ == "__main__":
    main()
