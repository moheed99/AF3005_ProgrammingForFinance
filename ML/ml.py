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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import base64
from PIL import Image
from datetime import datetime, timedelta
import io
import time

# ---- Harry Potter Theme Configuration ----

# Custom CSS for Harry Potter theme
def set_harry_potter_theme():
    # Harry Potter color scheme
    # Gryffindor: #740001 (dark red), #D3A625 (gold)
    # Slytherin: #1A472A (dark green), #5D5D5D (silver)
    # Ravenclaw: #0E1A40 (navy blue), #946B2D (bronze)
    # Hufflepuff: #ECB939 (yellow), #000000 (black)
    
    # Background image styling
    hp_background = """
    <style>
    .stApp {
        background-image: url("https://i.ibb.co/gTdnPxb/hogwarts-background.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main .block-container {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        color: #D3A625;
    }
    .sidebar .block-container {
        background-color: rgba(116, 0, 1, 0.85);
        border-radius: 15px;
        color: #D3A625;
    }
    h1, h2, h3 {
        color: #D3A625 !important;
        font-family: 'Lumos', sans-serif;
    }
    .stButton>button {
        background-color: #740001;
        color: #D3A625;
        border: 2px solid #D3A625;
        border-radius: 10px;
        cursor: url("https://i.ibb.co/nQ0p5TQ/wand-cursor.png"), auto;
    }
    .stButton>button:hover {
        background-color: #D3A625;
        color: #740001;
        border: 2px solid #740001;
    }
    div[data-testid="stText"] {
        font-family: 'Lumos', sans-serif;
    }
    /* Custom cursor */
    * {
        cursor: url("https://i.ibb.co/nQ0p5TQ/wand-cursor.png"), auto !important;
    }
    /* Harry Potter font import */
    @font-face {
        font-family: 'Lumos';
        src: url('https://dl.dropboxusercontent.com/s/w3y5x1gkzpyqtsa/lumos.ttf') format('truetype');
    }
    /* Success/Info/Error message styling */
    div[data-baseweb="notification"] {
        background-color: rgba(26, 71, 42, 0.9) !important;
        border: 2px solid #D3A625 !important;
        color: #D3A625 !important;
    }
    /* Table styling */
    .stDataFrame {
        background-color: rgba(14, 26, 64, 0.7) !important;
    }
    .stDataFrame table {
        color: #D3A625 !important;
    }
    /* Dropdown styling */
    div[data-baseweb="select"] {
        background-color: #0E1A40 !important;
        border: 1px solid #D3A625 !important;
    }
    </style>
    """
    st.markdown(hp_background, unsafe_allow_html=True)
    
# Function to load Harry Potter themed images
def load_hp_image(image_type):
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
    return images.get(image_type, images["hogwarts_logo"])

# Harry Potter themed notification
def hp_notification(notification_type, message):
    spell_sounds = {
        "success": "🔮 Spell Successful: ",
        "info": "📜 Ancient Runes Reveal: ",
        "warning": "⚡ Be Cautious, Wizard: ",
        "error": "🔥 Dark Magic Detected: "
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
        st.write(f"{prefix}{message}")

# Display Harry Potter themed GIF and title
def display_hp_title(title, image_type=None):
    if image_type:
        st.image(load_hp_image(image_type), width=400)
    
    st.markdown(f"<h1 style='text-align: center; font-family: Lumos, sans-serif;'>{title}</h1>", 
                unsafe_allow_html=True)

# Harry Potter themed divider
def hp_divider():
    divider = """
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <div style="flex-grow: 1; background-color: #D3A625; height: 2px;"></div>
        <div style="padding: 0 10px;">
            <img src="https://i.ibb.co/GWRyMCW/hogwarts-logo.png" height="30">
        </div>
        <div style="flex-grow: 1; background-color: #D3A625; height: 2px;"></div>
    </div>
    """
    st.markdown(divider, unsafe_allow_html=True)

# ---- Helper Functions ----

# Function to download results
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #D3A625; text-decoration: none; border: 2px solid #D3A625; padding: 8px 12px; border-radius: 5px; background-color: #740001;">📥 {text}</a>'
    return href

# Function to preprocess data
def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    
    # Convert data types
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    return df

# Feature engineering function
def feature_engineering(df, target=None):
    # Create feature and target dataframes
    if target and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            
        return X, y
    else:
        # Handle categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        return df, None

# Function to train model
def train_model(X_train, y_train, model_type):
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
    elif model_type == "K-Means Clustering":
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X_train)
    
    return model

# Function to evaluate model
def evaluate_model(model, X_test, y_test=None, model_type=None):
    results = {}
    
    if model_type == "Linear Regression":
        y_pred = model.predict(X_test)
        results["Mean Squared Error"] = mean_squared_error(y_test, y_pred)
        results["R² Score"] = r2_score(y_test, y_pred)
        results["Predictions"] = y_pred
    
    elif model_type == "Logistic Regression":
        y_pred = model.predict(X_test)
        results["Accuracy"] = accuracy_score(y_test, y_pred)
        results["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
        results["Predictions"] = y_pred
    
    elif model_type == "K-Means Clustering":
        results["Cluster Labels"] = model.predict(X_test)
        results["Cluster Centers"] = model.cluster_centers_
    
    return results

# ---- Main Application ----

def main():
    # Set Harry Potter theme
    set_harry_potter_theme()
    
    # Session state initialization
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = False
    if 'features_engineered' not in st.session_state:
        st.session_state.features_engineered = False
    if 'data_split' not in st.session_state:
        st.session_state.data_split = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'evaluation_done' not in st.session_state:
        st.session_state.evaluation_done = False
    if 'step' not in st.session_state:
        st.session_state.step = 0
    
    # Sidebar setup
    with st.sidebar:
        st.image(load_hp_image("hogwarts_logo"), width=150)
        st.markdown("<h2 style='text-align: center; font-family: Lumos, sans-serif;'>Financial Wizard's Chamber</h2>", unsafe_allow_html=True)
        hp_divider()
        
        # Data source selection
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Choose Your Magical Source</h3>", unsafe_allow_html=True)
        data_source = st.radio("", ("Upload Kragle Dataset 📚", "Fetch from Yahoo Finance 🧙‍♂️"))
        
        if data_source == "Upload Kragle Dataset 📚":
            uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
            if uploaded_file is not None and st.button("🪄 Cast Loading Spell"):
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.step = 1
        
        else:  # Yahoo Finance option
            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Choose Stock Ticker</h4>", unsafe_allow_html=True)
            ticker = st.text_input("Enter Stock Symbol", "AAPL")
            
            st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Select Time Period</h4>", unsafe_allow_html=True)
            period = st.selectbox("", ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"))
            
            if st.button("🪄 Accio Stock Data!"):
                with st.spinner("Summoning data from the financial realm..."):
                    try:
                        stock = yf.Ticker(ticker)
                        df = stock.history(period=period)
                        df.reset_index(inplace=True)
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.step = 1
                        hp_notification("success", f"Successfully summoned {ticker} data!")
                    except Exception as e:
                        hp_notification("error", f"Failed to summon data: {e}")
        
        hp_divider()
        
        # Model selection
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Choose Your Magical Model</h3>", unsafe_allow_html=True)
        model_type = st.selectbox("", ("Linear Regression", "Logistic Regression", "K-Means Clustering"))
        st.session_state.model_type = model_type
        
        # House selection for theme - fun feature!
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Select Your Hogwarts House</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Gryffindor", key="gryffindor"):
                st.session_state.house = "gryffindor"
                st.image(load_hp_image("gryffindor"), width=80)
            if st.button("Ravenclaw", key="ravenclaw"):
                st.session_state.house = "ravenclaw"
                st.image(load_hp_image("ravenclaw"), width=80)
        with col2:
            if st.button("Slytherin", key="slytherin"):
                st.session_state.house = "slytherin"
                st.image(load_hp_image("slytherin"), width=80)
            if st.button("Hufflepuff", key="hufflepuff"):
                st.session_state.house = "hufflepuff"
                st.image(load_hp_image("hufflepuff"), width=80)
            
    # Main content
    if st.session_state.step == 0:
        # Welcome page
        display_hp_title("The Financial Sorcerer's Apprentice", "welcome")
        
        st.markdown("""
        <div style="text-align:center; font-family: 'Lumos', sans-serif; margin: 2rem 0;">
            <p style="font-size: 1.2rem;">Welcome to Hogwarts School of Financial Wizardry</p>
            <p>Where data becomes magic and predictions turn to gold!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(load_hp_image("sorting_hat"), width=300)
        
        st.markdown("""
        <div style="text-align:center; font-family: 'Lumos', sans-serif; margin: 2rem 0;">
            <p>Begin your journey by selecting a dataset from the sidebar.</p>
            <p>The Sorting Hat will guide you through your machine learning adventure!</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.step == 1:
        # Step 1: Data Loading
        display_hp_title("The Magical Dataset Chamber", "data_load")
        
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Your Summoned Data Appears!</h3>", unsafe_allow_html=True)
        
        st.dataframe(st.session_state.df.head())
        
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Data Anatomy</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"📊 Rows: {st.session_state.df.shape[0]}")
            st.write(f"📈 Columns: {st.session_state.df.shape[1]}")
        with col2:
            st.write(f"🧪 Data Types: {len(st.session_state.df.dtypes.unique())} different types")
            st.write(f"🔮 Missing Values: {st.session_state.df.isna().sum().sum()}")
        
        if st.button("🪄 Proceed to Preprocessing"):
            hp_notification("success", "Data successfully loaded. Proceeding to the Cleansing Chamber!")
            st.session_state.step = 2
    
    elif st.session_state.step == 2:
        # Step 2: Preprocessing
        display_hp_title("The Cleansing Chamber", "preprocessing")
        
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Purify Your Data with Magical Cleansing</h3>", unsafe_allow_html=True)
        
        # Display missing values before preprocessing
        st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Dark Magic (Missing Values) Before Cleansing:</h4>", unsafe_allow_html=True)
        missing_before = st.session_state.df.isna().sum()
        st.write(missing_before[missing_before > 0] if missing_before.sum() > 0 else "No missing values detected")
        
        # Display data types before preprocessing
        st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Data Type Enchantments:</h4>", unsafe_allow_html=True)
        st.write(st.session_state.df.dtypes)
        
        if st.button("🪄 Cast Cleansing Spell"):
            with st.spinner("Performing magical cleansing..."):
                # Preprocess data
                st.session_state.df_clean = preprocess_data(st.session_state.df)
                st.session_state.preprocessed = True
                
                # Display statistics after preprocessing
                st.markdown("<h4 style='font-family: Lumos, sans-serif;'>After Cleansing:</h4>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"🧹 Rows after cleansing: {st.session_state.df_clean.shape[0]}")
                    st.write(f"🧙‍♂️ Rows removed: {st.session_state.df.shape[0] - st.session_state.df_clean.shape[0]}")
                
                with col2:
                    missing_after = st.session_state.df_clean.isna().sum().sum()
                    st.write(f"✨ Missing values after: {missing_after}")
                
                hp_notification("success", "Cleansing spell successful! Your data is now purified.")
                st.session_state.step = 3
    
    elif st.session_state.step == 3:
        # Step 3: Feature Engineering
        display_hp_title("The Transfiguration Chamber", "feature_engineering")
        
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Transform Your Features with Magical Transfiguration</h3>", unsafe_allow_html=True)
        
        # Display available columns
        st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Available Magical Components:</h4>", unsafe_allow_html=True)
        st.write(", ".join(st.session_state.df_clean.columns.tolist()))
        
        # Select target variable if using regression or classification
        if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
            target_col = st.selectbox("🎯 Select Target Variable (The prophecy you wish to predict)", 
                                      st.session_state.df_clean.columns)
        else:
            target_col = None
        
        if st.button("🪄 Cast Transfiguration Spell"):
            with st.spinner("Transforming features with advanced magic..."):
                # Feature engineering
                if target_col:
                    X, y = feature_engineering(st.session_state.df_clean, target_col)
                    st.session_state.X = X
                    st.session_state.y = y
                    
                    # Display feature and target information
                    st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Feature Transformation Results:</h4>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"🧙‍♂️ Features (X): {X.shape[1]} columns")
                        st.dataframe(X.head())
                    
                    with col2:
                        st.write(f"🔮 Target (y): {target_col}")
                        if y is not None:
                            if len(np.unique(y)) <= 10:  # If few unique values, show distribution
                                fig = px.pie(values=y.value_counts().values, 
                                            names=y.value_counts().index, 
                                            title=f"Distribution of {target_col}")
                                st.plotly_chart(fig)
                            else:  # Else show histogram
                                fig = px.histogram(y, title=f"Distribution of {target_col}")
                                st.plotly_chart(fig)
                    
                else:  # For clustering, no target variable
                    X, _ = feature_engineering(st.session_state.df_clean)
                    st.session_state.X = X
                    
                    # Display feature information
                    st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Feature Transformation Results:</h4>", unsafe_allow_html=True)
                    st.write(f"🧙‍♂️ Features for Clustering: {X.shape[1]} columns")
                    st.dataframe(X.head())
                
                st.session_state.features_engineered = True
                hp_notification("success", "Feature transfiguration complete! Your data is now ready for training.")
                st.session_state.step = 4
    
    elif st.session_state.step == 4:
        # Step 4: Train/Test Split
        display_hp_title("The Divination Chamber", "train_test")
        
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Dividing Your Data for Training and Testing</h3>", unsafe_allow_html=True)
        
        # Test size slider
        test_size = st.slider("🔮 Test Size Portion (The portion of data to test your prophecy)", 0.1, 0.5, 0.2, 0.05)
        
        if st.button("🪄 Cast Divination Spell"):
            with st.spinner("Dividing data with mystical precision..."):
                # Split data
                if hasattr(st.session_state, 'y') and st.session_state.y is not None:
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state.X, st.session_state.y, test_size=test_size, random_state=42)
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    # Visualize the split
                    st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Data Split Results:</h4>", unsafe_allow_html=True)
                    
                    split_data = {
                        'Set': ['Training Data', 'Testing Data'],
                        'Size': [X_train.shape[0], X_test.shape[0]]
                    }
                    split_df = pd.DataFrame(split_data)
                    
                    fig = px.pie(split_df, values='Size', names='Set', 
                                title='Training and Testing Data Split',
                                color_discrete_sequence=['#D3A625', '#740001'])
                    st.plotly_chart(fig)
                    
                else:  # For clustering
                    X_train, X_test = train_test_split(
                        st.session_state.X, test_size=test_size, random_state=42)
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    
                    # Visualize the split
                    st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Data Split Results:</h4>", unsafe_allow_html=True)
                    
                    split_data = {
                        'Set': ['Training Data', 'Testing Data'],
                        'Size': [X_train.shape[0], X_test.shape[0]]
                    }
                    split_df = pd.DataFrame(split_data)
                    
                    fig = px.pie(split_df, values='Size', names='Set', 
                                title='Training and Testing Data Split',
                                color_discrete_sequence=['#D3A625', '#740001'])
                    st.plotly_chart(fig)
                
                st.session_state.data_split = True
                hp_notification("success", "Divination complete! Your data has been split into training and testing sets.")
                st.session_state.step = 5
    
    elif st.session_state.step == 5:
        # Step 5: Model Training
        display_hp_title("The Spell Casting Chamber", "model_training")
        
        st.markdown("<h3 style='font-family: Lumos, sans-serif;'>Training Your Magical Model</h3>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="font-family: 'Lumos', sans-serif; margin: 1rem 0;">
            <p>You have chosen the <span style="color: #D3A625; font-weight: bold;">{st.session_state.model_type}</span> spell.</p>
            <p>This powerful magic will {
            "predict numerical values based on patterns in your data" if st.session_state.model_type == "Linear Regression" 
            else "classify data into distinct categories" if st.session_state.model_type == "Logistic Regression"
            else "find natural groupings in your data without prior labels"}.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🪄 Cast Training Spell"):
            with st.spinner("Casting powerful magical training spell..."):
                # Train model
                if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                    model = train_model(st.session_state.X_train, st.session_state.y_train, st.session_state.model_type)
                else:  # KMeans
                    model = train_model(st.session_state.X_train, None, st.session_state.model_type)
                
                st.session_state.model = model
                
                # Display training information
                st.markdown("<h4 style='font-family: Lumos, sans-serif;'>Training Complete!</h4>", unsafe_allow_html=True)
                
                if st.session_state.model_type == "Linear Regression":
                    st.write("📊 Model coefficients:")
                    coef_df = pd.DataFrame({
                        'Feature': st.session_state.X.columns,
                        'Coefficient': model.coef_
                    }).sort_values(by='Coefficient', ascending=False)
                    
                    fig = px.bar(coef_df, x='Feature', y='Coefficient', 
                                title='Feature Importance',
                                color_discrete_sequence=['#D3A625'])
                    st.plotly_chart(fig)
                    
                elif st.session_state.model_type == "Logistic Regression":
                    st.write("📊 Model coefficients:")
                    try:
                        coef_df = pd.DataFrame({
                            'Feature': st.session_state.X.columns,
                            'Coefficient': model.coef_[0]
                        }).sort_values(by='Coefficient', ascending=False)
                        
                        fig = px.bar(coef_df, x='Feature', y='Coefficient', 
                                    title='Feature Importance',
                                    color_discrete_sequence=['#D3A625'])
                        st.plotly_chart(fig)
                    except:
                        st.write("Multi-class logistic regression - coefficients not displayed")
                
                elif st.session_state.model_type == "K-Means Clustering":
                    st.write("🧙‍♂️ Cluster centers
