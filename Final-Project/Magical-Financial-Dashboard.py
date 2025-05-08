import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import base64
from datetime import datetime, timedelta
import time

# Set page configuration
st.set_page_config(
    page_title="Hogwarts Financial Wizardry",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Harry Potter theme
def add_harry_potter_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap');
        
        /* Main background */
        .stApp {
            background-image: url("https://wallpaperaccess.com/full/1104921.jpg");
            background-size: cover;
            background-attachment: fixed;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Cinzel', serif;
            color: #d4af37 !important;
            text-shadow: 2px 2px 4px #000000;
        }
        
        h1 {
            font-size: 3rem !important;
        }
        
        /* Text styling */
        p, li, div {
            font-family: 'Bookman Old Style', serif;
            color: #f0f0f0;
        }
        
        /* Cards with glass effect */
        .css-1r6slb0, .css-12w0qpk {
            background-color: rgba(0, 0, 0, 0.7) !important;
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(212, 175, 55, 0.3);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: rgba(40, 42, 54, 0.8);
            border-right: 1px solid #d4af37;
        }
        
        /* Buttons */
        .stButton>button {
            font-family: 'Cinzel', serif;
            background-color: #740001;
            color: #d4af37;
            border: 2px solid #d4af37;
            border-radius: 5px;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #d4af37;
            color: #740001;
            transform: scale(1.05);
            box-shadow: 0 0 15px #d4af37;
        }
        
        /* Slider */
        .stSlider {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Select boxes */
        .stSelectbox label, .stMultiSelect label {
            color: #d4af37 !important;
            font-family: 'Cinzel', serif;
        }
        
        .stSelectbox>div>div, .stMultiSelect>div>div {
            background-color: rgba(25, 25, 25, 0.9);
            border: 1px solid #d4af37;
            color: #d4af37;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #d4af37;
        }
        
        /* House colors */
        .gryffindor {
            color: #740001 !important;
            background-color: #d3a625 !important;
        }
        
        .slytherin {
            color: #1a472a !important;
            background-color: #5d5d5d !important;
        }
        
        .ravenclaw {
            color: #0e1a40 !important;
            background-color: #946b2d !important;
        }
        
        .hufflepuff {
            color: #372e29 !important;
            background-color: #ecb939 !important;
        }
        
        /* Animation for loading */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .floating {
            animation: float 3s ease-in-out infinite;
            display: inline-block;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 5px 5px 0 0;
            border: 1px solid #d4af37;
            border-bottom: none;
            color: #d4af37;
            font-family: 'Cinzel', serif;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(116, 0, 1, 0.8) !important;
            color: #d4af37 !important;
        }
    </style>
    """, unsafe_allow_html=True)

add_harry_potter_css()

# Custom loader function
def show_loading_animation(text="Casting financial spells"):
    with st.empty():
        for i in range(5):
            st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; margin: 2rem 0;">
                <h3 class="floating" style="margin: 0;">
                    {text} {"." * (i % 4 + 1)}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)

# Function to get magical background image
def get_magic_background(name):
    backgrounds = {
        "hogwarts": "https://wallpaperaccess.com/full/1104921.jpg",
        "diagon_alley": "https://wallpaperaccess.com/full/232174.jpg",
        "marauders_map": "https://wallpaperaccess.com/full/4755707.jpg",
        "quidditch": "https://wallpaperaccess.com/full/279565.jpg"
    }
    return backgrounds.get(name, backgrounds["hogwarts"])

# Function to show magical toast notification
def show_magic_toast(icon, text):
    st.markdown(f"""
    <div style="position: fixed; top: 20px; right: 20px; background-color: rgba(0,0,0,0.8); 
                border: 2px solid #d4af37; border-radius: 10px; padding: 10px; z-index: 9999;
                animation: float 3s ease-in-out infinite; display: flex; align-items: center;">
        <span style="font-size: 24px; margin-right: 10px;">{icon}</span>
        <span style="font-family: 'Cinzel', serif; color: #d4af37;">{text}</span>
    </div>
    """, unsafe_allow_html=True)

# Function to display a magical card
def magical_card(title, content, icon="✨"):
    st.markdown(f"""
    <div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 15px; 
                margin-bottom: 20px; border: 1px solid #d4af37; backdrop-filter: blur(5px);">
        <h3 style="display: flex; align-items: center;">
            <span style="margin-right: 10px;">{icon}</span> {title}
        </h3>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

# Function to convert a dataframe to styled HTML
def df_to_styled_html(df, title="Magical Data"):
    # Get house color based on title
    color_class = "gryffindor"  # default
    if "slytherin" in title.lower():
        color_class = "slytherin"
    elif "ravenclaw" in title.lower():
        color_class = "ravenclaw"
    elif "hufflepuff" in title.lower():
        color_class = "hufflepuff"
    
    # Convert to HTML with styling
    html = f"""
    <div style="margin-bottom: 20px;">
        <h3>{title}</h3>
        <div style="max-height: 400px; overflow-y: auto; border: 1px solid #d4af37; border-radius: 5px;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr class="{color_class}" style="position: sticky; top: 0;">
    """
    
    # Add headers
    for col in df.columns:
        html += f"<th style='padding: 8px; text-align: left; border-bottom: 2px solid #d4af37;'>{col}</th>"
    
    html += """
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add rows with alternating background
    for i, row in df.iterrows():
        bg_color = "rgba(20, 20, 20, 0.7)" if i % 2 == 0 else "rgba(40, 40, 40, 0.7)"
        html += f"<tr style='background-color: {bg_color};'>"
        
        for val in row:
            html += f"<td style='padding: 8px; border-bottom: 1px solid rgba(212, 175, 55, 0.3);'>{val}</td>"
        
        html += "</tr>"
    
    html += """
                </tbody>
            </table>
        </div>
    </div>
    """
    
    return html

# Function to create animated header
def animated_header(text):
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 class="floating">{text}</h1>
        <div style="background: linear-gradient(90deg, rgba(212,175,55,0), rgba(212,175,55,0.8), rgba(212,175,55,0)); 
                    height: 2px; width: 100%; margin: 10px 0;"></div>
    </div>
    """, unsafe_allow_html=True)

# Function to generate a Harry Potter-themed placeholder when no data is available
def display_hp_placeholder():
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; 
                height: 300px; text-align: center; background-color: rgba(0,0,0,0.7); 
                border-radius: 10px; padding: 20px; border: 1px dashed #d4af37;">
        <img src="https://i.gifer.com/7AM.gif" style="max-width: 150px; margin-bottom: 20px;">
        <h3>Accio Data!</h3>
        <p>Looks like someone cast "Evanesco" on your data. Upload a file or select some stocks to begin.</p>
    </div>
    """, unsafe_allow_html=True)

# ML Model Functions
def run_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    coefficients = model.coef_
    return y_test, y_pred, score, coefficients, model

def run_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    return y_test, y_pred, score, model

def run_kmeans(X, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans, X_scaled

# Function to create Harry Potter-themed plots
def create_hp_themed_plot(data, plot_type, title, x_col=None, y_col=None, color_col=None):
    # Harry Potter color palette
    hp_colors = ["#740001", "#d3a625", "#1a472a", "#0e1a40", "#ecb939", "#946b2d", "#5d5d5d"]
    
    if plot_type == "line":
        fig = px.line(data, x=x_col, y=y_col, title=title, color_discrete_sequence=hp_colors)
    elif plot_type == "bar":
        fig = px.bar(data, x=x_col, y=y_col, title=title, color_discrete_sequence=hp_colors)
    elif plot_type == "scatter":
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=hp_colors)
    elif plot_type == "pie":
        fig = px.pie(data, names=x_col, values=y_col, title=title, color_discrete_sequence=hp_colors)
    else:
        fig = px.line(data, title="Default Plot", color_discrete_sequence=hp_colors)
    
    # Apply Harry Potter theme
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0.5)",
        plot_bgcolor="rgba(20,20,20,0.5)",
        font_family="Bookman Old Style",
        font_color="#d4af37",
        title_font_family="Cinzel",
        title_font_color="#d4af37",
        legend_title_font_color="#d4af37",
        title_font_size=22,
        xaxis=dict(
            title_font_family="Cinzel",
            title_font_color="#d4af37",
            tickfont_color="#d4af37",
            gridcolor="rgba(212,175,55,0.2)"
        ),
        yaxis=dict(
            title_font_family="Cinzel",
            title_font_color="#d4af37",
            tickfont_color="#d4af37",
            gridcolor="rgba(212,175,55,0.2)"
        )
    )
    
    # Add subtle animation
    fig.update_traces(
        mode="markers+lines" if plot_type == "line" else None,
        marker=dict(size=8, symbol="diamond") if plot_type in ["scatter", "line"] else None,
    )
    
    return fig

# Create sidebar
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <img src="https://i.pinimg.com/originals/06/7a/5a/067a5a19f5d5529b5639c032d5ea853f.gif" style="width: 100%;">
    <h2>Wizard's Financial Tools</h2>
</div>
""", unsafe_allow_html=True)

# Add background selector
st.sidebar.markdown("### Change Magical Background")
background_option = st.sidebar.selectbox("Select Background", 
                                         ["Hogwarts Castle", "Diagon Alley", "Marauder's Map", "Quidditch Pitch"], 
                                         key="bg_select")

bg_mapping = {
    "Hogwarts Castle": "hogwarts",
    "Diagon Alley": "diagon_alley",
    "Marauder's Map": "marauders_map",
    "Quidditch Pitch": "quidditch"
}

# Apply the selected background
st.markdown(f"""
<style>
.stApp {{
    background-image: url("{get_magic_background(bg_mapping[background_option])}");
    background-size: cover;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)

# Sidebar options
st.sidebar.markdown("### Wizarding Tools")
app_mode = st.sidebar.selectbox("Choose Your Path", 
                              ["Welcome to Hogwarts", "Financial Crystal Ball", "Magical Stocks", "Data Transfiguration", "Wizard Analytics"], 
                              key="app_mode")

# Sidebar info
st.sidebar.markdown("""
<div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 15px; margin-top: 20px; border: 1px solid #d4af37;">
    <h3>Wizarding Houses & Markets</h3>
    <p>🦁 <span style="color: #d3a625;">Gryffindor</span>: High-risk, high-reward investments</p>
    <p>🐍 <span style="color: #1a472a;">Slytherin</span>: Strategic long-term investments</p>
    <p>🦅 <span style="color: #946b2d;">Ravenclaw</span>: Analytical approach, research-based</p>
    <p>🦡 <span style="color: #ecb939;">Hufflepuff</span>: Balanced, diversified portfolios</p>
</div>
""", unsafe_allow_html=True)

# Add animated separator
st.sidebar.markdown("""
<div style="display: flex; justify-content: center; margin: 20px 0;">
    <div style="text-align: center; font-size: 24px;" class="floating">
        ⚡ ✨ 🔮
    </div>
</div>
""", unsafe_allow_html=True)

# Session state to store uploaded data
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

# Main page content based on selected mode
if app_mode == "Welcome to Hogwarts":
    animated_header("Hogwarts School of Financial Wizardry")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #d4af37;">
            <h2>Welcome, Young Wizard! 🧙‍♂️</h2>
            <p>The ancient art of Financial Wizardry awaits you. This magical dashboard will help you analyze financial data with the power of machine learning and the charm of the wizarding world.</p>
            
            <h3>Features of this Magical Tool:</h3>
            <ul>
                <li>📊 <strong>Financial Crystal Ball:</strong> Predict future trends using Linear Regression</li>
                <li>📈 <strong>Magical Stocks:</strong> Track real-time stock market data with YFinance</li>
                <li>🧪 <strong>Data Transfiguration:</strong> Transform and cluster your data with K-Means</li>
                <li>✨ <strong>Wizard Analytics:</strong> Magical visualizations of your financial data</li>
            </ul>
            
            <p>Select your path from the sidebar to begin your journey into the magical world of financial analysis!</p>
        </div>
        
        <div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #d4af37;">
            <h3>Upload Your Financial Grimoire 📚</h3>
            <p>Upload your CSV file to begin analyzing your financial data:</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            show_loading_animation("Translating ancient runes")
            
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = data
                
                st.success("File successfully uploaded and decoded! 🧙‍♂️✨")
                
                # Display data preview in a magical table
                st.markdown("""
                <h3>Preview of Your Magical Data:</h3>
                """, unsafe_allow_html=True)
                
                st.markdown(df_to_styled_html(data.head(), "Scroll of Financial Knowledge"), unsafe_allow_html=True)
                
                # Summary statistics
                st.markdown("""
                <h3>Magical Statistics:</h3>
                """, unsafe_allow_html=True)
                
                st.markdown(df_to_styled_html(data.describe(), "Statistical Enchantments"), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Alas! A spell has gone wrong: {e}")
    
    with col2:
        st.markdown("""
        <div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 20px; border: 1px solid #d4af37; height: 300px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; margin-bottom: 20px;">
            <img src="https://i.gifer.com/74KZ.gif" style="max-width: 100%; margin-bottom: 20px;">
            <h3>The Sorting Hat Recommends</h3>
            <p>Based on market conditions, the Sorting Hat suggests focusing on:</p>
            <div style="font-size: 24px; color: #d3a625; font-weight: bold;">GRYFFINDOR INVESTMENTS</div>
            <p>Brave, high-growth potential stocks are favorable this quarter.</p>
        </div>
        
        <div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 20px; border: 1px solid #d4af37; margin-bottom: 20px;">
            <h3>Daily Prophet: Market News</h3>
            <p><strong>May 8, 2025:</strong> Gringotts announces new Goblin-led investment trusts.</p>
            <p><strong>May 7, 2025:</strong> Weasleys' Wizard Wheezes stock up 12% after new product line.</p>
            <p><strong>May 6, 2025:</strong> Ministry of Magic implements new regulations on magical commodities.</p>
            <p><strong>May 5, 2025:</strong> Nimbus Racing Broom Company announces record profits.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a magical animation
        st.markdown("""
        <div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 20px; border: 1px solid #d4af37; display: flex; justify-content: center; align-items: center;">
            <img src="https://i.gifer.com/FLbS.gif" style="max-width: 100%;">
        </div>
        """, unsafe_allow_html=True)

elif app_mode == "Financial Crystal Ball":
    animated_header("⚡ Financial Crystal Ball ⚡")
    
    st.markdown("""
    <div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #d4af37;">
        <h2>Linear Regression Divination 🔮</h2>
        <p>Use the ancient art of Linear Regression to predict future trends. Professor Trelawney would be impressed!</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <h3>Select Features for Prediction</h3>
            """, unsafe_allow_html=True)
            
            # Select columns for X (features) and y (target)
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            feature_cols = st.multiselect("Select Feature Columns (X)", numeric_columns, key="lr_features")
            target_col = st.selectbox("Select Target Column (y)", numeric_columns, key="lr_target")
            
            if feature_cols and target_col:
                if st.button("Cast Prediction Spell ✨", key="run_lr"):
                    try:
                        show_loading_animation("Consulting the crystal ball")
                        
                        X = data[feature_cols]
                        y = data[target_col]
                        
                        y_test, y_pred, score, coefficients, model = run_linear_regression(X, y)
                        
                        st.session_state.lr_results = {
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'score': score,
                            'coefficients': coefficients,
                            'feature_names': feature_cols,
                            'model': model
                        }
                        
                        magical_card(
                            "Prophecy Accuracy",
                            f"<p>The crystal ball's accuracy: <strong>{score:.4f}</strong> (R² Score)</p>",
                            "🔮"
                        )
                        
                        # Show coefficients
                        coef_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Magical Influence': coefficients
                        })
                        
                        st.markdown(df_to_styled_html(coef_df, "Magical Coefficients"), unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Alas! The crystal ball is cloudy: {e}")
        
        with col2:
            if 'lr_results' in st.session_state:
                results = st.session_state.lr_results
                
                # Create prediction vs actual plot
                prediction_df = pd.DataFrame({
                    'Actual': results['y_test'].values,
                    'Predicted': results['y_pred']
                }).reset_index()
                
                fig = create_hp_themed_plot(
                    prediction_df, 
                    "scatter", 
                    "Crystal Ball Predictions vs Reality",
                    "index",
                    ["Actual", "Predicted"]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create feature importance plot
                importance_df = pd.DataFrame({
                    'Feature': results['feature_names'],
                    'Importance': np.abs(results['coefficients'])
                }).sort_values('Importance', ascending=False)
                
                fig2 = create_hp_themed_plot(
                    importance_df,
                    "bar",
                    "Magical Influence of Features",
                    "Feature",
                    "Importance"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Add prediction tool
                st.markdown("""
                <div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 20px; border: 1px solid #d4af37; margin-top: 20px;">
                    <h3>Make a New Prediction</h3>
                    <p>Enter values for your features to get a prediction:</p>
                </div>
                """, unsafe_allow_html=True)
                
                pred_values = {}
                for feature in results['feature_names']:
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    mean_val = float(data[feature].mean())
                    
                    pred_values[feature] = st.slider(
                        f"{feature}", 
                        min_val, 
                        max_val,
                        mean_val,
                        key=f"pred_{feature}"
                    )
                
                if st.button("Reveal the Future ✨", key="predict_new"):
                    input_features = [[pred_values[f] for f in results['feature_names']]]
                    prediction = results['model'].predict(input_features)[0]
                    
                    st.markdown(f"""
                    <div style="background-color: rgba(116,0,1,0.7); border-radius: 10px; padding: 20px; 
                                border: 2px solid #d4af37; text-align: center; margin-top: 20px; animation: float 3s ease-in-out infinite;">
                        <h2>The Crystal Ball Reveals:</h2>
                        <h1 style="font-size: 3rem; color: #d4af37;">{prediction:.4f}</h1>
                        <p>For {target_col}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Show placeholder when no data is uploaded
        display_hp_placeholder()
        st.markdown("""
        <p style="text-align: center; margin-top: 20px;">
            Upload your financial grimoire in the "Welcome to Hogwarts" section to begin your divination.
        </p>
        """, unsafe_allow_html=True)

elif app_mode == "Magical Stocks":
    animated_header("Magical Stocks Portfolio")
    
    st.markdown(""")
    <div style="background-color: rgba(0,0,0,0.7); border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #d4af37;">
        <h2>Wizard Stock Market 📈</h2>
        <p>Track real-time stock market data with the power of YFinance, as if you had a Time-Turner!</p>
    </div>
    """, unsafe
