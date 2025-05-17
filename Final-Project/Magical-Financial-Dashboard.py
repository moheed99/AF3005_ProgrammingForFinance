import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import plotly.graph_objs as go
import yfinance as yf
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Futuristic Wizarding Finance",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Futuristic wallpapers for houses with holographic/magical tech themes
house_themes = {
    "None": {
        "background": "https://wallpaperaccess.com/full/2825704.gif",  # Cosmic magical starfield with shimmering particles
        "primary": "#00ffaa",
        "secondary": "#9d4edd",
        "text": "#ffffff",
        "accent": "#00e6e6"
    },
    "Gryffindor": {
        "background": "https://wallpaperaccess.com/full/1124640.jpg",  # Red-gold futuristic city with magical elements
        "primary": "#ff3c00",
        "secondary": "#ffd700",
        "text": "#ffffff",
        "accent": "#ff9e00"
    },
    "Slytherin": {
        "background": "https://wallpaperaccess.com/full/1329742.jpg",  # Green cyber-magical environment
        "primary": "#00ff66",
        "secondary": "#c0c0c0",
        "text": "#ffffff",
        "accent": "#008060"
    },
    "Ravenclaw": {
        "background": "https://wallpaperaccess.com/full/3237686.jpg",  # Blue tech-magical starry night
        "primary": "#0099ff",
        "secondary": "#c39c6b", 
        "text": "#ffffff",
        "accent": "#47b5ff"
    },
    "Hufflepuff": {
        "background": "https://wallpaperaccess.com/full/2558051.jpg",  # Yellow cyber-magical environment
        "primary": "#f9c80e",
        "secondary": "#202020",
        "text": "#ffffff",
        "accent": "#e8a200"
    }
}

# Select house theme from sidebar
st.sidebar.title("Futuristic Wizarding Finance")
selected_house = st.sidebar.selectbox("Choose Your House", options=list(house_themes.keys()))
theme = house_themes[selected_house]

# Inject CSS with dynamic colors and holographic elements
css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap');

/* Background and base styles */
body {{
    background-image: url('{theme["background"]}');
    background-size: cover;
    background-attachment: fixed;
    color: {theme["text"]};
    font-family: 'Orbitron', sans-serif;
}}

/* Holographic container effect */
.css-1r6slb0, .css-1d391kg {{
    background-color: rgba(20, 20, 35, 0.7) !important;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid {theme["primary"]}40;
    box-shadow: 
        0 0 15px {theme["primary"]}50,
        inset 0 0 10px {theme["primary"]}30;
    padding: 20px;
    position: relative;
    overflow: hidden;
}}

/* Holographic shimmer effect */
.css-1r6slb0::before, .css-1d391kg::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(
        45deg,
        transparent, {theme["primary"]}10, transparent, {theme["secondary"]}10
    );
    transform: rotate(30deg);
    animation: holoshimmer 6s linear infinite;
    z-index: -1;
}}

@keyframes holoshimmer {{
    0% {{ transform: translateY(-100%) rotate(30deg); }}
    100% {{ transform: translateY(100%) rotate(30deg); }}
}}

/* Headers */
h1, h2, h3, .css-1v0mbdj, .st-b, .css-1d391kg {{
    font-family: 'Cinzel', serif;
    color: {theme["primary"]};
    text-shadow: 0 0 15px {theme["primary"]}80;
    letter-spacing: 2px;
}}

/* Buttons with magical glow effect */
.stButton>button {{
    background: linear-gradient(45deg, {theme["primary"]}99, {theme["secondary"]}99);
    border: none;
    border-radius: 50px;
    color: #121212;
    font-weight: bold;
    padding: 10px 25px;
    font-family: 'Orbitron', sans-serif;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 0 15px {theme["primary"]}80;
    position: relative;
    z-index: 1;
    margin: 10px 0;
}}

.stButton>button:hover {{
    transform: scale(1.05);
    box-shadow: 0 0 35px {theme["primary"]};
}}

/* Input fields with tech glow */
.stTextInput>div>input {{
    background-color: rgba(30, 30, 50, 0.6);
    color: {theme["primary"]};
    border-radius: 50px;
    border: 1px solid {theme["primary"]};
    padding: 10px 20px;
    font-family: 'Orbitron', sans-serif;
    box-shadow: 0 0 10px {theme["primary"]}50;
}}

/* Sidebar styling */
.sidebar .sidebar-content {{
    background-image: url('{theme["background"]}');
    background-size: cover;
    color: {theme["primary"]} !important;
}}

/* Futuristic scrollbar */
::-webkit-scrollbar {{
    width: 8px;
}}
::-webkit-scrollbar-track {{
    background: rgba(20, 20, 35, 0.7);
}}
::-webkit-scrollbar-thumb {{
    background: {theme["primary"]};
    border-radius: 10px;
    box-shadow: 0 0 5px {theme["primary"]};
}}

/* Tooltip and hover effects */
div[data-baseweb="tooltip"] {{
    background-color: rgba(20, 20, 35, 0.9) !important;
    border: 1px solid {theme["primary"]} !important;
    box-shadow: 0 0 10px {theme["primary"]} !important;
}}

/* Magical icon pulsing */
@keyframes pulse {{
    0% {{ transform: scale(1); opacity: 0.8; }}
    50% {{ transform: scale(1.2); opacity: 1; }}
    100% {{ transform: scale(1); opacity: 0.8; }}
}}
.magic-icon {{
    display: inline-block;
    animation: pulse 2s infinite ease-in-out;
    margin-left: 8px;
}}

/* Special effect for DataFrame */
.dataframe {{
    background-color: rgba(20, 20, 35, 0.7) !important;
    color: {theme["text"]} !important;
    border: 1px solid {theme["primary"]}40 !important;
}}

/* Magic spell loading animation */
@keyframes casting {{
    0% {{ transform: rotate(0deg); filter: hue-rotate(0deg); }}
    100% {{ transform: rotate(360deg); filter: hue-rotate(360deg); }}
}}
.loading-spell {{
    width: 50px;
    height: 50px;
    border-radius: 50%;
    border: 3px solid transparent;
    border-top-color: {theme["primary"]};
    border-bottom-color: {theme["secondary"]};
    animation: casting 1.5s linear infinite;
    margin: 20px auto;
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Magic icons
WAND = "✨"
CRYSTAL = "🔮"
SPELL = "⚡"

# Sidebar navigation
st.sidebar.title(f"Magic Finance {WAND}")
page = st.sidebar.radio("", ["Welcome", "Data Explorer", "Prediction Spells", "Market Scrying"])

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "ticker_data" not in st.session_state:
    st.session_state.ticker_data = None

def load_csv_data():
    uploaded_file = st.sidebar.file_uploader("Upload financial scroll (CSV)", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.df = data
            st.sidebar.success(f"{SPELL} Scroll decoded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to decode: {e}")

def load_stock_data():
    ticker = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    if st.sidebar.button(f"Summon Data {CRYSTAL}"):
        try:
            with st.spinner("Casting summoning spell..."):
                data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.sidebar.error("No data found in the crystal ball.")
                st.session_state.ticker_data = None
            else:
                data.reset_index(inplace=True)
                st.session_state.ticker_data = data
                st.sidebar.success(f"{SPELL} Vision of {ticker} manifested!")
        except Exception as e:
            st.sidebar.error(f"Spell failed: {e}")

def reset_data():
    st.session_state.df = None
    st.session_state.ticker_data = None

def welcome_page():
    st.title(f"Futuristic Wizarding Finance {WAND}")
    
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; margin-bottom:30px;">
            <h2 style="color:{theme["primary"]}; text-shadow: 0 0 20px {theme["primary"]};">Where Technology Meets Magical Finance</h2>
            <p style="font-size:18px;">Harness the power of magical algorithms and future-tech divination</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style="padding:20px;">
                <h3>Magical Capabilities {CRYSTAL}</h3>
                <ul>
                    <li>Upload financial scrolls or conjure market data</li>
                    <li>Cast powerful prediction spells</li>
                    <li>Visualize financial futures with enchanted charts</li>
                    <li>Experience your house's unique magical interface</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="padding:20px;">
                <h3>Getting Started {WAND}</h3>
                <ol>
                    <li>Select your house theme in the sidebar</li>
                    <li>Navigate to Data Explorer to summon your data</li>
                    <li>Cast prediction spells with ML models</li>
                    <li>Scry the markets for real-time insights</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:40px;">
            <span style="font-size:24px; color:{theme["accent"]};">
                "Fortune favors the financially magical"
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def data_explorer():
    st.title(f"Data Explorer {CRYSTAL}")
    
    data_source = st.selectbox("Choose Your Data Source", 
                              ["Upload Financial Scroll (CSV)", "Summon Market Data"])
    
    if data_source == "Upload Financial Scroll (CSV)":
        load_csv_data()
    else:
        load_stock_data()
    
    if st.button(f"Clear Crystal Ball {WAND}"):
        reset_data()
        st.experimental_rerun()
    
    df = st.session_state.df
    ticker_data = st.session_state.ticker_data
    
    if df is not None:
        st.subheader("Scroll Contents")
        st.dataframe(df.head())
        
        st.markdown("### Numerical Insights")
        st.write(df.describe())
        
        # Show plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select column to visualize", numeric_cols)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[selected_col], 
                marker_color=theme["primary"],
                opacity=0.7
            ))
            fig.update_layout(
                title=f"Distribution of {selected_col}",
                xaxis_title=selected_col,
                yaxis_title="Frequency",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=theme["text"])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif ticker_data is not None:
        st.subheader("Market Vision")
        st.dataframe(ticker_data.head())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ticker_data['Date'], 
            y=ticker_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color=theme["primary"], width=2)
        ))
        fig.update_layout(
            title="Price Movements",
            xaxis_title="Date",
            yaxis_title="Price",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme["text"])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Summon data using the options above to begin your magical analysis.")

def prediction_spells():
    st.title(f"Prediction Spells {SPELL}")
    
    df = st.session_state.df
    ticker_data = st.session_state.ticker_data
    
    dataset_options = []
    if df is not None:
        dataset_options.append("Uploaded Scroll")
    if ticker_data is not None:
        dataset_options.append("Market Vision")
        
    if not dataset_options:
        st.info("First summon data in the 'Data Explorer' to cast prediction spells.")
        return
    
    dataset_choice = st.selectbox("Select Dataset for Spell Casting", options=dataset_options)
    data_df = df.copy() if dataset_choice == "Uploaded Scroll" else ticker_data.copy()
    
    spell_type = st.selectbox("Select Prediction Spell", 
                             ["Linear Premonition (Regression)", 
                              "Binary Oracle (Logistic Regression)", 
                              "Pattern Discovery (Clustering)"])
    
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if spell_type == "Linear Premonition (Regression)":
        st.markdown("### Linear Premonition Spell")
        st.markdown("Predicts numerical values based on patterns in your data.")
        
        target = st.selectbox("Select Target Variable", options=numeric_cols)
        features = st.multiselect("Select Feature Variables", 
                                 [col for col in numeric_cols if col != target])
        
        if st.button(f"Cast Linear Spell {WAND}"):
            if not features:
                st.warning("Select at least one feature to power the spell.")
                return
                
            with st.spinner("Casting prediction spell..."):
                X = data_df[features].fillna(0)
                y = data_df[target].fillna(0)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                
                st.success(f"Spell Accuracy: {1-mse/np.var(y_test):.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test, y=preds,
                    mode='markers',
                    marker=dict(color=theme["primary"]),
                    name='Predictions'
                ))
                fig.add_trace(go.Scatter(
                    x=[min(y_test), max(y_test)],
                    y=[min(y_test), max(y_test)],
                    mode='lines',
                    line=dict(color=theme["secondary"]),
                    name='Perfect Prediction'
                ))
                fig.update_layout(
                    title="Prediction Results",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=theme["text"])
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif spell_type == "Binary Oracle (Logistic Regression)":
        st.markdown("### Binary Oracle Spell")
        st.markdown("Predicts outcomes with two possibilities.")
        
        binary_cols = []
        for col in numeric_cols:
            if data_df[col].nunique() <= 2:
                binary_cols.append(col)
        
        if not binary_cols:
            st.warning("No binary columns found for the oracle to work with.")
            return
            
        target = st.selectbox("Select Binary Target", options=binary_cols)
        features = st.multiselect("Select Features for Divination", 
                                 [c for c in numeric_cols if c != target])
        
        if st.button(f"Cast Oracle Spell {WAND}"):
            if not features:
                st.warning("Select features to power the oracle.")
                return
                
            with st.spinner("Oracle is divining the future..."):
                X = data_df[features].fillna(0)
                y = data_df[target].fillna(0)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                st.success(f"Oracle Accuracy: {acc:.2f}")
                
                # Simplified confusion matrix display
                cm = pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted'])
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm.values,
                    x=['0', '1'] if cm.shape[1] == 2 else ['0'],
                    y=['0', '1'] if cm.shape[0] == 2 else ['0'],
                    colorscale=[[0, theme["primary"]], [1, theme["secondary"]]],
                    showscale=False
                ))
                fig.update_layout(
                    title="Oracle Vision Results",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=theme["text"])
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif spell_type == "Pattern Discovery (Clustering)":
        st.markdown("### Pattern Discovery Spell")
        st.markdown("Reveals hidden groups in your data.")
        
        features = st.multiselect("Select Features for Discovery", options=numeric_cols)
        clusters = st.slider("Number of patterns to find", 2, 8, 3)
        
        if st.button(f"Cast Discovery Spell {WAND}"):
            if len(features) < 2:
                st.warning("Select at least two features for pattern discovery.")
                return
                
            with st.spinner("Discovering hidden patterns..."):
                X = data_df[features].fillna(0)
                
                kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                data_df["Pattern"] = labels
                
                fig = go.Figure()
                for i in range(clusters):
                    cluster_data = data_df[data_df["Pattern"] == i]
                    fig.add_trace(go.Scatter(
                        x=cluster_data[features[0]],
                        y=cluster_data[features[1]],
                        mode='markers',
                        marker=dict(size=8),
                        name=f'Pattern {i+1}'
                    ))
                    
                # Add cluster centers
                fig.add_trace(go.Scatter(
                    x=kmeans.cluster_centers_[:, 0],
                    y=kmeans.cluster_centers_[:, 1],
                    mode='markers',
                    marker=dict(
                        color='white',
                        size=12,
                        line=dict(color=theme["primary"], width=2),
                        symbol='diamond'
                    ),
                    name='Pattern Centers'
                ))
                
                fig.update_layout(
                    title="Discovered Patterns",
                    xaxis_title=features[0],
                    yaxis_title=features[1],
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=theme["text"])
                )
                st.plotly_chart(fig, use_container_width=True)

def market_scrying():
    st.title(f"Market Scrying {CRYSTAL}")
    
    ticker = st.text_input("Enter Stock Symbol to Scry", value="TSLA")
    period = st.select_slider("Scrying Period", 
                             options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"])
    
    if st.button(f"Activate Crystal Ball {WAND}"):
        try:
            with st.spinner("Crystal ball activating..."):
                data = yf.download(ticker, period=period, interval="1d")
                
            if data.empty:
                st.error("The crystal ball shows nothing.")
                return
                
            data.reset_index(inplace=True)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Price Vision", "Volume Patterns", "Moving Energies"])
            
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
                fig.update_layout(
                    title=f"{ticker} Price Movements",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=theme["text"])
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=data['Date'],
                    y=data['Volume'],
                    marker_color=theme["primary"],
                    name='Volume'
                ))
                fig.update_layout(
                    title=f"{ticker} Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=theme["text"])
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                # Calculate moving averages
                data['MA10'] = data['Close'].rolling(window=10).mean()
                data['MA30'] = data['Close'].rolling(window=30).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=theme["primary"], width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['MA10'],
                    mode='lines',
                    name='10-Day MA',
                    line=dict(color=theme["secondary"], width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['MA30'],
                    mode='lines',
                    name='30-Day MA',
                    line=dict(color=theme["accent"], width=2)
                ))
                fig.update_layout(
                    title=f"{ticker} Moving Energies",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=theme["text"])
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Crystal ball malfunction: {e}")

# Main navigation router
if page == "Welcome":
    welcome_page()
elif page == "Data Explorer":
    data_explorer()
elif page == "Prediction Spells":
    prediction_spells()
elif page == "Market Scrying":
    market_scrying()

# Footer with magical animator
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; margin-top:30px;">
        <div style="display:inline-block; animation: casting 4s linear infinite;">
            <span style="font-size:30px; color:{theme["primary"]}; text-shadow: 0 0 10px {theme["primary"]}">
                {CRYSTAL} {WAND} {SPELL}
            </span>
        </div>
        <p style="margin-top:15px; font-size:14px; opacity:0.7;">
            Powered by Futuristic Wizarding Finance © {pd.Timestamp.now().year}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
