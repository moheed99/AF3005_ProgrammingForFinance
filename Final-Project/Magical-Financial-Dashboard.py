import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import yfinance as yf
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Harry Potter Financial Mystics - Themed",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Updated backgrounds with Harry Potter live photos/gifs
house_themes = {
    "None": {
        "background": "https://wallpapercave.com/wp/wp2182819.jpg", # Hogwarts wallpaper
        "primary": "#fedd00",
        "secondary": "#662d91",
        "text": "#eee7db",
        "button_bg": "linear-gradient(45deg, #662d91, #fedd00)",
        "button_hover_bg": "linear-gradient(45deg, #fedd00, #662d91)"
    },
    "Gryffindor": {
        "background": "https://wallpapercave.com/wp/wp2182820.jpg",  # Gryffindor wallpaper
        "primary": "#7F0909",  # dark red
        "secondary": "#FFC500",  # gold
        "text": "#fff2cc",
        "button_bg": "linear-gradient(45deg, #7F0909, #FFC500)",
        "button_hover_bg": "linear-gradient(45deg, #FFC500, #7F0909)"
    },
    "Slytherin": {
        "background": "https://wallpapercave.com/wp/wp2182821.jpg", # Slytherin wallpaper
        "primary": "#1A472A",  # dark green
        "secondary": "#AAAAAA",  # silver/gray
        "text": "#d0f0c0",
        "button_bg": "linear-gradient(45deg, #1A472A, #AAAAAA)",
        "button_hover_bg": "linear-gradient(45deg, #AAAAAA, #1A472A)"
    },
    "Hufflepuff": {
        "background": "https://wallpapercave.com/wp/wp2182822.jpg", # Hufflepuff wallpaper
        "primary": "#FFD700",  # gold
        "secondary": "#000000",  # black
        "text": "#f0e68c",
        "button_bg": "linear-gradient(45deg, #FFD700, #000000)",
        "button_hover_bg": "linear-gradient(45deg, #000000, #FFD700)"
    },
    "Ravenclaw": {
        "background": "https://wallpapercave.com/wp/wp2182823.jpg", # Ravenclaw wallpaper
        "primary": "#000080",  # navy blue
        "secondary": "#808080",  # gray
        "text": "#add8e6",
        "button_bg": "linear-gradient(45deg, #000080, #808080)",
        "button_hover_bg": "linear-gradient(45deg, #808080, #000080)"
    }
}

# Select house theme from sidebar
st.sidebar.title("Harry Potter Financial Mystics")
selected_house = st.sidebar.selectbox("Choose Your House Theme", options=list(house_themes.keys()))

theme = house_themes[selected_house]

# Inject fonts & CSS with dynamic colors and backgrounds for selected house
def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

remote_css("https://fonts.googleapis.com/css2?family=Creepster&display=swap")
remote_css("https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap")

background_css = f"""
<style>
@keyframes neonBorderMove {{
  0% {{
    filter: drop-shadow(0 0 6px {theme["primary"]});
    box-shadow:
      0 0 10px {theme["primary"]},
      inset 0 0 10px {theme["primary"]};
    border-image-slice: 1;
    border-width: 4px;
    border-style: solid;
    border-image-source: linear-gradient(45deg, {theme["primary"]}, {theme["secondary"]}, {theme["primary"]});
    background-position: 0% 50%;
  }}
  50% {{
    background-position: 100% 50%;
    filter: drop-shadow(0 0 12px {theme["secondary"]});
  }}
  100% {{
    filter: drop-shadow(0 0 6px {theme["primary"]});
    background-position: 0% 50%;
  }}
}}

body {{
    background-image: url('{theme["background"]}');
    background-size: cover;
    background-attachment: fixed;
    color: {theme["text"]};
    font-family: 'MedievalSharp', cursive;
    margin: 0;
    padding: 0;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
}}

h1, h2, h3, .css-1v0mbdj, .st-b, .css-1d391kg {{
    font-family: 'Creepster', cursive;
    color: {theme["primary"]};
    text-shadow:
      -1px -1px 0 #000,
       1px -1px 0 #000,
      -1px  1px 0 #000,
       1px  1px 0 #000;
}}
h3 {{
    font-family: 'MedievalSharp', cursive;
    color: {theme["secondary"]};
    text-shadow: 1px 1px 2px black;
}}

/* Neon animated border and glow for buttons */
.stButton>button {{
  background: {theme["button_bg"]};
  border: 4px solid transparent;
  border-radius: 12px;
  color: black;
  font-weight: bold;
  padding: 12px 28px;
  font-size: 20px;
  cursor: pointer;
  transition: all 0.4s ease;
  box-shadow:
    0 0 10px {theme["primary"]},
    inset 0 0 10px {theme["primary"]};
  animation: neonBorderMove 4s ease-in-out infinite;
  background-origin: border-box;
  background-clip: padding-box, border-box;
  position: relative;
  z-index: 1;
}}
.stButton>button:hover {{
  background: {theme["button_hover_bg"]};
  box-shadow: 0 0 35px 10px {theme["primary"]};
  transform: scale(1.15) rotate(-2deg);
  color: black;
}}

@keyframes glow {{
  0% {{
    box-shadow: 0 0 10px {theme["primary"]};
  }}
  50% {{
    box-shadow: 0 0 25px 15px {theme["primary"]};
  }}
  100% {{
    box-shadow: 0 0 10px {theme["primary"]};
  }}
}}

.stTextInput>div>input {{
  background-color: #2c1a4f;
  color: {theme["primary"]};
  border-radius: 10px;
  border: 1px solid {theme["primary"]};
  padding: 10px;
  font-size: 16px;
}}

.sidebar .sidebar-content {{
  background-image: url('{theme["background"]}');
  background-size: cover;
  background-repeat: no-repeat;
  color: {theme["primary"]} !important;
}}

.css-1aumxhk {{
  color: {theme["primary"]} !important;
}}

@keyframes wand-wiggle {{
  0% {{ transform: rotate(0deg); }}
  25% {{ transform: rotate(20deg); }}
  50% {{ transform: rotate(-20deg); }}
  75% {{ transform: rotate(20deg); }}
  100% {{ transform: rotate(0deg); }}
}}
.wand-icon {{
  display: inline-block;
  animation: wand-wiggle 3s infinite ease-in-out;
  font-size: 1.4rem;
  margin-left: 10px;
}}

::-webkit-scrollbar {{
  width: 10px;
}}
::-webkit-scrollbar-track {{
  background: #2c1a4f;
}}
::-webkit-scrollbar-thumb {{
  background: {theme["primary"]};
  border-radius: 10px;
}}

</style>
"""

st.markdown(background_css, unsafe_allow_html=True)

# Magical wand unicode
WAND = "🪄"

# Sidebar page navigation & data source options
st.sidebar.title("Harry Potter Financial Mystics")
st.sidebar.markdown("### Navigate the pages")
page = st.sidebar.radio("", ["Welcome", "Data Exploration", "ML Models", "Stock Market Live Dashboard"])

if "df" not in st.session_state:
    st.session_state.df = None
if "ticker_data" not in st.session_state:
    st.session_state.ticker_data = None

def load_data_from_upload():
    uploaded_file = st.sidebar.file_uploader("Upload your financial dataset (CSV)", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.df = data
            st.sidebar.success("Dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to load dataset: {e}")

def load_data_from_stock():
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL", key="ticker")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"), key="start_date")
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"), key="end_date")
    if st.sidebar.button("Fetch Data"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.sidebar.error("No data found for the provided ticker and date range.")
                st.session_state.ticker_data = None
            else:
                data.reset_index(inplace=True)
                st.session_state.ticker_data = data
                st.sidebar.success(f"Data for {ticker} loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to fetch stock data: {e}")

def reset_data():
    st.session_state.df = None
    st.session_state.ticker_data = None

def welcome_page():
    st.markdown(
        f"""
        <style>
        .welcome-screen {{
            background-image: url('{theme["background"]}');
            background-size: cover;
            background-position: center;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            text-align: center;
        }}
        .welcome-screen h1 {{
            font-size: 4em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px #000000;
        }}
        .welcome-screen p {{
            font-size: 1.5em;
            max-width: 600px;
            margin-bottom: 30px;
        }}
        .welcome-screen button {{
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }}
        </style>
        <div class="welcome-screen">
            <h1>Welcome to the Futuristic Wizardry World</h1>
            <p>Step into a realm where magic meets technology. Explore the wonders of our futuristic wizardry world and unleash your potential.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def data_exploration():
    st.title(f"Data Exploration {WAND}")
    data_source_option = st.selectbox("Choose Data Source", ["Upload CSV Dataset", "Fetch Stock Data from Yahoo Finance"])

    if data_source_option == "Upload CSV Dataset":
        load_data_from_upload()
    else:
        load_data_from_stock()

    if st.button("Reset Data"):
        reset_data()

    df = st.session_state.df
    ticker_data = st.session_state.ticker_data

    if df is not None:
        st.subheader("Uploaded Dataset Preview")
        st.dataframe(df.head(10))

        st.markdown("### Summary Statistics")
        st.write(df.describe())

        st.markdown("### Data Columns")
        st.write(df.columns.tolist())

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown("### Histograms of Numeric Columns")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, color=theme["primary"], ax=ax)
                ax.set_facecolor('#2c1a4f')
                ax.spines['bottom'].set_color(theme["primary"])
                ax.spines['left'].set_color(theme["primary"])
                ax.tick_params(axis='x', colors=theme["primary"])
                ax.tick_params(axis='y', colors=theme["primary"])
                ax.title.set_color(theme["primary"])
                st.pyplot(fig)
        else:
            st.info("No numeric columns available for histogram plots.")

    elif ticker_data is not None:
        st.subheader("Fetched Stock Data Preview")
        st.dataframe(ticker_data.head(10))

        st.markdown("### Closing Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Close'], mode='lines+markers', name='Close Price', line=dict(color=theme["primary"])))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["primary"])
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Load data using options above to begin exploration.")

def ml_models():
    st.title(f"Machine Learning Models {WAND}")

    df = st.session_state.df
    ticker_data = st.session_state.ticker_data

    dataset_options = []
    if df is not None:
        dataset_options.append("Uploaded Dataset")
    if ticker_data is not None:
        dataset_options.append("Stock Data")
    if not dataset_options:
        st.info("Load data in the 'Data Exploration' page to run ML models.")
        return

    dataset_choice = st.selectbox("Select Dataset for Modeling", options=dataset_options)

    data_df = df.copy() if dataset_choice == "Uploaded Dataset" else ticker_data.copy()
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
    st.markdown(f"**Numeric columns detected:** {numeric_cols}")
    if not numeric_cols:
        st.warning("No numeric columns found in the dataset for modeling.")
        return

    model_choice = st.selectbox("Select Machine Learning Model", ["Linear Regression", "Logistic Regression", "K-Means Clustering"])

    if model_choice == "Linear Regression":
        st.markdown("Linear Regression predicts a numeric target from numeric features.")

        target = st.selectbox("Select Target Variable (numeric)", options=numeric_cols)
        features = st.multiselect("Select Feature Variables (numeric)", options=[col for col in numeric_cols if col != target])

        if st.button("Run Linear Regression"):
            if not features:
                st.warning("Please select at least one feature.")
                return
            X = data_df[features].fillna(0)
            y = data_df[target].fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

            st.success(f"Mean Squared Error: {mse:.4f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=preds, mode='markers', name='Predicted'))
            fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal', line=dict(color='firebrick')))
            fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted",
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["primary"])
            st.plotly_chart(fig, use_container_width=True)

    elif model_choice == "Logistic Regression":
        st.markdown("Logistic Regression predicts a binary target.")

        binary_cols = [col for col in numeric_cols if data_df[col].dropna().nunique() == 2]
        if not binary_cols:
            st.warning("No binary numeric columns available for Logistic Regression target.")
            return

        target = st.selectbox("Select Binary Target Variable", options=binary_cols)
        features = st.multiselect("Select Numeric Features", options=[c for c in numeric_cols if c != target])

        if st.button("Run Logistic Regression"):
            if not features:
                st.warning("Please select at least one feature.")
                return

            X = data_df[features].fillna(0)
            y = data_df[target].fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            st.success(f"Accuracy: {acc:.4f}")

            cm = sns.heatmap(pd.crosstab(y_test, preds), annot=True, cmap="YlOrBr", fmt="d")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(plt.gcf())
            plt.clf()

    elif model_choice == "K-Means Clustering":
        st.markdown("K-Means groups data into clusters.")

        features = st.multiselect("Select Features", options=numeric_cols)
        clusters = st.slider("Number of clusters (k)", 2, 10, 3)

        if st.button("Run K-Means Clustering"):
            if not features:
                st.warning("Please select at least one feature.")
                return

            X = data_df[features].fillna(0)

            kmeans = KMeans(n_clusters=clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            data_df["Cluster"] = labels

            st.write("Cluster assignments preview:")
            st.dataframe(data_df[features + ["Cluster"]].head(20))

            if len(features) >= 2:
                fig = go.Figure()
                for cluster_num in range(clusters):
                    cluster_data = data_df[data_df["Cluster"] == cluster_num]
                    fig.add_trace(go.Scatter(
                        x=cluster_data[features[0]],
                        y=cluster_data[features[1]],
                        mode='markers',
                        name=f'Cluster {cluster_num}',
                        marker=dict(size=12)
                    ))
                fig.update_layout(title="K-Means Clusters",
                                  xaxis_title=features[0],
                                  yaxis_title=features[1],
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["primary"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least two features for cluster visualization.")

def stock_market_dashboard():
    st.title(f"Stock Market Live Dashboard {WAND}")

    ticker = st.text_input("Enter Stock Ticker to Track", value="AAPL", max_chars=10)
    if st.button("Show Stock Data"):
        try:
            data = yf.download(ticker, period="1mo", interval="1d")
            if data.empty:
                st.error(f"No data found for ticker {ticker}")
                return

            data.reset_index(inplace=True)
            st.subheader(f"Live Data for {ticker}")
            st.dataframe(data.tail(10))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines+markers",
                                     name="Close Price", line=dict(color=theme["primary"])))
            fig.update_layout(title=f"{ticker} Close Prices - Last Month",
                              xaxis_title="Date", yaxis_title="Price (USD)",
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["primary"])
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to retrieve stock data: {e}")

# Main navigation
if page == "Welcome":
    welcome_page()
elif page == "Data Exploration":
    data_exploration()
elif page == "ML Models":
    ml_models()
elif page == "Stock Market Live Dashboard":
    stock_market_dashboard()

# Footer with animated magic button themed by house with neon border
st.markdown("---")
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    magic_button_code = f"""
    <style>
    .magic-btn {{
        font-family: 'Creepster', cursive;
        background: {theme["button_bg"]};
        border: 4px solid transparent;
        border-radius: 50px;
        color: black;
        font-size: 24px;
        padding: 16px 60px;
        cursor: pointer;
        box-shadow:
            0 0 20px {theme["primary"]},
            inset 0 0 20px {theme["primary"]};
        animation: neonBorderMove 4s ease-in-out infinite;
        transition: transform 0.3s ease;
        position: relative;
        z-index: 1;
    }}
    .magic-btn:hover {{
        box-shadow: 0 0 40px 10px {theme["primary"]};
        transform: scale(1.2) rotate(-5deg);
    }}
    @keyframes neonBorderMove {{
      0% {{
        filter: drop-shadow(0 0 6px {theme["primary"]});
        box-shadow:
          0 0 10px {theme["primary"]},
          inset 0 0 10px {theme["primary"]};
        background-position: 0% 50%;
      }}
      50% {{
        background-position: 100% 50%;
        filter: drop-shadow(0 0 12px {theme["secondary"]});
      }}
      100% {{
        filter: drop-shadow(0 0 6px {theme["primary"]});
        background-position: 0% 50%;
      }}
    }}
    </style>
    <button class="magic-btn" onclick="alert('May your financial spells always succeed!')">✨ Cast Your Financial Spell ✨</button>
    """
    st.markdown(magic_button_code, unsafe_allow_html=True)
