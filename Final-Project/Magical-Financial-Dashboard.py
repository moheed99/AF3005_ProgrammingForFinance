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

# House theme backgrounds and color schemes
house_themes = {
    "None": {
        "background": "https://i.pinimg.com/originals/56/89/1f/56891f669a0229aa72633cc00eb4e35c.jpg",
        "primary": "#fedd00",
        "secondary": "#662d91",
        "text": "#eee7db",
        "button_bg": "linear-gradient(45deg, #662d91, #fedd00)",
        "button_hover_bg": "linear-gradient(45deg, #fedd00, #662d91)"
    },
    "Gryffindor": {
        "background": "https://i.pinimg.com/originals/06/9f/b6/069fb6b232ca2e2717be306e65aacb65.gif",  # flickering flames gif
        "primary": "#7F0909",  # dark red
        "secondary": "#FFC500",  # gold
        "text": "#fff2cc",
        "button_bg": "linear-gradient(45deg, #7F0909, #FFC500)",
        "button_hover_bg": "linear-gradient(45deg, #FFC500, #7F0909)"
    },
    "Slytherin": {
        "background": "https://i.pinimg.com/originals/c2/6c/30/c26c3047eae36b747ba0033d285c4830.gif", # green mist gif
        "primary": "#1A472A",  # dark green
        "secondary": "#AAAAAA",  # silver/gray
        "text": "#d0f0c0",
        "button_bg": "linear-gradient(45deg, #1A472A, #AAAAAA)",
        "button_hover_bg": "linear-gradient(45deg, #AAAAAA, #1A472A)"
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
body {{
    background-image: url('{theme["background"]}');
    background-size: cover;
    background-attachment: fixed;
    color: {theme["text"]};
    font-family: 'MedievalSharp', cursive;
    margin: 0;
    padding: 0;
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

.stButton>button {{
  background: {theme["button_bg"]};
  border-radius: 12px;
  border: none;
  color: black;
  font-weight: bold;
  padding: 12px 28px;
  font-size: 20px;
  cursor: pointer;
  transition: all 0.4s ease;
  box-shadow: 0 0 15px {theme["primary"]};
  animation: glow 2s infinite alternate;
}}
.stButton>button:hover {{
  background: {theme["button_hover_bg"]};
  box-shadow: 0 0 25px 10px {theme["primary"]};
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

.css-1r6slb0, .css-1d391kg {{
  background-color: rgba(44, 26, 79, 0.85);
  padding: 20px;
  border-radius: 20px;
  box-shadow: 0 0 15px {theme["primary"]};
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

/* Animated magic wand icon */
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

/* Scrollbar styles */
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

# Sidebar page navigation & data source options same as before
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

# Pages below unchanged but with colors from theme, repeated here for clarity

def welcome_page():
    st.title(f"Welcome to Harry Potter Financial Mystics {WAND}")
    st.markdown(
        f"""
        <p style="font-size:18px; color:{theme["text"]};">
        Step into a mystical realm where finance meets wizardry! Harness the power of enchanted data and ancient machine learning spells.
        </p>
        <img src="https://64.media.tumblr.com/07b92df0ebaaae494186922ad7a8809b/tumblr_p8qc4yLOuM1rtsao1o1_400.gif" alt="Harry Potter Magic" width="100%" style="border-radius: 15px;"/>
        <br>
        <h3 style="color:{theme["secondary"]};">What you can do here:</h3>
        <ul style="font-size:18px; color:{theme["text"]};">
        <li>🔮 Upload your financial datasets or fetch real-time stock market data</li>
        <li>🧙‍♂️ Apply powerful machine learning spells like Linear Regression, Logistic Regression, and K-Means Clustering</li>
        <li>📊 Visualize your data and results with enchanting graphs and charts</li>
        <li>✨ Enjoy magical animations and a Harry Potter inspired theme tailored to your house</li>
        </ul>
        """
        , unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### Select Data Source and House Theme in Sidebar")
    st.markdown(f"Go to **Data Exploration** page after loading your data.")

def data_exploration():
    st.title(f"Data Exploration {WAND}")
    data_source_option = st.selectbox("Choose Data Source", ["Upload CSV Dataset", "Fetch Stock Data from Yahoo Finance"])

    if data_source_option == "Upload CSV Dataset":
        load_data_from_upload()
    else:
        load_data_from_stock()

    if st.button("Reset Data"):
        reset_data()
        st.experimental_rerun()

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

# Footer with animated magic button themed by house
st.markdown("---")
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    magic_button_code = f"""
    <style>
    .magic-btn {{
        font-family: 'Creepster', cursive;
        background: {theme["button_bg"]};
        color: black;
        font-size: 24px;
        padding: 16px 60px;
        border-radius: 50px;
        border: none;
        cursor: pointer;
        box-shadow: 0 0 20px {theme["primary"]};
        animation: pulseGlow 2.5s infinite alternate;
        transition: transform 0.3s ease;
    }}
    .magic-btn:hover {{
        box-shadow: 0 0 40px 10px {theme["primary"]};
        transform: scale(1.2) rotate(-5deg);
    }}
    @keyframes pulseGlow {{
        0% {{box-shadow: 0 0 20px {theme["primary"]};}}
        50% {{box-shadow: 0 0 40px 20px {theme["primary"]};}}
        100% {{box-shadow: 0 0 20px {theme["primary"]};}}
    }}
    </style>
    <button class="magic-btn" onclick="alert('May your financial spells always succeed!')">✨ Cast Your Financial Spell ✨</button>
    """
    st.markdown(magic_button_code, unsafe_allow_html=True)
