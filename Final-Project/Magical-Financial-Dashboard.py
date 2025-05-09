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
from PIL import Image
import requests
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="Harry Potter Financial Mystics",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Harry Potter style font - using Google Fonts Creepster for a wizard vibe
# Streamlit cannot load external css easily so use markdown for heading font and inject css

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

# Inject fonts & CSS for Harry potter vibe
remote_css("https://fonts.googleapis.com/css2?family=Creepster&display=swap")
remote_css("https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap")

st.markdown(
    """
<style>
body {
    background-image: url('https://i.pinimg.com/originals/56/89/1f/56891f669a0229aa72633cc00eb4e35c.jpg');
    background-size: cover;
    background-attachment: fixed;
    color: #eee7db;
}
h1, h2, .css-1v0mbdj, .st-b, .css-1d391kg {
    font-family: 'Creepster', cursive;
    color: #fedd00;
    text-shadow:
      -1px -1px 0 #000,
       1px -1px 0 #000,
      -1px  1px 0 #000,
       1px  1px 0 #000;
}
h3 {
    font-family: 'MedievalSharp', cursive;
    color: #bb7722;
    text-shadow:
      1px 1px 2px black;
}
.stButton>button {
  background: linear-gradient(45deg, #662d91, #fedd00);
  border-radius: 12px;
  border: none;
  color: #000;
  font-weight: bold;
  padding: 10px 24px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 18px;
  cursor: pointer;
  transition: background 0.5s ease;
}
.stButton>button:hover {
  background: linear-gradient(45deg, #fedd00, #662d91);
  color: #000;
  box-shadow: 0 0 15px #fedd00;
  transform: scale(1.1);
}
.stTextInput>div>input {
  background-color: #2c1a4f;
  color: #fedd00;
  border-radius: 8px;
  border: 1px solid #fedd00;
  padding: 8px;
}
.css-1r6slb0, .css-1d391kg {
  background-color: rgba(44, 26, 79, 0.85);
  padding: 15px;
  border-radius: 15px;
  box-shadow: 0 0 15px #fedd00;
}
</style>
""",
    unsafe_allow_html=True,
)

# Title / header
st.title("🧙‍♂️ Harry Potter Financial Mystics 🧙‍♀️")
st.markdown("Welcome to the magical realm of financial data analysis! Upload your own Kragle datasets or fetch real-time stock data to unravel market mysteries using the ancient arts of machine learning.")

# Sidebar for inputs
st.sidebar.header("1. Choose Data Source")

data_source = st.sidebar.selectbox("Select Data Source", ["Upload Kragle Dataset (CSV)", "Fetch Real-Time Stock Data (Yahoo Finance)"])

df = None

if data_source == "Upload Kragle Dataset (CSV)":
    uploaded_file = st.sidebar.file_uploader("Upload your financial dataset (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to load dataset: {e}")
elif data_source == "Fetch Real-Time Stock Data (Yahoo Finance)":
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    if st.sidebar.button("Fetch Data"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error("No data found for the provided ticker and date range.")
            else:
                df = data.reset_index()
                st.success(f"Data for {ticker} loaded successfully!")
        except Exception as e:
            st.error(f"Failed to fetch stock data: {e}")

if df is not None:
    st.subheader("Preview of your data:")
    st.dataframe(df.head(10))

    st.markdown("---")
    st.header("2. Choose Machine Learning Model")

    model_choice = st.selectbox("Select ML Model to Apply", ["Linear Regression", "Logistic Regression", "K-Means Clustering"])

    # Prepare data for models
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.markdown(f"**Numeric columns detected for modelling:** {numeric_cols}")

    if model_choice == "Linear Regression":
        st.markdown("Linear Regression predicts a numeric target based on numeric features.")

        # Select target and features
        target = st.selectbox("Select Target Variable (numeric)", options=numeric_cols)
        features = st.multiselect("Select Feature Variables (numeric)", options=[col for col in numeric_cols if col != target])

        if st.button("Run Linear Regression"):
            if target and features:
                X = df[features].fillna(0)
                y = df[target].fillna(0)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)

                st.write(f"Mean Squared Error: {mse:.4f}")

                # Plot actual vs predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=preds, mode='markers', name='Predictions'))
                fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Prediction', line=dict(color='firebrick')))
                fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted",
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#fedd00')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select target and features.")

    elif model_choice == "Logistic Regression":
        st.markdown("Logistic Regression predicts a categorical target (binary classification) from features.")

        # We will look for numeric columns with only 2 unique values to suggest binary target options
        binary_cols = [col for col in numeric_cols if df[col].dropna().nunique() == 2]

        target = st.selectbox("Select Target Variable (binary numeric)", options=binary_cols)
        features = st.multiselect("Select Feature Variables (numeric)", options=[col for col in numeric_cols if col != target])

        if st.button("Run Logistic Regression"):
            if target and features:
                X = df[features].fillna(0)
                y = df[target].fillna(0)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                st.write(f"Accuracy: {acc:.4f}")

                # Confusion matrix plot
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
            else:
                st.warning("Please select target and features.")

    elif model_choice == "K-Means Clustering":
        st.markdown("K-Means Clustering groups data into clusters based on features.")

        features = st.multiselect("Select Feature Variables (numeric)", options=numeric_cols)

        clusters = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)

        if st.button("Run K-Means Clustering"):
            if features:
                X = df[features].fillna(0)

                kmeans = KMeans(n_clusters=clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(X)

                df_clusters = df.copy()
                df_clusters["Cluster"] = cluster_labels

                st.write("Cluster assignment preview:")
                st.dataframe(df_clusters[features + ["Cluster"]].head(20))

                # 2D scatter plot of first two features colored by cluster
                if len(features) >= 2:
                    fig = go.Figure()
                    for cluster_num in range(clusters):
                        cluster_data = df_clusters[df_clusters["Cluster"] == cluster_num]
                        fig.add_trace(go.Scatter(
                            x=cluster_data[features[0]],
                            y=cluster_data[features[1]],
                            mode='markers',
                            name=f'Cluster {cluster_num}',
                            marker=dict(size=10)
                        ))
                    fig.update_layout(title="K-Means Clusters",
                                      xaxis_title=features[0],
                                      yaxis_title=features[1],
                                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                      font_color='#fedd00')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select at least two features for visualization.")
            else:
                st.warning("Please select features for clustering.")

else:
    st.info("Please upload a dataset or fetch stock data to begin.")

# Footer with magical animated button
st.markdown("---")
col1, col2, col3 = st.columns([1,3,1])
with col2:
    # Magic button with animation
    button_code = """
    <style>
    .magic-button {
      background: linear-gradient(45deg, #fedd00, #662d91);
      border: none;
      color: black;
      padding: 15px 50px;
      text-align: center;
      text-decoration: none;
      display: inline-block;
      font-size: 24px;
      font-family: 'Creepster', cursive;
      cursor: pointer;
      box-shadow: 0 0 10px #fedd00;
      border-radius: 50px;
      animation: pulse 2s infinite;
      transition: 0.3s ease-in-out;
    }
    .magic-button:hover {
      box-shadow: 0 0 25px 10px #fedd00;
      transform: scale(1.1);
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 10px #fedd00; }
      50% { box-shadow: 0 0 25px 15px #fedd00; }
      100% { box-shadow: 0 0 10px #fedd00; }
    }
    </style>
    <button class="magic-button" onclick="alert('May the magic of finance be with you!')">✨ Cast Your Financial Spell ✨</button>
    """
    st.markdown(button_code, unsafe_allow_html=True)

