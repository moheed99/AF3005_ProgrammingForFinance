# Financial Machine Learning Application 💹

This interactive web application provides a user-friendly interface to perform a complete machine learning workflow on financial data. It allows users to load data, preprocess it, engineer features, train various models, evaluate their performance, and visualize the results.

## Course Information

*   **Course Name: AF3005 – Programming for Finance
*   **Instructor Name: Dr. Usama Arshad

## App Overview 📚

The Financial ML Application guides users through the essential stages of a data science project applied to finance:

1.  **Data Loading:** Users can either upload their own financial data in CSV format or fetch historical stock data directly from Yahoo Finance for a specified ticker symbol and time period.
2.  **Preprocessing:** Includes crucial steps to clean and prepare the data:
    *   Handling missing values (dropping rows or imputing with mean/median/mode).
    *   Detecting and treating outliers using the IQR method (removing or capping).
    *   Applying feature scaling (StandardScaler or MinMaxScaler) to normalize numeric features, which is essential for many ML algorithms.
3.  **Feature Engineering & Selection:** Allows users to:
    *   Select the specific features to be used for model training.
    *   Choose the target variable for supervised learning tasks (Linear/Logistic Regression).
    *   Optionally create new features from existing data, such as:
        *   Date-based features (Year, Month, Day, Day of Week, Quarter).
        *   Common technical indicators (Moving Averages, RSI, Bollinger Bands, MACD) if OHLC data is available.
        *   Interaction features between numeric columns.
4.  **Train/Test Split:** Divides the data into training and testing sets to ensure unbiased model evaluation for supervised models. Includes optional stratification for classification tasks.
5.  **Model Training:** Trains one of the selected machine learning models on the prepared training data:
    *   **Linear Regression:** For predicting continuous numerical values.
    *   **Logistic Regression:** For binary or multi-class classification problems.
    *   **K-Means Clustering:** For unsupervised grouping of data points based on similarity.
6.  **Model Evaluation:** Assesses the trained model's performance using appropriate metrics:
    *   **Regression:** R² Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE).
    *   **Classification:** Accuracy, Confusion Matrix, Classification Report (Precision, Recall, F1-Score), ROC Curve & AUC, Precision-Recall Curve.
    *   **Clustering:** Inertia (Within-Cluster Sum of Squares), Silhouette Score, Elbow Method plot.
7.  **Results Visualization:** Presents the evaluation results and model insights using interactive Plotly charts:
    *   Actual vs. Predicted plots, Residual plots (Regression).
    *   Confusion Matrix heatmaps, ROC curves, PR curves (Classification).
    *   PCA-based cluster plots, Parallel Coordinates plots, Feature Distribution box plots (Clustering).
    *   Feature importance/coefficient plots.

The application aims to provide a practical learning tool for understanding and applying machine learning techniques in a financial context.

## Key Features ✨

*   ✅ **Multiple Data Sources:** Load data via CSV upload or Yahoo Finance API.
*   🧹 **Comprehensive Preprocessing:** Missing value handling, outlier treatment, feature scaling.
*   🛠️ **Flexible Feature Engineering:** Select features, create date components, technical indicators, and interactions.
*   🧠 **Choice of Models:** Linear Regression, Logistic Regression, and K-Means Clustering.
*   📊 **Detailed Evaluation:** Relevant metrics and visualizations for each model type.
*   📈 **Interactive Visualizations:** Utilizes Plotly for dynamic charts aiding interpretation.
*   🔁 **Step-by-Step Workflow:** Guides the user logically through the ML pipeline.
*   ⚙️ **State Management:** Uses Streamlit's session state to maintain progress.
*   🔄 **Reset Functionality:** Easily start a new analysis from scratch.

## Technologies Used 💻

*   **Python:** Core programming language.
*   **Streamlit:** Web application framework.
*   **Pandas:** Data manipulation and analysis.
*   **Numpy:** Numerical computing.
*   **Scikit-learn:** Machine learning library (models, preprocessing, metrics, etc.).
*   **Plotly:** Interactive data visualization.
*   **yfinance:** Fetching financial data from Yahoo Finance.

## Deployment Link 🌐

You can access the live application here:

➡️ **[https://moheedml.streamlit.app]** ⬅️

*(Example: `https://your-app-name.streamlit.app/`)*

## Demo Video 🎬

Watch a walkthrough of the application's features and usage:

➡️ **[https://www.linkedin.com/posts/moheed-ul-hassan-14a45a257_python-streamlit-machinelearning-activity-7324457410553610241-7TdI?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD8_uUoBTuETGHQvBUjiw6IRlr1WgnkgSiQ]** ⬅️

*(Consider embedding the video if your platform supports it, or provide a clear link).*

## Setup and Running Instructions 🚀

To run this application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
    *(Replace with your actual repository link)*

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment (Linux/macOS)
    source venv/bin/activate
    # Activate the environment (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Make sure you have a `requirements.txt` file (like the one generated previously) in the repository root.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run ml.py
    ```
    *(Replace `ml.py` with the actual name of your Python script if different)*

5.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Acknowledgements 🙏

Special thanks to Dr. Usama Arshad for the guidance provided during the AF3005 – Programming for Finance course.
