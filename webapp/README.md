# Log Anomaly Detection Tool

A powerful cybersecurity tool designed to detect, analyze, and explain anomalies in server logs using advanced machine learning models and Generative AI.

## 🚀 Overview

This web application allows security analysts to upload network/server logs (CSV or raw text) and automatically screen them for potential security threats. It leverages two distinct unsupervised learning models—**Autoencoder** and **Isolation Forest**—to identify deviations from normal traffic patterns. Detected anomalies are then analyzed by an LLM (OpenAI GPT) to provide human-readable explanations, attack type classification, and severity ratings.

## ✨ Key Features

*   **Dual-Model Detection**:
    *   **Autoencoder**: Deep learning model that flags anomalies based on high reconstruction error.
    *   **Isolation Forest**: Ensemble method that isolates anomalies based on feature splits.
*   **AI-Powered Analysis**:
    *   Integrates with OpenAI API to explain *why* a specific log entry is anomalous.
    *   Predicts **Attack Types** (e.g., SQL Injection, Brute Force, DoS).
    *   Assigns **Severity Levels** (Critical, High, Medium, Low).
*   **Interactive Cyber Dashboard**:
    *   Dark-mode UI tailored for security operations.
    *   Summary statistics and toggleable data tables.
    *   Detailed log viewer modal with feature importance graphs.
*   **Reporting**:
    *   Generates comprehensive **PDF Reports** summarizing all detected threats.

## 🛠️ Technology Stack

*   **Backend**: Python, Flask
*   **ML/Data**: TensorFlow (Keras), Scikit-Learn, Pandas, NumPy, SHAP
*   **AI**: OpenAI API (gpt-4o-mini)
*   **Frontend**: HTML5, Bootstrap 5, Chart.js, Vanilla CSS
*   **Reporting**: FPDF

## 📂 Project Structure

```
webapp/
├── app.py                 # Main Flask application entry point
├── model_utils.py         # Core logic for loading models, preprocessing, and AI analysis
├── report_generator.py    # Logic for generating PDF reports
├── templates/             # HTML templates (index.html, results.html)
├── static/                # CSS and other static assets
├── models/                # Directory containing pre-trained .h5 and .pkl model files
└── uploads/               # Temporary storage for uploaded logs and generated reports
```

## ⚙️ Setup & Installation

1.  **Prerequisites**:
    *   Python 3.8+
    *   An active OpenAI API Key

2.  **Install Dependencies**:
    Ensure you have the required packages installed. You can typically install them via pip:
    ```bash
    pip install flask pandas numpy tensorflow scikit-learn shap openai fpdf joblib
    ```

3.  **Configure API Key**:
    *   The application looks for an API key in a file named `keys.py` located in the parent directory of `webapp`, or strictly reads it from the environment depending on configuration.
    *   Ensure `OPENAI_API_KEY` is available to the application.

##  ▶️ Usage

1.  **Start the Application**:
    Navigate to the project root and run:
    ```bash
    python webapp/app.py
    ```

2.  **Access the Dashboard**:
    Open your web browser and go to `http://localhost:5000`.

3.  **Analyze Logs**:
    *   Upload a `.csv` log file (or a standard server log file).
    *   Wait for the analysis to complete.
    *   Review the **Threat Analysis Dashboard**.
    *   Download PDF reports for offline review.

## 📊 Data Format

The tool supports standard server log formats processed into CSVs with columns such as:
*   `request` (e.g., "GET /admin HTTP/1.1")
*   `status` (e.g., 200, 404, 500)
*   `bytes` (Response size)
*   `response_time` (Duration)

## ⚠️ Note

This tool is intended for educational and defensive security purposes. Always ensure you have authorization before analyzing logs from systems you do not own.
