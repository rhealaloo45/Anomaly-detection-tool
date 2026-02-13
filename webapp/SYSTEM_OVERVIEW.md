# Anomaly Detection System Overview

This document provides an in-depth technical summary of the Log Anomaly Detection application, detailing its architecture, data flow, and underlying machine learning methodologies.

## 1. High-Level Architecture

The application is a **Flask-based web interface** that serves as a frontend for a sophisticated pipeline involving two unsupervised machine learning models and a Generative AI agent for explanation.

### Core Workflow:
1.  **Ingestion**: User uploads a server log file (CSV or raw text).
2.  **Preprocessing**: Data is parsed, cleaned, and transformed into numeric features.
3.  **Detection**: Two independent models (Autoencoder & Isolation Forest) scan the data for anomalies.
4.  **Explainability**: SHAP (SHapley Additive exPlanations) identifies *why* a data point is anomalous.
5.  **Interpretation**: OpenAI's GPT-3.5 analyzes the log context and SHAP features to describe the attack.
6.  **Presentation**: Results are displayed on an interactive dashboard and exported as PDF reports.

---

## 2. Component Analysis

### A. Data Ingestion & Parsing (`model_utils.py`)
The system supports two formats:
*   **Structured CSV**: Automatically maps columns like `status_code`, `bytes`, `response_time`, and `url` to a standard schema.
*   **Raw Syslogs**: Uses Regex pattern matching to extract fields from standard Apache/Nginx combined log formats.
*   *Robustness*: If parsing fails, it falls back to empty frames rather than crashing.

### B. Feature Engineering
Before feeding data to models, raw logs are transformed:
*   **Numerical Features**: `bytes` and `response_time` undergo **Log Transformation** (`np.log1p`) to handle skewed distributions (e.g., massive file transfers vs. small packets).
*   **Categorical Features**: `HTTP Method` (GET, POST, etc.) is **One-Hot Encoded** (for Autoencoder) or **Label Encoded** (for Isolation Forest).
*   **Scaling**: `MinMaxScaler` normalizes all features to a [0, 1] range to ensure model stability.

### C. Detection Models

#### 1. Autoencoder (Deep Learning)
*   **Architecture**: A neural network that learns to compress input data into a lower-dimensional latent space and then reconstruct it.
*   **Logic**: The model is trained on *normal* traffic. When it encounters an anomaly (e.g., a massive SQL injection payload), it fails to reconstruct the input accurately.
*   **Scoring**: We calculate the **Mean Squared Error (MSE)** between the input and the reconstruction.
*   **Thresholding**: An anomaly is flagged if the MSE exceeds the 95th percentile of the training error distribution.

#### 2. Isolation Forest (Ensemble Learning)
*   **Logic**: An algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value.
*   **Intuition**: Anomalies are few and different; they are "easier" to isolate (require fewer splits) than normal points.
*   **Scoring**: Returns an anomaly score (lower is more anomalous). We flag the bottom outlier percentile.

### D. Explainability & AI Integration

#### 1. SHAP (Feature Importance)
*   For every detected anomaly, we calculate **SHAP values**.
*   **Autoencoder**: Uses `KernelExplainer` to perturb inputs and measure impact on reconstruction error.
*   **Isolation Forest**: Uses `TreeExplainer` to trace the isolation path.
*   *Outcome*: This tells us precisely which columns (e.g., "High Response Time" or "Unusual HTTP Method") contributed most to the anomaly score.

#### 2. Generative AI (OpenAI GPT)
*   The system constructs a prompt containing:
    *   The raw log entry.
    *   The top 3 anomalous features identified by SHAP.
*   **The LLM Returns**:
    *   **Attack Type**: (e.g., "XSS Attempt", "DoS").
    *   **Severity**: (Low/Medium/High/Critical).
    *   **Explanation**: A human-readable summary.
    *   **LLM Prediction**: A deeper technical prediction.

### E. Reporting & Consistency
To ensure report consistency between the web view and the PDF:
*   **Caching**: Results from the initial analysis are serialized to a JSON file (`filename.csv.json`).
*   **PDF Generation (`report_generator.py`)**: Reads from this cache to generate the PDF, ensuring that the random nature of LLM generation doesn't produce different text for the reported file.

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, Bootstrap 5 (Dark Theme), Jinja2 Templating |
| **Backend** | Python 3.x, Flask |
| **ML Engine** | TensorFlow (Keras), Scikit-Learn |
| **Analysis** | SHAP, NumPy, Pandas |
| **AI Layer** | OpenAI API (GPT-3.5-Turbo) |
| **Visualization** | Chart.js (Feature Graphs) |

## 4. Performance Optimization
*   **Caching**: Prevents re-running expensive ML inference on report download.
*   **Vectorization**: Uses Pandas and NumPy for bulk data processing.
*   **Error Handling**: SHAP calculations are wrapped in try-catch blocks to prevent a single mathematical error from crashing the entire analysis pipeline.
