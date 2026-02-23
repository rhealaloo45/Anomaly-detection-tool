# Anomaly Detection System Overview

This document provides an in-depth technical summary of the Log Anomaly Detection application, detailing its architecture, data flow, and underlying machine learning methodologies.

## 1. High-Level Architecture

The application is a **Flask-based web interface** that serves as a frontend for a sophisticated pipeline involving two unsupervised machine learning models and an AI agent for clustering and classification.

### Architecture Diagram:
```text
  [User Log Upload] (CSV / Raw Syslog)
         |
         v
  [Data Ingestion / Parsing] 
  (Heuristic Column Mapping / Dynamic Regex IP Extraction)
         |
         v
  [Feature Engineering] ---> (Log-transform, MinMaxScaler, Encoding)
         |
         +-----------------+
         |                 |
         v                 v
  [Autoencoder]    [Isolation Forest]
  (Reconstruction)   (Ensemble Path)
         |                 |
         v                 v
   [Anomaly Mask]   [Dynamic 5% Threshold Fallback]
         \                 /
          \               /
           v             v
      [Clustering (HDBSCAN)]
          |             | -> (PCA 2D Projection for UI Visualization)
                   |
                   v
       [Generative AI (OpenAI GPT)]
       (Severity, Attack Type, Explanation)
                   |
                   v
          [Dashboard & PDF Report]
```

### Core Workflow:
1.  **Ingestion & Parsing**: User uploads a server log file. The parser intelligently determines tabular delimiters or falls back to unstructured robust regex mapping (e.g., stripping bad Excel quotes and chaining multi-IPs).
2.  **Preprocessing**: Data is dynamically mapped, cleaned, and transformed into numeric features (safe missing value imputation).
3.  **Detection**: Two independent models (Autoencoder & Isolation Forest) scan the data for anomalies. Isolation Forest includes a dynamic threshold fallback.
4.  **Clustering**: The detected anomalies are grouped together algorithmically using HDBSCAN to isolate similar attack profiles. Simultaneously, PCA is used to project the data into a 2D space strictly for frontend visual scatter-plotting.
5.  **Interpretation**: OpenAI analyzes the representative samples from each cluster to determine Attack Type, Severity, and an Explanation.
6.  **Presentation & Caching**: Results are stored robustly (handling complex Numpy datatypes) into a JSON cache, then displayed on a highly interactive dashboard with drill-down accordion views designed to minimize visual clutter, and can be exported as PDF reports.

---

## 2. Component Analysis

### A. Data Ingestion & Parsing (`model_utils.py`)
The parser is built for extreme fault-tolerance:
*   **Structured Tabular (CSV/TSV)**: Algorithmically detects delimiters. Uses heuristic mapping to search for arbitrary header names (e.g., mapping `size` or `length` to `bytes`). Maps disjointed request structures back together.
*   **Unstructured Raw Syslogs**: If a file exhibits syslog formats, a dynamic regex engine bypasses the CSV reader. It dynamically strips rogue formatting quotes (common in Excel exports) and gracefully extracts any number of chaining proxy IP addresses to feed into the schema.
*   **Fail-Safe**: Bad tabular renders elegantly return detailed error interfaces to the user UI instead of crashing the server with Python 500 exceptions.

### B. Feature Engineering
Before feeding data to models, raw logs are transformed:
*   **Numerical Features**: `bytes` and `response_time` undergo **Log Transformation** (`np.log1p`) to handle skewed distributions (e.g., massive file transfers vs. small packets).
*   **Categorical Features**: `HTTP Method` (GET, POST, etc.) is **One-Hot Encoded** (for Autoencoder) or **Label Encoded** (for Isolation Forest).
*   **Scaling**: `MinMaxScaler` normalizes all features to a [0, 1] range to ensure model stability. Missing integer columns safely default to `200` (Status) or `0` (Bytes).

### C. Detection Models

#### 1. Autoencoder (Deep Learning)
*   **Architecture**: A neural network that learns to compress input data into a lower-dimensional latent space and then reconstruct it.
*   **Logic**: The model is trained on *normal* traffic. When it encounters an anomaly (e.g., a massive SQL injection payload), it fails to reconstruct the input accurately.
*   **Scoring**: We calculate the **Mean Squared Error (MSE)** between the input and the reconstruction. Anomaly is flagged if MSE > threshold.

#### 2. Isolation Forest (Ensemble Learning)
*   **Logic**: An algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value.
*   **Scoring**: Returns an anomaly score (lower is more anomalous).
*   **Dynamic Thresholding**: Contains a fallback mechanism. If the model detects zero anomalies against a dataset, it mathematically adjusts to flag the absolute bottom 5% of outliers, preventing blind spots.

### D. Explainability & AI Integration
#### Generative AI Classification (OpenAI GPT)
*   Instead of making the AI evaluate thousands of logs, anomalies are first grouped together using HDBSCAN.
*   **The LLM Prompt Details**: Because sending all cluster logs to the LLM would exceed token limits and cost heavily, the system distills the cluster into a structured profile containing:
    *   **Aggregated Analytics**: Mathematical mean for `response_time` and total `bytes`.
    *   **Categorical Modes**: The most frequent Source IP, Destination IP, and Target Port across the cluster.
    *   **Failure Rates**: The percentage of HTTP requests that failed (e.g., 401 or 403), vital for detecting brute-force activity.
    *   **Feature Importance (SHAP)**: The system averages the underlying SHAP values across the cluster to provide the LLM with the exact *Top 3 Features* the ML models used to trigger the anomaly.
    *   **Representative Logs**: Exactly 5 raw logs are extracted from the cluster to provide the AI with concrete, deterministic evidence of the attack pattern without overwhelming context limitations.
*   The internal prompting aggressively constrains the LLM to map its findings against a specific, expansive cybersecurity threat portfolio, explicitly evaluating for: **Cross-Site Scripting (XSS), Sensitive Information Disclosure, SQL Injections, Insecure Deserialization, Broken Authentication & Failures, SSTI, Path Traversals (LFI), OS Command Injection & Remote Code Execution (RCE), CSRF, Rate Limiting Anomalies / DoS, IDOR, Clickjacking, Insecure Input Validation, Open Redirects, Cache Deception & Poisoning, SSRF, Hardcoded Credentials, and Recon/Scanner Indicators**.
*   **The LLM Returns**:
    *   **Attack Type**: One of the categorized threats from the explicit list above.
    *   **Severity**: (Low/Medium/High/Critical).
    *   **Explanation**: A human-readable summary.

### E. Presentation, Reporting & Consistency
To ensure report consistency between the web view and the PDF, and provide a clean user experience:
*   **Interactive UI**: The frontend dashboard utilizes dynamic inline cluster overviews, featuring expandable accordion components that allow users to cleanly drill down into cluster-specific anomalies and view raw log subsets without cluttering the master data tables.
*   **Caching**: Results from the initial analysis are serialized utilizing a custom `NumpyEncoder` capable of safely dumping `np.bool_` and `np.float32` structures to a JSON file (`filename.json`).
*   **PDF Generation (`report_generator.py`)**: Reads from this cache to generate the repeatable PDF.

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, Bootstrap 5 (Dark Theme), Jinja2 Templating |
| **Backend** | Python 3.x, Flask |
| **ML Engine** | TensorFlow (Keras), Scikit-Learn |
| **Analysis** | PCA, HDBSCAN Clustering, NumPy, Pandas |
| **AI Layer** | OpenAI API (GPT-3.5-Turbo) |

## 4. Performance Optimization
*   **Robust Parsers**: Regex extractors skip missing data padding rather than panicking.
*   **Caching**: Prevents re-running expensive ML inference on report download.
*   **Vectorization**: Uses Pandas and NumPy for bulk data processing.
