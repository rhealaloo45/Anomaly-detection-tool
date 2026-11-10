# Log anomaly detection: Autoencoder + Isolation Forest

This repo finds unusual requests in web-server logs and explains them. Two unsupervised models, a dense autoencoder and an Isolation Forest, flag anomalous log lines, and SHAP shows which features drove each flag. HDBSCAN groups the flagged lines into clusters, and an LLM labels each cluster with a likely attack type, a severity and an explanation. A Flask dashboard shows the results and exports PDF reports.

It contains the model notebooks and the first web app. The follow-on is **[anomoly-pipeline](https://github.com/rhealaloo45/anomoly-pipeline)**, which adds a regex signature engine, a four-model voting ensemble and a JSON API.

## What's here

| Path | What it is |
|---|---|
| [`autoencoder.ipynb`](autoencoder.ipynb) | Parse logs, build features, train the autoencoder, set the reconstruction-error threshold |
| [`isolation.ipynb`](isolation.ipynb) | The same data through an Isolation Forest, with SHAP explanations |
| [`test_ensemble.ipynb`](test_ensemble.ipynb), `ensemble.py` | An experimental weighted ensemble of the two models |
| [`webapp/`](webapp/) | Flask dashboard: upload logs, see clusters, download PDF reports ([README](webapp/README.md), [system overview](webapp/SYSTEM_OVERVIEW.md)) |
| `synthetic_logs_1k.csv`, `synthetic_logs_10k.csv` | Synthetic syslog-wrapped Apache access logs for training and trying the app |

## How it works

```
log file ─► parse ─► features ─┬─► Autoencoder (reconstruction error) ─┐
                               └─► Isolation Forest (anomaly score) ───┤
                                                                       ▼
            dashboard + PDF ◄── LLM labels each cluster ◄── HDBSCAN clusters of flagged lines
                                (attack type, severity,      (+ SHAP top features,
                                 explanation)                 PCA 2-D plot)
```

1. **Parse.** A regex reads Apache access lines wrapped in a syslog envelope (`<150>… httpd[pid]: … "GET /path HTTP/1.1" 200 …`). CSV exports are mapped onto the same columns.
2. **Features.** HTTP status, response size, response time and HTTP method. Size and time are log-transformed (`log1p`) and min-max scaled. The method is one-hot encoded for the autoencoder and label-encoded for the Isolation Forest.
3. **Autoencoder** (Keras, `webapp/train_models.py`). Dense layers 16 → 8 → 4 → 8 → 16, trained for 50 epochs on reconstruction (MSE). A line is anomalous when its reconstruction error exceeds the training mean + 2 standard deviations.
4. **Isolation Forest** (scikit-learn). 100 trees with `contamination=0.05`. If a file produces no flags, the app falls back to flagging the lowest 5% of scores.
5. **Explain.** SHAP `KernelExplainer` (autoencoder, 50 background samples) and `TreeExplainer` (Isolation Forest) give the top features behind each flag.
6. **Cluster.** HDBSCAN groups the flagged lines. PCA projects them to 2-D for the dashboard's scatter plot.
7. **Classify.** Each cluster of 5 or more lines is summarised before the LLM sees it: size and time span, the most common source and destination IPs, mean response time and bytes, the failed-login rate, the top 3 SHAP features and 3 sample lines. GPT-4o-mini (through Azure OpenAI) returns JSON with an attack type, a confidence, a severity and its reasoning. The prompt asks it to consider a list of common web attacks: SQL injection, XSS, path traversal, SSRF, broken authentication, DoS, reconnaissance and others. Answers are cached by summary hash, so re-running a file costs no extra calls.
8. **Report.** The dashboard shows summary stats, clusters and per-line details. Results are cached as JSON so the PDF report (FPDF) matches the screen. `/alerts` and `/clusters` return the latest results as JSON.

## Results

The logs are unlabelled, so the notebooks report how many lines each model flags, not precision or recall.

| Model | Data | Lines flagged |
|---|---|---|
| Autoencoder (mean + 2σ threshold) | `synthetic_logs_10k.csv`, 10,000 lines | 299 (about 3%) |
| Isolation Forest (`contamination=0.05`) | `synthetic_logs_1k.csv`, 1,000 lines | 49 |

The Isolation Forest's flag rate is set by its contamination parameter (about 5%). The autoencoder's rate follows from its error threshold.

## Stack

Python, Flask, TensorFlow/Keras, scikit-learn, SHAP, HDBSCAN, pandas, NumPy, OpenAI SDK (Azure OpenAI, GPT-4o-mini), FPDF, Bootstrap, Chart.js.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # or put OPENAI_API_KEY = "..." in keys.py at the repo root (git-ignored)
python webapp/app.py
```

Open http://localhost:5000 and upload `synthetic_logs_1k.csv`.

- The LLM client in `webapp/llm_classifier.py` points at an Azure OpenAI endpoint. Set `endpoint` to your own resource, or drop `base_url` to call OpenAI directly.
- Without a key, detection, SHAP and clustering still run, and clusters are labelled "Unclassified Cluster".
- Pre-trained models are in `webapp/models/`. To retrain them on `synthetic_logs_10k.csv`, run `python webapp/train_models.py`.

This tool is for defensive use on logs you are authorised to analyse.
