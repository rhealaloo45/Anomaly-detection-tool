import pandas as pd
import numpy as np
import re
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
import shap
import openai
import json
from clustering_engine import cluster_anomalies
from cluster_summarizer import summarize_cluster
from llm_classifier import classify_cluster
from sklearn.decomposition import PCA

# Constants
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'synthetic_logs_10k.csv')
LLM_ENABLED = True
CLUSTER_MIN_SIZE = 5

# Regex Pattern (Same as train_models.py)
LOG_PATTERN = re.compile(r'<150>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\d+)\s+(\S+)\s+(\d+)\s+"(.*?)"\s+"(.*?)"')

# Load OpenAI Key
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from keys import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def parse_logs(file_path):
    # Determine if it's likely a raw syslog file based on the first line
    is_raw_log = False
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            # If exported from Excel wrongly, it might be surrounded by quotes entirely:
            clean_first = first_line[1:-1].replace('""', '"') if first_line.startswith('"') and first_line.endswith('"') else first_line
            # Common raw log indicators: starts with syslog facility <150>, or date like Jan 28, or IP address
            if clean_first.startswith('<') or re.match(r'^[A-Z][a-z]{2}\s+\d+', clean_first) or re.match(r'^\d{1,3}\.\d{1,3}\.', clean_first):
                is_raw_log = True
    except Exception:
        pass

    # 1. Try reading as tabular data (CSV/TSV)
    if not is_raw_log:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                
            delimiter = ','
            if ';' in first_line and first_line.count(';') > first_line.count(','):
                delimiter = ';'
            elif '\t' in first_line and first_line.count('\t') > first_line.count(','):
                delimiter = '\t'

            df = pd.read_csv(file_path, sep=delimiter, on_bad_lines='skip')
            
            # We only consider it a valid tabular dataset if it parses into multiple sensible columns
            if len(df.columns) > 1 and not first_line.strip().startswith('<'):
                df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
                
                # --- Dynamic Column Mapping ---
                if 'status' not in df.columns:
                    for col in df.columns:
                        if 'status' in col or 'code' in col:
                            df['status'] = df[col]
                            break
                    if 'status' not in df.columns: df['status'] = 200
                    
                if 'bytes' not in df.columns:
                    for col in df.columns:
                        if 'byte' in col or 'size' in col or 'length' in col:
                            df['bytes'] = df[col]
                            break
                    if 'bytes' not in df.columns: df['bytes'] = 0
                    
                if 'response_time' not in df.columns:
                    for col in df.columns:
                        if 'time' in col or 'duration' in col or 'latency' in col or 'elapsed' in col:
                            df['response_time'] = df[col]
                            break
                    if 'response_time' not in df.columns: df['response_time'] = 0
                    
                if 'request' not in df.columns:
                    method_col = None
                    path_col = None
                    for col in df.columns:
                        if col in ['method', 'cs_method', 'request_method', 'action']:
                            method_col = col
                        if col in ['url', 'uri', 'path', 'endpoint', 'cs_uri_stem', 'cs_uri']:
                            path_col = col
                    
                    if method_col and path_col:
                        df['request'] = df[method_col].astype(str) + " " + df[path_col].astype(str) + " HTTP/1.1"
                    elif method_col:
                        df['request'] = df[method_col].astype(str) + " /UNKNOWN HTTP/1.1"
                    elif path_col:
                        df['request'] = "GET " + df[path_col].astype(str) + " HTTP/1.1"
                    else:
                        df['request'] = "UNKNOWN /UNKNOWN HTTP/1.1"

                # Fill NA and clean types safely
                df['status'] = pd.to_numeric(df['status'], errors='coerce').fillna(200).astype(int)
                df['bytes'] = pd.to_numeric(df['bytes'].astype(str).str.replace('-', '0'), errors='coerce').fillna(0).astype(int)
                df['response_time'] = pd.to_numeric(df['response_time'].astype(str).str.replace('-', '0'), errors='coerce').fillna(0).astype(int)
                df['request'] = df['request'].astype(str).fillna("UNKNOWN / UNKNOWN HTTP/1.1")
                
                print("Successfully parsed as tabular data with dynamic column mappings.")
                if len(df) > 0:
                    return df
                
        except Exception as e:
            print(f"Tabular parsing failed, falling back to regex: {e}")

    # 2. Fallback: Parse using Syslog Regex (for unstructured raw log files)
    data = []
    
    # A highly flexible fallback regex mapping an arbitrary unformatted syslog header into key fields.
    # Group extraction targets: 1:syslog_ts, 2:host, 3:process, 4:ip_block, 5:apache_ts, 6:request, 7:status, 8:bytes, 9:response_time, 10:referer, 11:user_agent
    DYNAMIC_LOG_PATTERN = re.compile(
        r'^<150>([A-Za-z]{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*?):\s+(.*?)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\d+)\s+(\S+)\s+(\S+)\s+\d*\s*"(.*?)"\s+"(.*?)"'
    )
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_str = line.strip()
                # Dynamically unfold strings improperly re-escaped into Excel CSV single columns
                if line_str.startswith('"') and line_str.endswith('"'):
                    line_str = line_str[1:-1].replace('""', '"')
                    
                # 1. Try rigid legacy pattern (which maps out specific properties in order)
                match = LOG_PATTERN.match(line_str)
                if match:
                    data.append(match.groups())
                else:
                    # 2. Try the flexible dynamic fallback pattern (which coalesces variadic IP lists and unpredictable variables together safely)
                    match2 = DYNAMIC_LOG_PATTERN.match(line_str)
                    if match2:
                        g = match2.groups()
                        # Extract the dynamic match list: (syslog_ts, host, process, ip_block, apache_ts, request, status, bytes, response_time, referer, user_agent)
                        
                        ip_block = g[3] 
                        # just split the first two IPs gracefully
                        ips = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ip_block)
                        ip1 = ips[0] if len(ips) > 0 else "0.0.0.0"
                        ip2 = ips[1] if len(ips) > 1 else ip1
                        
                        row = (
                            g[0], g[1], g[2], ip1, ip2, "0", "UNKNOWN", g[4], g[5], g[6], g[7], g[8], g[9], g[10]
                        )
                        data.append(row)
        
        if data:
            columns = [
                "syslog_ts", "host", "process", "ip1", "ip2", "session_id", "domain", 
                "apache_ts", "request", "status", "bytes", "response_time", "referer", "user_agent"
            ]
            df = pd.DataFrame(data, columns=columns)
            
            # The model requires status, response_time, and bytes as integers during inference
            df['status'] = pd.to_numeric(df['status'], errors='coerce').fillna(200).astype(int)
            df['bytes'] = pd.to_numeric(df['bytes'].replace('-', '0'), errors='coerce').fillna(0).astype(int)
            df['response_time'] = pd.to_numeric(df['response_time'].replace('-', '0'), errors='coerce').fillna(0).astype(int)
            df['request'] = df['request'].astype(str)
            
            return df
    except Exception as e:
        print(f"Regex parsing failed: {e}")
    
    # 3. Last Resort: Return empty
    return pd.DataFrame({"error": [f"Could not parse file: {file_path}"]})

class ModelHandler:
    def __init__(self):
        self.ae_model = None
        self.ae_scaler = None
        self.ae_threshold = None
        self.ae_columns = None
        self.iso_forest = None
        self.iso_encoder = None
        self.ae_explainer = None
        self.iso_explainer = None
        self.ae_background = None
        
        self.load_models()
        self.init_explainers()

    def load_models(self):
        # Autoencoder
        self.ae_model = load_model(os.path.join(MODELS_DIR, 'autoencoder.h5'), compile=False)
        self.ae_scaler = joblib.load(os.path.join(MODELS_DIR, 'ae_scaler.pkl'))
        self.ae_threshold = joblib.load(os.path.join(MODELS_DIR, 'ae_threshold.pkl'))
        self.ae_columns = joblib.load(os.path.join(MODELS_DIR, 'ae_columns.pkl'))
        
        # Isolation Forest
        self.iso_forest = joblib.load(os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
        self.iso_encoder = joblib.load(os.path.join(MODELS_DIR, 'iso_method_encoder.pkl'))

    def init_explainers(self):
        # We need some background data for AE KernelExplainer
        # Load a small sample from synthetic logs
        if os.path.exists(LOG_FILE_PATH):
            try:
                df = parse_logs(LOG_FILE_PATH)
                # Process for AE
                df_ae = self.preprocess_ae(df)
                
                if not df_ae.empty:
                    self.ae_background = shap.sample(df_ae, 50)
                    
                    def predict_error(X):
                        # Ensure X is numpy array
                        if isinstance(X, pd.DataFrame):
                            X = X.values
                        
                        # Predict
                        rec = self.ae_model.predict(X, verbose=0)
                        
                        # Calculate MSE per sample
                        errors = np.mean(np.square(X - rec), axis=1)
                        return np.array(errors).flatten()

                    self.ae_explainer = shap.KernelExplainer(predict_error, self.ae_background)
                else:
                    print("Warning: Background data for AE explainer is empty.")
                    self.ae_explainer = None
            except Exception as e:
                print(f"Failed to init AE explainer: {e}")
                self.ae_explainer = None
        
        # Iso Forest Explainer
        try:
            self.iso_explainer = shap.TreeExplainer(self.iso_forest)
        except Exception as e:
            print(f"Failed to init Iso Forest explainer: {e}")
            self.iso_explainer = None

    def preprocess_ae(self, df):
        df_new = df.copy()
        df_new['Status'] = pd.to_numeric(df_new['status'], errors='coerce').fillna(0)
        df_new['Size'] = pd.to_numeric(df_new['bytes'].replace('-', '0'), errors='coerce').fillna(0)
        df_new['Duration'] = pd.to_numeric(df_new['response_time'].replace('-', '0'), errors='coerce').fillna(0)
        df_new['Method'] = df_new['request'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'UNKNOWN')
        
        data_model = pd.get_dummies(df_new[['Status', 'Size', 'Duration', 'Method']], columns=['Method'])
        
        # Align columns
        for col in self.ae_columns:
            if col not in data_model.columns:
                data_model[col] = 0
        data_model = data_model[self.ae_columns]
        
        # Log transform
        data_model['Size'] = np.log1p(data_model['Size'])
        data_model['Duration'] = np.log1p(data_model['Duration'])
        
        data_model = data_model.astype('float32')
        
        # Scale
        numerical_cols = ['Status', 'Size', 'Duration']
        data_model[numerical_cols] = self.ae_scaler.transform(data_model[numerical_cols])
        
        return data_model

    def preprocess_iso(self, df):
        df_new = df.copy()
        df_new['bytes'] = df_new['bytes'].replace('-', '0').astype(int)
        df_new['response_time'] = df_new['response_time'].astype(int)
        df_new['status'] = df_new['status'].astype(int)
        df_new['http_method'] = df_new['request'].apply(lambda x: x.split()[0] if x else "UNKNOWN")
        
        # Safe encode
        known_methods = set(self.iso_encoder.classes_)
        df_new['http_method_encoded'] = df_new['http_method'].apply(
            lambda x: self.iso_encoder.transform([x])[0] if x in known_methods else -1
        )
        
        features = ['response_time', 'bytes', 'status', 'http_method_encoded']
        return df_new[features]

    def analyze(self, file_path):
        import time
        start_time = time.time()
        
        df = parse_logs(file_path)
        if df.empty:
            return {"error": "No data found or parsing failed."}
            
        if "error" in df.columns:
            return {"error": str(df["error"].iloc[0])}
            
        total_logs = len(df)
        
        # Autoencoder Analysis
        X_ae = self.preprocess_ae(df)
        reconstructions = self.ae_model.predict(X_ae, verbose=0)
        mse_vals = np.mean(np.square(X_ae - reconstructions), axis=1)
        ae_anomalies_mask = mse_vals > self.ae_threshold
        ae_anomalies = df[ae_anomalies_mask].copy()
        ae_anomalies['error'] = mse_vals[ae_anomalies_mask]
        
        # Isolation Forest Analysis
        X_iso = self.preprocess_iso(df)
        iso_preds = self.iso_forest.predict(X_iso)
        iso_scores_all = self.iso_forest.decision_function(X_iso)
        iso_anomalies_mask = iso_preds == -1
        
        # Fallback: if the trained model detects strictly 0 anomalies (perhaps due to 
        # training contamination mismatch), flag the lowest 5% anomaly scores dynamically.
        if not iso_anomalies_mask.any() and len(df) > 0:
            dynamic_thresh = np.percentile(iso_scores_all, 5)
            # Only apply if it actually isolates a small distinct minority (not the whole set)
            if dynamic_thresh < np.median(iso_scores_all):
                iso_anomalies_mask = iso_scores_all <= dynamic_thresh

        iso_anomalies = df[iso_anomalies_mask].copy()
        iso_anomalies['score'] = iso_scores_all[iso_anomalies_mask]
        
        results = {
            "autoencoder": {
                "count": int(np.sum(ae_anomalies_mask)),
                "anomalies": [],
                "all_anomalies": [],
                "clusters": [],
                "pca_data": []
            },
            "isolation_forest": {
                "count": int(np.sum(iso_anomalies_mask)),
                "anomalies": [],
                "all_anomalies": [],
                "clusters": [],
                "pca_data": []
            },
            "total_logs": total_logs
        }
        
        # Calculate Critical Thresholds (Top 10% severe)
        ae_crit_thresh = np.percentile(ae_anomalies['error'], 90) if not ae_anomalies.empty else float('inf')
        # For Iso Forest, lower score is more anomalous. So bottom 10%.
        iso_crit_thresh = np.percentile(iso_anomalies['score'], 10) if not iso_anomalies.empty else float('-inf')
        
        # --- Generate Initial Explanations for ALL Autoencoder Anomalies ---
        ae_diff = np.square(X_ae.values - reconstructions)
        ae_feature_names = X_ae.columns.tolist()
        
        for idx in ae_anomalies.index:
            row_idx = df.index.get_loc(idx)
            row_diff = ae_diff[row_idx]
            max_feat_idx = np.argmax(row_diff)
            max_feat = ae_feature_names[max_feat_idx]
            error_val = float(ae_anomalies.loc[idx, 'error'])
            
            explanation = f"Classified as anomaly due to high reconstruction error in '{max_feat}'."
            
            results["autoencoder"]["all_anomalies"].append({
                "index": int(idx),
                "log": df.loc[idx].to_dict(),
                "error": error_val,
                "explanation": explanation,
                "is_critical": error_val >= ae_crit_thresh
            })
            
        # --- Generate Initial Explanations for ALL Isolation Forest Anomalies ---
        if not iso_anomalies.empty:
            iso_anom_data = X_iso.loc[iso_anomalies.index]
            shap_values_iso_all = self.iso_explainer.shap_values(iso_anom_data)
            iso_feature_names = X_iso.columns.tolist()
            
            for i, idx in enumerate(iso_anomalies.index):
                shap_val = shap_values_iso_all[i]
                # Find feature with most negative SHAP value (pushing score towards -1)
                min_shap_idx = np.argmin(shap_val)
                top_feat = iso_feature_names[min_shap_idx]
                score_val = float(iso_anomalies.loc[idx, 'score'])
                
                explanation = f"Classified as anomaly primarily due to '{top_feat}' pattern."
                
                results["isolation_forest"]["all_anomalies"].append({
                    "index": int(idx),
                    "log": df.loc[idx].to_dict(),
                    "score": score_val,
                    "explanation": explanation,
                    "is_critical": score_val <= iso_crit_thresh
                })
        
        # --- Generate Cluster-Level AI Explanations ---

        # 1. AE Clusters
        if not ae_anomalies.empty:
            shap_dict_ae = {}
            for idx in ae_anomalies.index:
                instance = X_ae.iloc[[df.index.get_loc(idx)]]
                if self.ae_explainer:
                    try:
                        shap_vals = self.ae_explainer.shap_values(instance)
                        shap_dict_ae[idx] = shap_vals[0].flatten() if isinstance(shap_vals, list) else shap_vals.flatten()
                    except Exception:
                        shap_dict_ae[idx] = ae_diff[df.index.get_loc(idx)]
                else:
                    shap_dict_ae[idx] = ae_diff[df.index.get_loc(idx)]
            
            features_df_ae = X_ae.loc[ae_anomalies.index]
            cluster_data_ae = cluster_anomalies(features_df_ae, ae_anomalies)
            cluster_summaries_ae = summarize_cluster(cluster_data_ae, df, shap_dict_ae, X_ae.columns.tolist())
            
            # Compute PCA for visualization
            if len(features_df_ae) >= 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(features_df_ae.values)
                # Map back to array with cluster_id later
                pca_dict = {idx: pca_result[i] for i, idx in enumerate(ae_anomalies.index)}
            else:
                pca_dict = {idx: [0.0, 0.0] for idx in ae_anomalies.index}
            
            processed_indices_ae = set()
            for c_id, summary in cluster_summaries_ae.items():
                if LLM_ENABLED and summary['size'] >= CLUSTER_MIN_SIZE:
                    ai_result = classify_cluster(summary)
                else:
                    ai_result = {"attack_type": "Unclassified Cluster", "severity": "Medium", "reasoning": "Cluster too small or AI disabled.", "common_pattern": "Pattern needs manual review"}
                
                results["autoencoder"]["clusters"].append({
                    "cluster_id": c_id,
                    "attack_type": ai_result.get("attack_type", "Unknown"),
                    "severity": ai_result.get("severity", "Medium"),
                    "confidence": ai_result.get("confidence", "0.0"),
                    "reasoning": ai_result.get("reasoning", ""),
                    "common_pattern": ai_result.get("common_pattern", ""),
                    "size": summary["size"],
                    "log_indices": summary["log_indices"]
                })
                
                for idx in summary["log_indices"]:
                    processed_indices_ae.add(idx)
                    pattern = ai_result.get("attack_type", "Unknown")
                    top_3 = ", ".join(summary["top_3_shap_features"])
                    per_log_exp = f"This log is part of an anomaly cluster '{pattern}'. Contributing features: {top_3}."
                    
                    row = df.loc[idx]
                    graph_data = self.get_top_features(shap_dict_ae[idx], X_ae.columns.tolist())
                    
                    for item in results["autoencoder"]["all_anomalies"]:
                        if item["index"] == idx:
                            item["explanation"] = per_log_exp
                            item["attack_type"] = pattern
                            item["severity"] = ai_result.get("severity", "Medium")
                            item["llm_prediction"] = ai_result.get("reasoning", "")
                            break
                            
                    results["autoencoder"]["anomalies"].append({
                        "index": int(idx),
                        "log": row.to_dict(),
                        "error": float(ae_anomalies.loc[idx, 'error']),
                        "explanation": per_log_exp,
                        "attack_type": pattern,
                        "severity": ai_result.get("severity", "Medium"),
                        "llm_prediction": ai_result.get("reasoning", ""),
                        "cluster_id": c_id,
                        "graph_data": graph_data
                    })
                    
            # Handle noise
            for idx in ae_anomalies.index:
                if idx not in processed_indices_ae:
                    row = df.loc[idx]
                    graph_data = self.get_top_features(shap_dict_ae[idx], X_ae.columns.tolist())
                    results["autoencoder"]["anomalies"].append({
                        "index": int(idx),
                        "log": row.to_dict(),
                        "error": float(ae_anomalies.loc[idx, 'error']),
                        "explanation": "Isolated anomaly (noise). Did not cluster with others.",
                        "attack_type": "Isolated Anomaly",
                        "severity": "Low",
                        "llm_prediction": "N/A",
                        "cluster_id": "None",
                        "graph_data": graph_data
                    })
                    
            # Populate pca_data for frontend
            for idx in ae_anomalies.index:
                cluster_id = "None"
                attack_type = "Isolated Anomaly"
                for item in results["autoencoder"]["anomalies"]:
                    if item["index"] == idx:
                        cluster_id = item.get("cluster_id", "None")
                        attack_type = item.get("attack_type", "Isolated Anomaly")
                        break
                results["autoencoder"]["pca_data"].append({
                    "x": float(pca_dict[idx][0]),
                    "y": float(pca_dict[idx][1]),
                    "cluster_id": cluster_id,
                    "attack_type": attack_type,
                    "index": int(idx)
                })

        # 2. Iso Forest Clusters
        if not iso_anomalies.empty:
            shap_dict_iso = {idx: shap_values_iso_all[i] for i, idx in enumerate(iso_anomalies.index)}
            
            features_df_iso = X_iso.loc[iso_anomalies.index]
            cluster_data_iso = cluster_anomalies(features_df_iso, iso_anomalies)
            cluster_summaries_iso = summarize_cluster(cluster_data_iso, df, shap_dict_iso, X_iso.columns.tolist())
            
            if len(features_df_iso) >= 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(features_df_iso.values)
                pca_dict_iso = {idx: pca_result[i] for i, idx in enumerate(iso_anomalies.index)}
            else:
                pca_dict_iso = {idx: [0.0, 0.0] for idx in iso_anomalies.index}
            
            processed_indices_iso = set()
            for c_id, summary in cluster_summaries_iso.items():
                if LLM_ENABLED and summary['size'] >= CLUSTER_MIN_SIZE:
                    ai_result = classify_cluster(summary)
                else:
                    ai_result = {"attack_type": "Unclassified Cluster", "severity": "Medium", "reasoning": "Cluster too small or AI disabled.", "common_pattern": "Pattern needs manual review"}
                
                results["isolation_forest"]["clusters"].append({
                    "cluster_id": c_id,
                    "attack_type": ai_result.get("attack_type", "Unknown"),
                    "severity": ai_result.get("severity", "Medium"),
                    "confidence": ai_result.get("confidence", "0.0"),
                    "reasoning": ai_result.get("reasoning", ""),
                    "common_pattern": ai_result.get("common_pattern", ""),
                    "size": summary["size"],
                    "log_indices": summary["log_indices"]
                })
                
                for idx in summary["log_indices"]:
                    processed_indices_iso.add(idx)
                    pattern = ai_result.get("attack_type", "Unknown")
                    top_3 = ", ".join(summary["top_3_shap_features"])
                    per_log_exp = f"This log is part of an anomaly cluster '{pattern}'. Contributing features: {top_3}."
                    
                    row = df.loc[idx]
                    graph_data = self.get_top_features(shap_dict_iso[idx], X_iso.columns.tolist())
                    
                    for item in results["isolation_forest"]["all_anomalies"]:
                        if item["index"] == idx:
                            item["explanation"] = per_log_exp
                            item["attack_type"] = pattern
                            item["severity"] = ai_result.get("severity", "Medium")
                            item["llm_prediction"] = ai_result.get("reasoning", "")
                            break
                            
                    results["isolation_forest"]["anomalies"].append({
                        "index": int(idx),
                        "log": row.to_dict(),
                        "explanation": per_log_exp,
                        "attack_type": pattern,
                        "severity": ai_result.get("severity", "Medium"),
                        "llm_prediction": ai_result.get("reasoning", ""),
                        "cluster_id": c_id,
                        "graph_data": graph_data
                    })
                    
            for idx in iso_anomalies.index:
                if idx not in processed_indices_iso:
                    row = df.loc[idx]
                    graph_data = self.get_top_features(shap_dict_iso[idx], X_iso.columns.tolist())
                    results["isolation_forest"]["anomalies"].append({
                        "index": int(idx),
                        "log": row.to_dict(),
                        "explanation": "Isolated anomaly (noise). Did not cluster with others.",
                        "attack_type": "Isolated Anomaly",
                        "severity": "Low",
                        "llm_prediction": "N/A",
                        "cluster_id": "None",
                        "graph_data": graph_data
                    })
                    for item in results["isolation_forest"]["all_anomalies"]:
                        if item["index"] == idx:
                            item["explanation"] = "Isolated anomaly (noise). Did not cluster with others."
                            item["attack_type"] = "Isolated Anomaly"
                            item["severity"] = "Low"
                            item["llm_prediction"] = "N/A"
                            break
                            
            # Populate pca_data for frontend
            for idx in iso_anomalies.index:
                cluster_id = "None"
                attack_type = "Isolated Anomaly"
                for item in results["isolation_forest"]["anomalies"]:
                    if item["index"] == idx:
                        cluster_id = item.get("cluster_id", "None")
                        attack_type = item.get("attack_type", "Isolated Anomaly")
                        break
                results["isolation_forest"]["pca_data"].append({
                    "x": float(pca_dict_iso[idx][0]),
                    "y": float(pca_dict_iso[idx][1]),
                    "cluster_id": cluster_id,
                    "attack_type": attack_type,
                    "index": int(idx)
                })
                
        end_time = time.time()
        results["duration"] = round(end_time - start_time, 2)
        return results

    def get_top_features(self, shap_values, feature_names, top_n=5):
        pairs = zip(feature_names, shap_values)
        sorted_pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
        top_pairs = sorted_pairs[:top_n]
        return {
            "labels": [p[0] for p in top_pairs],
            "values": [float(p[1]) for p in top_pairs]
        }

    def get_ai_analysis_ae(self, log_entry, shap_values, feature_names):
        if not OPENAI_API_KEY:
            return self.get_dummy_analysis("OpenAI API Key missing.")
        
        # Summarize top 3 features contributing to error
        feat_importance = sorted(zip(feature_names, shap_values), key=lambda x: -abs(x[1]))
        top_features = feat_importance[:3]
        shap_text = ", ".join([f"{f}: {v:.4f}" for f, v in top_features])
        
        prompt = f"""
        You are a cybersecurity expert. Analyze this server log anomaly (Autoencoder).
        
        Log Details: {log_entry}
        
        Key Anomalous Features: {shap_text}
        
        Provide a JSON response with the following keys:
        - "attack_type": Short name of the attack/issue (e.g., "SQL Injection", "Brute Force", "Data Exfiltration", "Suspicious User Agent", or "Configuration Issue").
        - "severity": One of ["Critical", "High", "Medium", "Low"].
        - "llm_prediction": A detailed prediction of what kind of attack has taken place based purely on the log analysis.
        - "explanation": A clear 1-2 sentence explanation of why this was flagged, referencing the specific features or log content logic.
        
        Return ONLY valid JSON.
        """
        return self.call_openai_json(prompt)

    def get_ai_analysis_iso(self, log_entry, shap_values, feature_names):
        if not OPENAI_API_KEY:
            return self.get_dummy_analysis("OpenAI API Key missing.")
            
        feat_importance = sorted(zip(feature_names, shap_values), key=lambda x: -abs(x[1]))
        top_features = feat_importance[:3]
        shap_text = ", ".join([f"{f}: {v:.4f}" for f, v in top_features])
        
        prompt = f"""
        You are a cybersecurity expert. Analyze this server log anomaly (Isolation Forest).
        
        Log Details: {log_entry}
        Feature Analysis: {shap_text}
        
        Provide a JSON response with the following keys:
        - "attack_type": Short name of the attack/issue (e.g., "SQL Injection", "Brute Force", "Data Exfiltration", "Suspicious User Agent", or "Configuration Issue").
        - "severity": One of ["Critical", "High", "Medium", "Low"].
        - "llm_prediction": A detailed prediction of what kind of attack has taken place based purely on the log analysis.
        - "explanation": A clear 1-2 sentence explanation of why this was flagged, referencing the specific features or log content logic.
        
        Return ONLY valid JSON.
        """
        return self.call_openai_json(prompt)

    def get_dummy_analysis(self, message):
        return {
            "attack_type": "Unknown",
            "severity": "Low",
            "llm_prediction": message,
            "explanation": message
        }

    def call_openai_json(self, prompt):
        import json
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            # Try parsing JSON
            try:
                # Handle cases where LLM might wrap in markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()
                
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback if valid JSON isn't returned
                return self.get_dummy_analysis(f"Failed to parse AI response: {content[:50]}...")
                
        except Exception as e:
            return self.get_dummy_analysis(f"Error: {str(e)}")

# Singleton instance
handler = ModelHandler()
