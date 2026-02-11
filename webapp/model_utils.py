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

# Constants
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'synthetic_logs_10k.csv')
import json

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
    # 1. Try reading as a standard CSV (Header-based)
    try:
        df = pd.read_csv(file_path)
        # Normalize columns: lowercase, strip spaces, replace special chars
        df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
        
        # Comprehensive mapping for standard columns
        column_mapping = {
            # Status
            'code': 'status', 'status_code': 'status', 'sc_status': 'status', 's_status': 'status', 'return_code': 'status', 'response_code': 'status',
            # Bytes
            'size': 'bytes', 'content_length': 'bytes', 'length': 'bytes', 'sc_bytes': 'bytes', 'cs_bytes': 'bytes', 'bytes_sent': 'bytes', 'response_size': 'bytes',
            # Time
            'time': 'response_time', 'duration': 'response_time', 'time_taken': 'response_time', 'elapsed': 'response_time', 'latency': 'response_time',
            # Request/URL
            'url': 'request', 'uri': 'request', 'path': 'request', 'message': 'request', 'cs_uri_stem': 'request', 'cs_uri': 'request',
            # Method (Optional, if request is missing)
            'method': 'method', 'cs_method': 'method', 'request_method': 'method'
        }
        
        # Apply renaming
        df = df.rename(columns=column_mapping)
        
        # Handle 'method' column if 'request' is missing
        if 'request' not in df.columns and 'method' in df.columns:
            # Create a dummy request string from method so downstream logic works
            df['request'] = df['method'] + ' /UNKNOWN_PATH HTTP/1.1'
            
        # Ensure required columns exist, fill with defaults if missing
        if 'status' not in df.columns:
            df['status'] = 200 # Default to success
        if 'bytes' not in df.columns:
            df['bytes'] = 0
        if 'response_time' not in df.columns:
            df['response_time'] = 0
            
        # Check if we have the critical 'request' column now
        if 'request' in df.columns:
            # Basic cleaning
            df['status'] = pd.to_numeric(df['status'], errors='coerce').fillna(0).astype(int)
            df['bytes'] = pd.to_numeric(df['bytes'], errors='coerce').fillna(0).astype(int)
            df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce').fillna(0).astype(int)
            df['request'] = df['request'].astype(str)
            
            print("Successfully parsed as CSV with mapped columns.")
            return df
            
    except Exception as e:
        print(f"CSV parsing failed, falling back to regex: {e}")

    # 2. Fallback: Parse using Syslog Regex (for raw log files)
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = LOG_PATTERN.match(line.strip())
                if match:
                    data.append(match.groups())
        
        if data:
            columns = [
                "syslog_ts", "host", "process", "ip1", "ip2", "session_id", "domain", 
                "apache_ts", "request", "status", "bytes", "response_time", "referer", "user_agent"
            ]
            df = pd.DataFrame(data, columns=columns)
            return df
    except Exception as e:
        print(f"Regex parsing failed: {e}")
    
    # 3. Last Resort: Return empty
    return pd.DataFrame()

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
            df = parse_logs(LOG_FILE_PATH)
            # Process for AE
            df_ae = self.preprocess_ae(df)
            self.ae_background = shap.sample(df_ae, 50)
            
            def predict_error(X):
                if isinstance(X, pd.DataFrame):
                    X = X.values
                rec = self.ae_model.predict(X, verbose=0)
                return np.mean(np.square(X - rec), axis=1)

            self.ae_explainer = shap.KernelExplainer(predict_error, self.ae_background)
        
        # Iso Forest Explainer
        self.iso_explainer = shap.TreeExplainer(self.iso_forest)

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
        df = parse_logs(file_path)
        if df.empty:
            return {"error": "No data found or parsing failed."}
        
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
        iso_anomalies = df[iso_anomalies_mask].copy()
        iso_anomalies['score'] = iso_scores_all[iso_anomalies_mask]
        
        results = {
            "autoencoder": {
                "count": int(np.sum(ae_anomalies_mask)),
                "anomalies": [],
                "all_anomalies": []
            },
            "isolation_forest": {
                "count": int(np.sum(iso_anomalies_mask)),
                "anomalies": [],
                "all_anomalies": []
            }
        }
        
        # Calculate Critical Thresholds (Top 10% severe)
        ae_crit_thresh = np.percentile(ae_anomalies['error'], 90) if not ae_anomalies.empty else float('inf')
        # For Iso Forest, lower score is more anomalous. So bottom 10%.
        iso_crit_thresh = np.percentile(iso_anomalies['score'], 10) if not iso_anomalies.empty else float('-inf')
        
        # --- Generate Explanations for ALL Autoencoder Anomalies ---
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
            
        # --- Generate Explanations for ALL Isolation Forest Anomalies ---
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
        
        # --- Generate Explanations for ALL Anomalies ---
        # Note: This loops through ALL detected anomalies to get AI analysis.
        
        # AE Explanations
        if not ae_anomalies.empty:
            for idx in ae_anomalies.index:
                row = df.loc[idx]
                instance = X_ae.iloc[[idx]] 
                
                shap_vals = self.ae_explainer.shap_values(instance)
                shap_val = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
                shap_val = shap_val.flatten()
                
                feature_names = X_ae.columns.tolist()
                ai_analysis = self.get_ai_analysis_ae(row.to_dict(), shap_val, feature_names)
                graph_data = self.get_top_features(shap_val, feature_names)

                # Update "all_anomalies" list (used for report)
                for item in results["autoencoder"]["all_anomalies"]:
                    if item["index"] == idx:
                        item["explanation"] = ai_analysis["explanation"]
                        item["attack_type"] = ai_analysis["attack_type"]
                        item["severity"] = ai_analysis["severity"]
                        item["llm_prediction"] = ai_analysis["llm_prediction"]
                        break

                # Add to detailed anomalies list (for UI display)
                results["autoencoder"]["anomalies"].append({
                    "index": int(idx),
                    "log": row.to_dict(),
                    "error": float(ae_anomalies.loc[idx, 'error']),
                    "explanation": ai_analysis["explanation"],
                    "attack_type": ai_analysis["attack_type"],
                    "severity": ai_analysis["severity"],
                    "llm_prediction": ai_analysis["llm_prediction"],
                    "graph_data": graph_data
                })

        # Iso Forest Explanations
        if not iso_anomalies.empty:
            # Calculate SHAP values for all anomalies at once if possible or loop
            # Here we loop to keep it simple and aligned
            
            shap_values_iso = self.iso_explainer.shap_values(X_iso.loc[iso_anomalies.index])
            
            for i, idx in enumerate(iso_anomalies.index):
                row = df.loc[idx]
                shap_val = shap_values_iso[i]
                feature_names = X_iso.columns.tolist()
                ai_analysis = self.get_ai_analysis_iso(row.to_dict(), shap_val, feature_names)
                graph_data = self.get_top_features(shap_val, feature_names)
                
                for item in results["isolation_forest"]["all_anomalies"]:
                    if item["index"] == idx:
                        item["explanation"] = ai_analysis["explanation"]
                        item["attack_type"] = ai_analysis["attack_type"]
                        item["severity"] = ai_analysis["severity"]
                        item["llm_prediction"] = ai_analysis["llm_prediction"]
                        break
                
                results["isolation_forest"]["anomalies"].append({
                    "index": int(idx),
                    "log": row.to_dict(),
                    "explanation": ai_analysis["explanation"],
                    "attack_type": ai_analysis["attack_type"],
                    "severity": ai_analysis["severity"],
                    "llm_prediction": ai_analysis["llm_prediction"],
                    "graph_data": graph_data
                })
                
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
