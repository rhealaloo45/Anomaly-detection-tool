import pandas as pd
import numpy as np
import re
import shap
import openai
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import API Key
try:
    from keys import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = None
    print("Warning: keys.py not found or OPENAI_API_KEY missing.")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Data Loading and Parsing ---
def parse_logs(file_path):
    print(f"Loading and parsing logs from {file_path}...")
    # Regex pattern from notebooks
    log_pattern = re.compile(r'<(\d+)>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\S+)\s+(\S+)\s+(\S+)\s+"(.*?)"\s+"(.*?)"')
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            match = log_pattern.match(line.strip())
            if match:
                data.append(match.groups())

    columns = [
        'Priority', 'Syslog_Timestamp', 'Server', 'Process', 'Client_IP', 
        'Second_IP', 'Port_ID', 'Host', 'Log_Timestamp', 'Request', 
        'Status', 'Size', 'Duration', 'Referer', 'User_Agent'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Numeric conversions
    df['Status'] = pd.to_numeric(df['Status'], errors='coerce').fillna(0)
    df['Size'] = pd.to_numeric(df['Size'].replace('-', '0'), errors='coerce').fillna(0)
    df['Duration'] = pd.to_numeric(df['Duration'].replace('-', '0'), errors='coerce').fillna(0)
    
    print(f"Loaded {len(df)} records.")
    return df

# --- 2. Feature Engineering ---
def preprocess_features(df):
    print("Preprocessing features...")
    # Extract Method
    df['Method'] = df['Request'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'UNKNOWN')
    
    # Select raw features
    data_model = df[['Status', 'Size', 'Duration', 'Method']].copy()
    
    # One-Hot Encode Method (Used for both AE and IF for consistency)
    data_model = pd.get_dummies(data_model, columns=['Method'])
    
    # Log transform Size and Duration
    data_model['Size'] = np.log1p(data_model['Size'])
    data_model['Duration'] = np.log1p(data_model['Duration'])
    
    # Convert to float32
    data_model = data_model.astype('float32')
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['Status', 'Size', 'Duration']
    data_model[numerical_cols] = scaler.fit_transform(data_model[numerical_cols])
    
    return data_model, scaler

# --- 3. Models ---

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation="relu")(input_layer)
    encoder = Dense(8, activation="relu")(encoder)
    encoder = Dense(4, activation="relu")(encoder)
    
    decoder = Dense(8, activation="relu")(encoder)
    decoder = Dense(16, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="sigmoid")(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_ensemble(X_train, X_test):
    # --- Autoencoder ---
    print("\nTraining Autoencoder...")
    input_dim = X_train.shape[1]
    ae = build_autoencoder(input_dim)
    ae.fit(
        X_train, X_train,
        epochs=30, # Reduced epochs for script speed
        batch_size=32,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=0
    )
    
    # Calculate threshold
    reconstructions = ae.predict(X_train)
    train_loss = tf.keras.losses.mse(reconstructions, X_train).numpy()
    ae_threshold = np.mean(train_loss) + 2 * np.std(train_loss)
    print(f"Autoencoder Threshold: {ae_threshold:.4f}")
    
    # --- Isolation Forest ---
    print("\nTraining Isolation Forest...")
    # contamination=0.05 assumes ~5% anomalies
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(X_train)
    
    return ae, ae_threshold, iso_forest

# --- 4. Ensemble Prediction ---
def ensemble_predict(ae, ae_threshold, iso_forest, X):
    # AE Prediction
    reconstructions = ae.predict(X)
    mse = tf.keras.losses.mse(reconstructions, X).numpy()
    ae_anomalies = mse > ae_threshold
    
    # IF Prediction (-1 is anomaly)
    if_preds = iso_forest.predict(X)
    if_anomalies = if_preds == -1
    
    # Ensemble: Union (Flag if EITHER model thinks it's an anomaly)
    # You could also do Intersection (AND) for higher precision
    ensemble_anomalies = ae_anomalies | if_anomalies
    
    return ensemble_anomalies, mse, if_preds

# --- 5. Explainability ---
def generate_openai_explanation(log_entry, shap_values, feature_names):
    if not OPENAI_API_KEY:
        return "OpenAI API Key not provided."
        
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Top features
    feat_importance = sorted(zip(feature_names, shap_values), key=lambda x: -abs(x[1]))
    top_features = feat_importance[:5]
    shap_text = ", ".join([f"{f}: {v:.4f}" for f, v in top_features])
    
    prompt = f"""
    Analyze this server log anomaly detected by an ensemble model (Autoencoder + Isolation Forest).
    Log Details: {log_entry}
    Key contributors to anomaly (SHAP values): {shap_text}
    
    Provide a concise explanation of why this is anomalous.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Using 3.5 for speed/cost in script
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {e}"

def explain_anomalies(X_bg, X_anomalies, df_anomalies, ae_model, iso_model, feature_names):
    print("\n--- Generating Explanations for Top Anomalies ---")
    
    # We'll use KernelExplainer on the Autoencoder as the primary explainer for simplicity in this ensemble script,
    # or we could explain IF. Let's explain the AE's perspective since it gives a continuous error score which is often more granular.
    # Alternatively, we can use TreeExplainer for IF.
    # Let's use AE KernelExplainer as it's more generic for the ensemble concept (treating model as black box).
    
    # Wrapper for ensemble probability or score? 
    # To keep it simple and fast, let's explain the Autoencoder Reconstruction Error using KernelExplainer.
    
    def ae_predict_error(data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        rec = ae_model.predict(data)
        return np.mean(np.square(data - rec), axis=1)

    # Use a small background sample
    background = shap.sample(X_bg, 20) 
    explainer = shap.KernelExplainer(ae_predict_error, background)
    
    # Explain top 3
    for idx in range(min(3, len(X_anomalies))):
        instance = X_anomalies.iloc[[idx]]
        log_row = df_anomalies.iloc[idx]
        
        print(f"\nAnomaly #{idx+1}")
        print(f"Log: {log_row['Request']} | Status: {log_row['Status']} | IP: {log_row['Client_IP']}")
        
        shap_vals = explainer.shap_values(instance)
        
        # Explanation
        explanation = generate_openai_explanation(log_row.to_dict(), shap_vals[0], feature_names)
        print("Analysis:", explanation)


# --- Main ---
if __name__ == "__main__":
    file_path = 'synthetic_logs_10k.csv'
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        exit()
        
    # 1. Load
    df = parse_logs(file_path)
    
    # 2. Preprocess
    data_model, scaler = preprocess_features(df)
    
    # Split
    X_train, X_test = train_test_split(data_model, test_size=0.2, random_state=42)
    
    # 3. Train
    ae, ae_thresh, iso = train_ensemble(X_train, X_test)
    
    # 4. Predict on Test Set
    anomalies_mask, ae_mse, if_preds = ensemble_predict(ae, ae_thresh, iso, X_test)
    
    num_anomalies = np.sum(anomalies_mask)
    print(f"\nDetected {num_anomalies} anomalies in test set out of {len(X_test)} records.")
    
    if num_anomalies > 0:
        # Get the original rows for the anomalies
        # X_test is a slice, we need the indices to map back to df
        test_indices = X_test.index
        anomaly_indices = test_indices[anomalies_mask]
        
        df_anomalies = df.loc[anomaly_indices]
        X_anomalies = X_test[anomalies_mask]
        
        # 5. Explain
        explain_anomalies(X_train, X_anomalies, df_anomalies, ae, iso, X_test.columns)
    else:
        print("No anomalies detected.")

    # --- 6. Inference on New File (Optional) ---
    new_file = 'user_log.csv'
    if os.path.exists(new_file):
        print(f"\nProcessing new file: {new_file}...")
        try:
            # Parse new file
            df_new = parse_logs(new_file)
            
            if not df_new.empty:
                # Preprocess (reuse scaler and method columns)
                # We need to ensure columns match exactly what model expects
                
                # 1. Basic Prep
                df_new['Method'] = df_new['Request'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'UNKNOWN')
                data_new = df_new[['Status', 'Size', 'Duration', 'Method']].copy()
                data_new = pd.get_dummies(data_new, columns=['Method'])
                data_new['Size'] = np.log1p(data_new['Size'])
                data_new['Duration'] = np.log1p(data_new['Duration'])
                data_new = data_new.astype('float32')
                
                # 2. Align Columns (Add missing method columns if any)
                # Get expected columns from training data
                expected_cols = X_train.columns
                for col in expected_cols:
                    if col not in data_new.columns:
                        data_new[col] = 0
                
                # Reorder to match training
                data_new = data_new[expected_cols]
                
                # 3. Scale
                numerical_cols = ['Status', 'Size', 'Duration']
                data_new[numerical_cols] = scaler.transform(data_new[numerical_cols])
                
                # Predict
                anomalies_mask_new, _, _ = ensemble_predict(ae, ae_thresh, iso, data_new)
                num_new_anomalies = np.sum(anomalies_mask_new)
                print(f"Found {num_new_anomalies} anomalies in {new_file}.")
                
                if num_new_anomalies > 0:
                     # Reset index to match mask
                     df_new = df_new.reset_index(drop=True)
                     df_anomalies_new = df_new.loc[anomalies_mask_new]
                     X_anomalies_new = data_new[anomalies_mask_new]
                     explain_anomalies(X_train, X_anomalies_new, df_anomalies_new, ae, iso, expected_cols)
            
        except Exception as e:
            print(f"Error processing new file: {e}")

