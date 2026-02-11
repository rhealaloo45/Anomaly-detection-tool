import pandas as pd
import numpy as np
import re
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Constants
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Ensure models dir exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Regex Pattern (Common Apache/Nginx combined log format or similar custom format used in previous parts)
# Adjusting to match the likely format based on previous usage: 
# <150>Oct 10 20:30:15 server process: ...
LOG_PATTERN = re.compile(r'<(\d+)>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\S+)\s+(\S+)\s+(\S+)\s+"(.*?)"\s+"(.*?)"')

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
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}) # Safe rename
        
        # Handle 'method' column if 'request' is missing
        if 'request' not in df.columns and 'method' in df.columns:
            # Create a dummy request string from method so downstream logic works
            df['request'] = df['method'] + ' /UNKNOWN_PATH HTTP/1.1'
            
        # Ensure required columns exist, fill with defaults if missing
        if 'status' not in df.columns:
            # Heuristic: Check for any column that looks like status (integers 200-599)
            print("Warning: 'status' column missing, defaulting to 200")
            df['status'] = 200 
        if 'bytes' not in df.columns:
            df['bytes'] = 0
        if 'response_time' not in df.columns:
            df['response_time'] = 0
            
        # IMPORTANT: If 'request' is missing, we can't really train effectively without method extraction
        if 'request' not in df.columns:
             print("Warning: 'request' column missing, creating dummy")
             df['request'] = "GET / HTTP/1.1"

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
                "priority", "syslog_ts", "host", "process", "ip1", "ip2", "session_id", "domain", 
                "apache_ts", "request", "status", "bytes", "response_time", "referer", "user_agent"
            ]
            df = pd.DataFrame(data, columns=columns)
            return df
    except Exception as e:
        print(f"Regex parsing failed: {e}")
    
    # 3. Last Resort: Return empty
    return pd.DataFrame()

def train_autoencoder(df):
    print("Training Autoencoder...")
    
    # Preprocessing
    df_ae = df.copy()
    
    # Handle potentially non-numeric or missing data
    # Note: parse_logs already tries to ensure these are numeric, but redundancy is safe
    df_ae['Status'] = pd.to_numeric(df_ae['status'], errors='coerce').fillna(0)
    df_ae['Size'] = pd.to_numeric(df_ae['bytes'], errors='coerce').fillna(0)
    df_ae['Duration'] = pd.to_numeric(df_ae['response_time'], errors='coerce').fillna(0)
    
    # Extract Method
    df_ae['Method'] = df_ae['request'].apply(lambda x: x.split()[0] if isinstance(x, str) and len(x.split()) > 0 else 'UNKNOWN')
    
    # Select Features
    features_cols = ['Status', 'Size', 'Duration', 'Method']
    data_model = df_ae[features_cols].copy()
    
    # One-Hot Encode Method
    data_model = pd.get_dummies(data_model, columns=['Method'])
    
    # Save Feature Columns (Critical for inference alignment)
    model_columns = data_model.columns.tolist()
    joblib.dump(model_columns, os.path.join(MODELS_DIR, 'ae_columns.pkl'))
    
    # Log Transformation
    data_model['Size'] = np.log1p(data_model['Size'])
    data_model['Duration'] = np.log1p(data_model['Duration'])
    
    data_model = data_model.astype('float32')
    
    # Normalize
    scaler = MinMaxScaler()
    numerical_cols = ['Status', 'Size', 'Duration']
    # Fit scaler only on numerical columns
    data_model[numerical_cols] = scaler.fit_transform(data_model[numerical_cols])
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'ae_scaler.pkl'))
    
    # Convert to numeric (ensure bools from get_dummies become ints/floats)
    X = data_model.values
    
    # Train/Test Split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Build Model
    input_dim = X.shape[1]
    
    autoencoder = Sequential([
        Dense(16, activation="relu", input_shape=(input_dim,)),
        Dense(8, activation="relu"),
        Dense(4, activation="relu"),
        Dense(8, activation="relu"),
        Dense(16, activation="relu"),
        Dense(input_dim, activation="sigmoid") 
    ])
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train
    autoencoder.fit(
        X_train, X_train,
        epochs=20, # Reduced epochs for faster responsiveness in demo
        batch_size=32,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=1
    )
    
    # Calculate and Save Threshold
    reconstructions = autoencoder.predict(X)
    train_loss = np.mean(np.square(X - reconstructions), axis=1)
    threshold = np.mean(train_loss) + 2 * np.std(train_loss)
    
    # Save Model and Threshold
    autoencoder.save(os.path.join(MODELS_DIR, 'autoencoder.h5'))
    joblib.dump(threshold, os.path.join(MODELS_DIR, 'ae_threshold.pkl'))
    print(f"Autoencoder trained. Threshold: {threshold}")

def train_isolation_forest(df):
    print("Training Isolation Forest...")
    
    df_iso = df.copy()
    
    # Clean and Encode
    # Note: parse_logs already ensures these fields exist
    df_iso['bytes'] = pd.to_numeric(df_iso['bytes'], errors='coerce').fillna(0).astype(int)
    df_iso['response_time'] = pd.to_numeric(df_iso['response_time'], errors='coerce').fillna(0).astype(int)
    df_iso['status'] = pd.to_numeric(df_iso['status'], errors='coerce').fillna(0).astype(int)
    
    df_iso['http_method'] = df_iso['request'].apply(lambda x: x.split()[0] if isinstance(x, str) and x else "UNKNOWN")
    
    method_encoder = LabelEncoder()
    df_iso['http_method_encoded'] = method_encoder.fit_transform(df_iso['http_method'])
    
    # Save Encoder
    joblib.dump(method_encoder, os.path.join(MODELS_DIR, 'iso_method_encoder.pkl'))
    
    features = ['response_time', 'bytes', 'status', 'http_method_encoded']
    X = df_iso[features]
    
    # Train
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(X)
    
    # Save Model
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    print("Isolation Forest trained.")

def retrain_models(file_path):
    """
    Main function to parse data and retrain both models.
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": "Dataset file not found."}
        
    try:
        # Universal parsing (handles both CSV and Regex logs)
        df = parse_logs(file_path)
            
        if df.empty:
             return {"success": False, "error": "Parsed dataset is empty or invalid format."}
             
        # Train
        train_autoencoder(df)
        train_isolation_forest(df)
        
        return {"success": True, "message": "Models successfully retrained and saved."}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Test run
    # retrain_models('path/to/log')
    pass
