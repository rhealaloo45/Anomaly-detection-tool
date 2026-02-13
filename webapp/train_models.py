import pandas as pd
import numpy as np
import re
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Constants
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'synthetic_logs_10k.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Ensure models dir exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Regex Pattern
LOG_PATTERN = re.compile(r'<150>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\d+)\s+(\S+)\s+(\d+)\s+"(.*?)"\s+"(.*?)"')

def parse_logs(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            match = LOG_PATTERN.match(line.strip())
            if match:
                data.append(match.groups())
            else:
                # Try the other regex pattern from autoencoder notebook if the first one fails or mix
                # Autoencoder pattern: r'<(\d+)>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\S+)\s+(\S+)\s+(\S+)\s+"(.*?)"\s+"(.*?)"'
                # Isolation pattern: r'<150>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\d+)\s+(\S+)\s+(\d+)\s+"(.*?)"\s+"(.*?)"'
                # The synthetic logs seem to start with <150>, so the Isolation pattern is likely correct for this file.
                pass
    
    columns = [
        "syslog_ts", "host", "process", "ip1", "ip2", "session_id", "domain", 
        "apache_ts", "request", "status", "bytes", "response_time", "referer", "user_agent"
    ]
    # Note: Autoencoder notebook uses slightly different column names but same data.
    # Autoencoder cols: 'Priority', 'Syslog_Timestamp', 'Server', 'Process', 'Client_IP', 'Second_IP', 'Port_ID', 'Host', 'Log_Timestamp', 'Request', 'Status', 'Size', 'Duration', 'Referer', 'User_Agent'
    # The Isolation notebook regex matches the file provided in context better (starts with <150> which matches the summarized file content).
    # I will use the columns from Isolation notebook as base, but ensure compatibility.
    
    df = pd.DataFrame(data, columns=columns)
    return df

def train_autoencoder(df):
    print("Training Autoencoder...")
    # Preprocessing
    # Use logic from autoencoder.ipynb but adapted to the dataframe columns
    # mapping: status->Status, bytes->Size, response_time->Duration
    
    df_ae = df.copy()
    df_ae['Status'] = pd.to_numeric(df_ae['status'], errors='coerce').fillna(0)
    df_ae['Size'] = pd.to_numeric(df_ae['bytes'].replace('-', '0'), errors='coerce').fillna(0)
    df_ae['Duration'] = pd.to_numeric(df_ae['response_time'].replace('-', '0'), errors='coerce').fillna(0)
    
    df_ae['Method'] = df_ae['request'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'UNKNOWN')
    
    # Features
    features_cols = ['Status', 'Size', 'Duration', 'Method']
    data_model = df_ae[features_cols].copy()
    
    # One-Hot Encode
    data_model = pd.get_dummies(data_model, columns=['Method'])
    
    # Save the columns to ensure alignment during inference
    model_columns = data_model.columns.tolist()
    joblib.dump(model_columns, os.path.join(MODELS_DIR, 'ae_columns.pkl'))
    
    # Log transform
    data_model['Size'] = np.log1p(data_model['Size'])
    data_model['Duration'] = np.log1p(data_model['Duration'])
    
    data_model = data_model.astype('float32')
    
    # Normalize
    scaler = MinMaxScaler()
    numerical_cols = ['Status', 'Size', 'Duration']
    data_model[numerical_cols] = scaler.fit_transform(data_model[numerical_cols])
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'ae_scaler.pkl'))
    
    # Train/Test Split
    X_train, X_test = train_test_split(data_model, test_size=0.2, random_state=42)
    
    # Build Model
    input_dim = data_model.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation="relu")(input_layer)
    encoder = Dense(8, activation="relu")(encoder)
    encoder = Dense(4, activation="relu")(encoder)
    decoder = Dense(8, activation="relu")(encoder)
    decoder = Dense(16, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="sigmoid")(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train
    autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=1
    )
    
    # Calculate Threshold
    reconstructions = autoencoder.predict(data_model)
    train_loss = tf.keras.losses.mse(reconstructions, data_model)
    threshold = np.mean(train_loss) + 2 * np.std(train_loss)
    
    # Save Model and Threshold
    autoencoder.save(os.path.join(MODELS_DIR, 'autoencoder.h5'))
    joblib.dump(threshold, os.path.join(MODELS_DIR, 'ae_threshold.pkl'))
    print("Autoencoder trained and saved.")

def train_isolation_forest(df):
    print("Training Isolation Forest...")
    # Preprocessing
    df_iso = df.copy()
    df_iso['bytes'] = df_iso['bytes'].replace('-', '0').astype(int)
    df_iso['response_time'] = df_iso['response_time'].astype(int)
    df_iso['status'] = df_iso['status'].astype(int)
    
    df_iso['http_method'] = df_iso['request'].apply(lambda x: x.split()[0] if x else "UNKNOWN")
    
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
    print("Isolation Forest trained and saved.")

if __name__ == "__main__":
    print(f"Loading data from {LOG_FILE_PATH}")
    if os.path.exists(LOG_FILE_PATH):
        df = parse_logs(LOG_FILE_PATH)
        train_autoencoder(df)
        train_isolation_forest(df)
        print("All models trained.")
    else:
        print("Log file not found.")
