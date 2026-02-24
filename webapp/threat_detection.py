import pandas as pd
import numpy as np
import re
from urllib.parse import unquote
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter

class ThreatFeatureExtractor:
    def __init__(self):
        # Compiled regex patterns for performance
        self.patterns = {
            'xss': re.compile(r'(<script>|javascript:|onerror=|onload=|eval\(|alert\()', re.IGNORECASE),
            'sqli': re.compile(r'(\'|\"|;|union\s+select|insert\s+into|update\s+set|delete\s+from|drop\s+table|--|\#|\/\*|\*\/)', re.IGNORECASE),
            'path_traversal': re.compile(r'(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/)', re.IGNORECASE),
            'cmd_injection': re.compile(r'(;|\||\|\||&|&&|\$\(|\`|bash\s+-c|sh\s+-c)', re.IGNORECASE),
            'lfi': re.compile(r'(/etc/passwd|/windows/win.ini|file://|php://input)', re.IGNORECASE),
            'rce': re.compile(r'(eval\(|exec\(|system\(|passthru\(|shell_exec\()', re.IGNORECASE)
        }

    def _url_entropy(self, url):
        if not url: return 0
        p, lns = Counter(url), float(len(url))
        return -sum(count/lns * np.log2(count/lns) for count in p.values())

    def extract_features(self, df):
        if df.empty:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=df.index)
        
        # Helper to extract URL from request string "METHOD /path HTTP/1.1"
        def get_url(req):
            try:
                parts = req.split()
                if len(parts) >= 2:
                    return unquote(parts[1])
                return ""
            except:
                return ""

        urls = df['request'].apply(get_url)
        
        # URL Features
        features['url_length'] = urls.apply(len)
        features['url_depth'] = urls.apply(lambda x: x.count('/'))
        # Estimate special chars count (non-alphanumeric)
        features['special_chars'] = urls.apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)) if x else 0)
        features['entropy'] = urls.apply(self._url_entropy)
        
        # Pattern Matching Scores (Binary)
        for name, pattern in self.patterns.items():
            features[f'{name}_score'] = urls.apply(lambda x: 1 if x and pattern.search(x) else 0)
            
        # Behavioral Features 
        # Convert status to binary error indicator
        if 'status' in df.columns:
            features['status_error'] = df['status'].apply(lambda x: 1 if int(x) >= 400 else 0)
        else:
            features['status_error'] = 0
            
        # Log scaled bytes and time
        if 'bytes' in df.columns:
            features['bytes_log'] = np.log1p(pd.to_numeric(df['bytes'], errors='coerce').fillna(0))
        else:
            features['bytes_log'] = 0
            
        if 'response_time' in df.columns:
            features['time_log'] = np.log1p(pd.to_numeric(df['response_time'], errors='coerce').fillna(0))
        else:
            features['time_log'] = 0
        
        return features

class ThreatClassifier:
    def classify(self, features_df):
        classifications = pd.Series(['Unknown'] * len(features_df), index=features_df.index)
        
        # Rule-based classification 
        rules = [
            ('SQL Injection', features_df['sqli_score'] > 0),
            ('XSS', features_df['xss_score'] > 0),
            ('Path Traversal', features_df['path_traversal_score'] > 0),
            ('Command Injection', features_df['cmd_injection_score'] > 0),
            ('LFI', features_df['lfi_score'] > 0),
            ('RCE', features_df['rce_score'] > 0)
        ]
        
        for label, mask in rules:
            classifications[mask] = label
            
        return classifications

class ThreatAwareClusterer:
    def __init__(self, eps=0.5, min_samples=3):
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def fit_predict(self, features_df):
        if features_df.empty:
            return np.array([])
            
        # Select features for clustering
        # We focus on the threat scores and structural features
        # Filter columns that actually exist
        cluster_cols = [c for c in features_df.columns if '_score' in c or c in ['url_length', 'url_depth', 'entropy', 'time_log', 'special_chars']]
        
        X = features_df[cluster_cols].fillna(0)
        
        if X.empty or X.shape[1] == 0:
             return np.array([-1] * len(features_df))

        # Scale features
        try:
            X_scaled = self.scaler.fit_transform(X)
            labels = self.clusterer.fit_predict(X_scaled)
        except Exception as e:
            print(f"Clustering failed: {e}")
            labels = np.array([-1] * len(features_df))
            
        return labels

class ThreatDetectionPipeline:
    def __init__(self):
        self.extractor = ThreatFeatureExtractor()
        self.classifier = ThreatClassifier()
        self.clusterer = ThreatAwareClusterer()
        
    def analyze(self, anomalies_df):
        if anomalies_df.empty:
            return pd.DataFrame()
            
        # 1. Feature Extraction
        features = self.extractor.extract_features(anomalies_df)
        
        # 2. Rule-Based Classification
        rule_labels = self.classifier.classify(features)
        
        # 3. Clustering 
        # We cluster all anomalies to find groupings
        cluster_labels = self.clusterer.fit_predict(features)
        
        # 4. Result Assembly
        results = pd.DataFrame(index=anomalies_df.index)
        results['rule_label'] = rule_labels
        results['cluster_id'] = cluster_labels
        
        # 5. Cluster Interpretation
        cluster_summaries = {}
        unique_clusters = set(cluster_labels)
        if -1 in unique_clusters: unique_clusters.remove(-1) # -1 is noise
        
        for cid in unique_clusters:
            mask = cluster_labels == cid
            cluster_subset = features[mask]
            subset_labels = rule_labels[mask]
            
            # Dominant label
            common_label = subset_labels.mode()
            label_str = common_label[0] if not common_label.empty else "Unknown"
            
            if label_str == "Unknown":
                # Try to infer from average high scores if any
                score_cols = [c for c in features.columns if '_score' in c]
                if score_cols:
                    means = cluster_subset[score_cols].mean()
                    top_score = means.idxmax()
                    if means[top_score] > 0.2: # Threshold
                        label_str = f"Pattern: {top_score.replace('_score', '').upper()}"
                    else:
                        label_str = "Unclassified Cluster"
                else: 
                     label_str = "Unclassified Cluster"
            
            cluster_summaries[cid] = label_str
            
        # Map cluster labels back
        results['cluster_label'] = results['cluster_id'].map(cluster_summaries).fillna("Noise")
        
        # Final combined label: Rule > Cluster > Unknown
        results['final_threat_type'] = results.apply(
            lambda x: x['rule_label'] if x['rule_label'] != 'Unknown' else (x['cluster_label'] if x['cluster_label'] != 'Noise' else 'Unknown Pattern'),
            axis=1
        )
        
        return results
