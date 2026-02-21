import pandas as pd
import numpy as np

def summarize_cluster(cluster_dict, df, shap_values_dict, feature_names):
    """
    Summarize a collection of clustered anomalies.
    
    Args:
        cluster_dict (dict): Dictionary mapping cluster IDs to their data.
        df (pd.DataFrame): Original full DataFrame.
        shap_values_dict (dict): Maps index to the array of SHAP values.
        feature_names (list): Ordered feature names.
        
    Returns:
        dict: Summary of clusters enriched with statistics.
    """
    summaries = {}
    for c_id, c_data in cluster_dict.items():
        indices = c_data["log_indices"]
        c_df = df.loc[indices]
        
        # Time range (try to use 'syslog_ts' or 'apache_ts')
        time_span = "Unknown"
        ts_cols = [c for c in c_df.columns if 'ts' in c or 'time' in c and 'response' not in c]
        if 'syslog_ts' in c_df.columns:
            sorted_times = sorted(c_df['syslog_ts'].dropna().astype(str).tolist())
            if sorted_times:
                time_span = f"{sorted_times[0]} - {sorted_times[-1]}"
        elif 'apache_ts' in c_df.columns:
            sorted_times = sorted(c_df['apache_ts'].dropna().astype(str).tolist())
            if sorted_times:
                time_span = f"{sorted_times[0]} - {sorted_times[-1]}"
                
        # Top source IP
        top_src_ip = "Unknown"
        if 'ip1' in c_df.columns and not c_df['ip1'].dropna().empty:
            top_src_ip = c_df['ip1'].mode().iloc[0]
            
        # Top destination IP
        top_dest_ip = "Unknown"
        if 'ip2' in c_df.columns and not c_df['ip2'].dropna().empty:
            top_dest_ip = c_df['ip2'].mode().iloc[0]
            
        # Top port
        top_port = "N/A"
        if 'port_id' in c_df.columns and not c_df['port_id'].dropna().empty:
            top_port = str(c_df['port_id'].mode().iloc[0])
            
        # Mean stats
        if 'response_time' in c_df.columns:
            mean_resp = float(pd.to_numeric(c_df['response_time'], errors='coerce').mean())
            if np.isnan(mean_resp): mean_resp = 0.0
        else:
            mean_resp = 0.0
            
        if 'bytes' in c_df.columns:
            mean_bytes = float(pd.to_numeric(c_df['bytes'], errors='coerce').mean())
            if np.isnan(mean_bytes): mean_bytes = 0.0
        else:
            mean_bytes = 0.0
        
        # Failed login rate (status 401 or 403)
        failed_logins = 0
        if 'status' in c_df.columns:
            failed_logins = len(c_df[c_df['status'].astype(str).str.startswith('40')])
        fail_rate = float(failed_logins / len(c_df)) if len(c_df) > 0 else 0.0
        
        # Aggregate SHAP to find cluster behavior profile
        cluster_shap = []
        for idx in indices:
            if idx in shap_values_dict:
                val = shap_values_dict[idx]
                # convert to numpy 1D if needed
                if isinstance(val, list): val = np.array(val)
                elif isinstance(val, float): val = np.array([val]*len(feature_names)) # fallback
                cluster_shap.append(val)
                
        top_3_shap = []
        if cluster_shap:
            avg_shap = np.mean(np.array(cluster_shap), axis=0)
            feat_importance = sorted(zip(feature_names, avg_shap), key=lambda x: -abs(x[1]))
            top_3_shap = [f[0] for f in feat_importance[:3]]
            
        # Representative logs
        reps = c_df.head(5).to_dict('records')
        
        summaries[c_id] = {
            "size": len(indices),
            "log_indices": indices,
            "cluster_confidence_score": c_data.get("cluster_confidence_score", 1.0),
            "time_span": time_span,
            "top_source_ip": str(top_src_ip),
            "top_destination_ip": str(top_dest_ip),
            "top_port": str(top_port),
            "mean_response_time": mean_resp,
            "mean_bytes": mean_bytes,
            "failed_login_rate": fail_rate,
            "top_3_shap_features": top_3_shap,
            "representative_logs": reps
        }
    return summaries
