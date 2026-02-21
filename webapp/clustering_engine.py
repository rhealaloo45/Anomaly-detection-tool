import hdbscan
import numpy as np

def cluster_anomalies(features_df, anomalies_df):
    """
    Cluster anomalies using HDBSCAN.
    
    Args:
        features_df (pd.DataFrame): Scaled numerical features or latent vectors.
        anomalies_df (pd.DataFrame): Original logs for the anomalies.
    
    Returns:
        dict: A dictionary of clusters mapped by cluster_id.
    """
    if features_df is None or features_df.empty or len(features_df) < 2:
        return {}

    # Map indices back to original DataFrame
    indices = anomalies_df.index.tolist()
    
    # Configure HDBSCAN - minimum cluster size is small for logs but adaptable
    min_cluster_size = max(2, min(5, len(features_df) // 2))
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', prediction_data=True)
    labels = clusterer.fit_predict(features_df.values)
    probs = clusterer.probabilities_
    
    cluster_dict = {}
    for i, row_idx in enumerate(indices):
        label = labels[i]
        prob = probs[i]
        
        # -1 indicates noise in HDBSCAN. We can either group them into an "Unclassified" cluster 
        # or discard. Requirements say "Remove noise cluster (-1 label)". 
        if label == -1:
            continue
            
        cluster_id = f"Cluster_{label}"
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = {
                "log_indices": [],
                "size": 0,
                "confidence_scores": []
            }
            
        cluster_dict[cluster_id]["log_indices"].append(row_idx)
        cluster_dict[cluster_id]["size"] += 1
        cluster_dict[cluster_id]["confidence_scores"].append(prob)
        
    # Finalize cluster summaries
    final_clusters = {}
    for c_id, c_data in cluster_dict.items():
        if c_data["size"] > 0:
            c_data["cluster_confidence_score"] = float(np.mean(c_data["confidence_scores"]))
            del c_data["confidence_scores"]
            final_clusters[c_id] = c_data
            
    return final_clusters
