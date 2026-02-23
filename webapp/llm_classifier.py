import openai
import json
import os
import hashlib

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".cluster_cache.json")

def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        print(f"Error saving cluster cache: {e}")

try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from keys import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def classify_cluster(cluster_summary):
    """
    Query OpenAI to classify a cluster of logs based on common behavioral patterns.
    
    Args:
        cluster_summary (dict): Enriched cluster data computed by cluster_summarizer.
        
    Returns:
        dict: JSON parsed dict containing attack_type, confidence, severity, etc.
    """
    cache = _load_cache()
    # Create a stable identifier for the cluster content
    summary_str = json.dumps({
        k: v for k, v in cluster_summary.items() if k not in ['representative_logs']
    }, sort_keys=True)
    summary_hash = hashlib.md5(summary_str.encode('utf-8')).hexdigest()
    
    if summary_hash in cache:
        print(f"Loaded cluster classification from cache: {summary_hash}")
        return cache[summary_hash]

    if not OPENAI_API_KEY:
        return _get_dummy_cluster_analysis("Missing API Key")
        
    prompt = f"""
    You are a cybersecurity expert analyzing a cluster of anomalous server logs.
    
    Cluster Summary:
    - Size: {cluster_summary['size']} logs
    - Time Span: {cluster_summary['time_span']}
    - Top Source IP: {cluster_summary['top_source_ip']}
    - Top Destination IP: {cluster_summary['top_destination_ip']}
    - Mean Response Time: {cluster_summary['mean_response_time']}
    - Mean Bytes: {cluster_summary['mean_bytes']}
    - Failed Login Rate: {cluster_summary['failed_login_rate']}
    - Top 3 SHAP Features (anomalous indicators): {', '.join(cluster_summary['top_3_shap_features'])}
    
    Representative Logs (first 3):
    {json.dumps(cluster_summary['representative_logs'][:3], indent=2)}
    
    Based on this behavioral pattern across the cluster, provide a JSON response with ONLY the following keys:
    - "attack_type": "Short name of attack. Explicitly consider threats like: Cross Site Scripting (XSS), Sensitive Information Disclosure, SQL Injections, Insecure Deserialization, Broken Authentication, SSTI, Path Traversals, OS Command Injection, CSRF, Rate limiting anomalies, IDOR, Clickjacking, Insecure input validation, Open redirect, Cache Deception, Cache Poisoning, LFI, SSRF, Hardcoded Credentials, Remote Code Execution (RCE), Authentication Failures, Recon/Scanner Indicators, DoS, Data Exfiltration, or Configuration Issue."
    - "confidence": "Decimal between 0 and 1 indicating your confidence"
    - "severity": "Critical, High, Medium, or Low"
    - "reasoning": "1-2 sentence explanation of why this grouping represents this attack"
    - "common_pattern": "Describe the common behavior in this cluster"
    - "mitigation_steps": ["Step 1", "Step 2"]
    
    Return ONLY valid JSON.
    """
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        result = json.loads(content)
        cache[summary_hash] = result
        _save_cache(cache)
        return result
        
    except Exception as e:
        return _get_dummy_cluster_analysis(f"LLM Error: {str(e)}")

def _get_dummy_cluster_analysis(message):
    return {
        "attack_type": "Unclassified Cluster",
        "confidence": "0.0",
        "severity": "Unknown",
        "reasoning": message,
        "common_pattern": "Unknown",
        "mitigation_steps": []
    }
