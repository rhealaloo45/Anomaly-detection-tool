from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import json
import numpy as np
from model_utils import handler
from report_generator import generate_report

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, 
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): 
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Analyze
        results = handler.analyze(filepath)
        
        if "error" in results:
            return render_template('index.html', error=results["error"])
        
        # Cache results to JSON for consistency in reports
        try:
            results_path = filepath + ".json"
            with open(results_path, 'w') as f:
                json.dump(results, f, cls=NumpyEncoder)
        except Exception as e:
            print(f"Error caching results: {e}")
            
        return render_template('results.html', results=results, filename=file.filename)

@app.route('/download/<model_type>/<filename>')
def download_report(model_type, filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found.", 404
        
    # Try to load cached results first (for consistency)
    results_path = filepath + ".json"
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading cached results, re-analyzing: {e}")
            results = handler.analyze(filepath)
    else:
        results = handler.analyze(filepath) # Fallback
    
    if model_type == 'autoencoder':
        model_results = results['autoencoder']
    elif model_type == 'isolation_forest':
        model_results = results.get('isolation_forest', {})
    else:
        return "Invalid model type.", 400
        
    report_path = generate_report(model_type, model_results, filename)
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)

