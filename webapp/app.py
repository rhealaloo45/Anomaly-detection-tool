from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from model_utils import handler
from report_generator import generate_report

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
            
        return render_template('results.html', results=results, filename=file.filename)

@app.route('/download/<model_type>/<filename>')
def download_report(model_type, filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found.", 404
        
    results = handler.analyze(filepath) # Re-analyze to get data
    
    if model_type == 'autoencoder':
        anomalies = results['autoencoder']['all_anomalies']
    elif model_type == 'isolation_forest':
        anomalies = results['isolation_forest']['all_anomalies']
    else:
        return "Invalid model type.", 400
        
    report_path = generate_report(model_type, anomalies, filename)
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
