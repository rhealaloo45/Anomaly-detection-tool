from flask import Flask, render_template, request, redirect, url_for
import os
from model_utils import handler

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
