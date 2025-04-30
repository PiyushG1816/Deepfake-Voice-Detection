from flask import Flask, request, render_template, jsonify
from predict import predict_audio, extract_mfcc, model  # Import everything needed
import os
import numpy as np

app = Flask(__name__,template_folder="templates")
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.lower().endswith(('.wav', '.mp3', '.npy')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            prediction = predict_audio(filepath)
            
            if prediction == "silent":
                return render_template('result.html', 
                    prediction="Error: The uploaded file is silent or too quiet")
            elif prediction == "error":
                return render_template('result.html',
                    prediction="Error: Could not process the audio file")

            # Normal prediction flow
            return render_template('result.html', prediction=prediction)
        
        return "Please upload a .npy, .wav, or .mp3 file!"

    return '''
    <!doctype html>
    <title>Upload Audio File</title>
    <h1>Upload an Audio (.wav, .mp3) or MFCC (.npy) file</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/result')
def result():
    return render_template('result.html')

# 💥 New Predict API Route
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Extract MFCC features
    features = extract_mfcc(filepath)

    if features == "silent":
        return jsonify({'message': 'This is a silent audio file.'})

    if features is None:
        return jsonify({'error': 'Error processing file.'})

    # Normal prediction
    prediction = model.predict(np.expand_dims(features, axis=0))
    predicted_class = np.argmax(prediction)

    return jsonify({'prediction': int(predicted_class)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render will set this environment variable
    app.run(host="0.0.0.0", port=port)
