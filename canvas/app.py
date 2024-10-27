from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set the folder where files will be saved
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable to store the last uploaded filename
last_uploaded_filename = None

@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle SVC file uploads
@app.route('/upload_svc', methods=['POST'])
def upload_svc():
    global last_uploaded_filename
    if 'svc_file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    svc_file = request.files['svc_file']
    filename = secure_filename(svc_file.filename)
    svc_file.save(os.path.join(UPLOAD_FOLDER, filename))  # Save file to UPLOAD_FOLDER
    last_uploaded_filename = filename  # Store the filename
    return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shut down the Flask server."""
    shutdown_server()
    return jsonify({'message': 'Server is shutting down.'}), 200

def shutdown_server():
    """Shutdown the Flask server using os.kill."""
    os.kill(os.getpid(), signal.SIGINT)  # Send the interrupt signal to the current process

# Route for checking file upload status
@app.route('/check_upload', methods=['GET'])
def check_upload():
    global last_uploaded_filename
    if last_uploaded_filename is not None:
        filename = last_uploaded_filename
        last_uploaded_filename = None
        return jsonify({"message": "File uploaded", "filename": filename}), 200
    else:
        return jsonify({"message": "No file uploaded yet"}), 204

if __name__ == '__main__':
    app.run(debug=True)
