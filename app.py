import sys
import threading
import time
stdout = sys.stdout
from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import shutil
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from FallDetection import FallDetector
# change logging to the default because super_gradients changes this
sys.stdout = stdout


# create an instance of the FLask application
app = Flask(__name__)

# Configuration
# directories containing the model's weight and the detection output
##################################################################################################
CONFIDENCE_THRESHOLD = 0.70 # set this value to balance between precision & recalls
##################################################################################################
DETECTED_FRAMES_FOLDER = './outputs/fall_detected_frames_bounding_box/'
TRAINED_MODEL_PATH = './average_model.pth'

# directories for uploading videos and processing the images
UPLOAD_FOLDER = './static/videos/'
PROCESSED_FOLDER = './static/processed'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# create a new blank processed folder for each run
if os.path.exists(PROCESSED_FOLDER):
    shutil.rmtree(PROCESSED_FOLDER)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = ['mp4']
app.config['SAVED_MODEL'] = TRAINED_MODEL_PATH


# Application
@app.route('/', methods=['GET', 'POST'])
def index():
    # homepage
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # output folder
    output_path = './static/processed/'
    # upload functionality of the webpage
    if 'file' not in request.files:
        return "No video file found"
    file = request.files['file']
    if file.filename == "":
        return "No video file selected"
    if file and not file.filename.endswith('.mp4'):
        return "Invalid file format. Please upload an MP4 file."
    
    # Save the file to your server
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)
    
    # Load the fall detection model
    fall_detector = FallDetector(trained_model_path=TRAINED_MODEL_PATH)
    # Perform detection on the video and showed the detect frames
    fall_image_predictions = fall_detector.detect_video(video_path=video_path, confidence_threshold=CONFIDENCE_THRESHOLD)
    if len(fall_image_predictions) > 0:
        # if the returned length of fall image predictions > 0, then fall is detected
        # save the preprocessed frames in PROCESSED_FOLDER
        for i, fall_image_prediction in enumerate(fall_image_predictions):
            fall_image_array = np.array(fall_image_prediction.image)
            # convert numpy array to PIL image
            image = Image.fromarray(fall_image_array.astype('uint8'), 'RGB')
            #save the image as jpg in os.path.join(PROCESSED_FOLDER, f'fall_{i+1}.jpg')
            fall_image_path = os.path.join(PROCESSED_FOLDER, f'fall_{i+1}.jpg')
            image.save(fall_image_path)
            
        # TODO: display all the saved fall images on a html page along with a big red 'FALL DETECTED' header
        return redirect(url_for('show_results', filename=filename))
    
    else:
        # TODO: display a big green header 'No Falls Detected'
        return redirect(url_for('no_fall_detected'))

@app.route('/results/<filename>')
def show_results(filename):
    images = [f for f in os.listdir(PROCESSED_FOLDER) if os.path.isfile(os.path.join(PROCESSED_FOLDER, f))]
    return render_template('results.html', images=images)

@app.route('/no_fall_detected')
def no_fall_detected():
    return render_template('no_fall_detected.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     # file upload error handling logic
#     if 'video' not in request.files:
#         return jsonify({"error": "No video file found"})
    
#     video = request.files['video']
#     if video.filename == '':
#         return jsonify({"error": "No video file selected"})
    
#     # file upload successfully if in one of the listed allowed extensions
#     if video and allowed_file(video.filename):
#         filename = secure_filename(video.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         return render_template('preview_html', video_name=video.filename)
#     return "invalid video file"
        



# def allowed_file(filename):
    # check if the filename is in one of the allowed extensions
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_app():
    flaskPort = 8786
    print('Starting Fall Detection Server...')
    app.run(host = '0.0.0.0', port = flaskPort, debug=True, use_reloader=False)
        
if __name__ == '__main__':
    sys.stdout = stdout  
    # flask_thread = threading.Thread(target=run_app)
    # flask_thread.start()
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("Shutting down Flask server...")
    app.run(debug=True)  
    print('Fall Detection Server Terminating...')
    