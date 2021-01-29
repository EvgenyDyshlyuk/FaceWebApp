# https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask

import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory, flash
from werkzeug.utils import secure_filename
import shutil

import utils

app = Flask(__name__)
app.secret_key = "secret key"

MAX_FILE_SIZE = 16 # MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg','.JPG', '.jpe', '.jpeg', '.jif',
    '.jfif', '.jfi', '.png', '.gif']

app.config['UPLOAD_PATH'] = 'tmp/uploads'
app.config['CROPPED_PATH'] = 'tmp/cropped'

UPLOAD_FOLDER = os.path.join(os.getcwd(), app.config['UPLOAD_PATH'])
CROPPED_FOLDER = os.path.join(os.getcwd(), app.config['CROPPED_PATH'])

if os.path.isdir('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')
os.mkdir(UPLOAD_FOLDER)
os.mkdir(CROPPED_FOLDER)


def validate_image(stream):
    """Get file format"""
    header = stream.read(512) # read only first 516 bytes
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + format

@app.errorhandler(413)
def too_large(e):
    return "File is too large, max size = "+str(MAX_FILE_SIZE)+" MB", 413

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return "Invalid image extension", 400
        elif validate_image(uploaded_file.stream) not in \
        app.config['UPLOAD_EXTENSIONS']:
            return "Invalid checked image extension", 400            
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return '', 204 #redirect('/preview') # #

@app.route('/preview')
def preview():
    #utils.mtcnn_filter_save(app.config['UPLOAD_PATH'])
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('preview.html', files=files)
 
@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/cropped')
def cropped():
    utils.mtcnn_filter_save(app.config['UPLOAD_PATH'], app.config['CROPPED_PATH'])
    files = os.listdir(app.config['CROPPED_PATH'])
    return render_template('cropped.html', files=files)

@app.route('/cropped/<filename>')
def crop(filename):
    return send_from_directory(app.config['CROPPED_PATH'], filename)



if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)
