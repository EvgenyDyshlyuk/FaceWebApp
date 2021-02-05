# https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory, flash
from werkzeug.utils import secure_filename

import utils

app = Flask(__name__)

#secret_key by os.urandom(24)
app.secret_key = \
    "e\x006\xfb\xd1r\x1f\xbc\xf4\xc7$H\xc9\xdc\x12\xd1\x16\x82U\\9\x88\xea\xd0"

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg','.JPG', '.jpe', '.jpeg', '.jif',
                                   '.jfif', '.jfi', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'tmp/uploads'
app.config['CROPPED_PATH'] = 'tmp/cropped'
app.config['TSNE_PATH'] = 'tmp/tsne'

@app.before_first_request
def delete_and_create_dirs():
    utils.delete_create_dirs([app.config['UPLOAD_PATH'],
                            app.config['CROPPED_PATH'],
                            app.config['TSNE_PATH']])

def validate_image(stream):
    """Get file format from first 512 file bytes using imghdr"""
    header = stream.read(512) # read only first 516 bytes
    stream.seek(0) # return to byte 0
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + format

@app.errorhandler(413)
def too_large(e):
    return "File is too large, max size = " + \
        str(app.config['MAX_CONTENT_LENGTH']/(1024*1024)) + " MB", 413


@app.route('/')
def index():
    files_upload = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files_upload)

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return ("Invalid image extension", 400)
        elif validate_image(uploaded_file.stream) not in \
                                            app.config['UPLOAD_EXTENSIONS']:
            return ("Invalid validated image extension", 400)            
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return ('', 204) # 204 empty response

@app.route('/uploads/<filename>')
def upload_send(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)



@app.route('/preview')
def preview():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('preview.html', files=files)
 

@app.route('/cropped')
def cropped():
    utils.mtcnn_filter_save(app.config['UPLOAD_PATH'],
                            app.config['CROPPED_PATH'])
    files_cropped = os.listdir(app.config['CROPPED_PATH'])
    print("files", files_cropped)
    return render_template('cropped.html', files=files_cropped)

@app.route('/cropped/<filename>')
def crop_send(filename):
    return send_from_directory(app.config['CROPPED_PATH'], filename)


@app.route('/tsne')
def tsne():
    utils.tsne(app.config['CROPPED_PATH'], app.config['TSNE_PATH'])
    files_tsne = os.listdir(app.config['TSNE_PATH'])
    print('files:',files_tsne)
    return render_template("tsne.html", files_tsne = files_tsne)

@app.route('/tsne/<filename>')
def tsne_send(filename):
    return send_from_directory(app.config['TSNE_PATH'], filename)



if __name__ == "__main__":
    app.run(debug=True, threaded=True)
