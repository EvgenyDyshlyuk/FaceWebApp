import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

import shutil

# https://medium.com/dev-genius/get-started-with-multiple-files-upload-using-flask-e8a2f5402e20

app=Flask(__name__)

app.secret_key = "secret key"

#It will allow below 16MB contents only, you can change it
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# remove folder if exists

# If directory exists: delete it
if os.path.isdir(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
# Create new directory
os.mkdir(UPLOAD_FOLDER)
 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpe', 'jpeg', 'jif', 'jfif', 'jfi',
                          'gif', 'png'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')
        
        ct=0
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                ct+=1

        flash(str(ct)+' file(s) successfully uploaded')
        return redirect('/')
        


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)

    