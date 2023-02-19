## Create flask app boilerplate
from flask import Flask, request, jsonify, render_template, redirect, flash, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os


# Measure disparity
import sys
sys.path.insert(0, '../')
from treehacks.measure_disparity import run_measure_disparity


UPLOAD_FOLDER = 'predictions/'
ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'LJLKjfadskjadsfl4231'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1000 * 1000 ## 64 MB

#@app.route("/", methods=["GET"])
#def index():
#    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('File type not allowed')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('measure_unfairness', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/measure-disparity/<name>')
def measure_unfairness(name):
    # Measure disparity
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    res = run_measure_disparity(path)

    # TODO: Create text explanations of each disparity metric
    return f'<h1>{name} uploaded.</h1> {res}'#send_from_directory(app.config["UPLOAD_FOLDER"], name)

if __name__ == '__main__':
    app.debug = True
    app.run()