import os
from flask import Flask, request, render_template, redirect, url_for
from file_processor import read_file_lines
from content_processor import process_content
import CONFIG

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

# Создаем папку для хранения загруженных файлов
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    enable_trans = 'enable_trans' in request.form

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file and (file.filename.endswith('.txt') or file.filename.endswith('.csv')):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        content = read_file_lines(filepath)
        image_path, json_data = process_content(content, enable_trans, file.filename)

        return render_template('wordcloud.html', image=image_path, json_data=json_data)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=CONFIG.PORT)
