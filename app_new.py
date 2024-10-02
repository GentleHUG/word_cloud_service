import os
from flask import Flask, request, render_template, redirect, url_for
from file_processor import read_file_lines
from content_processor_new import ContentProcessor
import CONFIG
import logging

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# TODO: дописать импорт для этого и разнести по папкам
# TODO: дописать параметры, которые нужно получать классу, а именно:
# def __init__(self, ru_words_path: str, en_words_path: str, enable_trans: bool = False):
cont_proc = ContentProcessor(...)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Создаем папку для хранения загруженных файлов
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    if file and (file.filename.endswith(".txt") or file.filename.endswith(".csv")):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        content = read_file_lines(filepath)

        # TODO: куда то сюда присунуть file.filename))))
        # image_path, json_data = process_content(content, file.filename)
        # TODO: прописать аргументы
        preprocessed = cont_proc.preprocess(content)
        result = cont_proc.process(preprocessed)  # TODO: отсюда возвращается JSON надо развернуть Андрей

        return render_template("wordcloud.html", image=image_path, json_data=json_data)

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=CONFIG.PORT)
