import os
from flask import Flask, request, render_template, redirect, url_for, send_file
from file_processor import read_file_lines
from json_processor import create_json_file
import CONFIG
import logging

from content_processor_new import ContentProcessor
from image_processor import ImageProcessor

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = CONFIG.UPLOAD_PATH
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

cont_proc = ContentProcessor(ru_words_path=CONFIG.RU_BANNED_WORDS_PATH, en_words_path=CONFIG.EN_BANNED_WORDS_PATH)
image_proc = ImageProcessor(CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT)

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

	enable_trans = request.form.get("enable_trans")

	file = request.files["file"]
	if file.filename == "":
		return redirect(url_for("index"))

	if file and (file.filename.endswith(".txt") or file.filename.endswith(".csv")):
		filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
		file.save(filepath)

		content = read_file_lines(filepath)
		preprocessed = cont_proc.preprocess(content, enable_trans)
		processed = cont_proc.process(preprocessed)

		res = {top_cluster.cluster_content[0]: top_cluster.cluster_weight for top_cluster in processed}

		image_path = image_proc.generate_word_cloud(res, file.filename)
		json_path = create_json_file(res, file.filename)
		return render_template("wordcloud.html", image=image_path, json=json_path)

	return redirect(url_for("index"))


@app.route("/download/<filename>")
def download_file(filename):
	return send_file(os.path.join(CONFIG.UPLOAD_PATH, filename), as_attachment=True)


if __name__ == "__main__":
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	app.run(debug=True, port=CONFIG.PORT)
