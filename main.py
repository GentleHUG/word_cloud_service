from fastapi import FastAPI, Response
from matplotlib import cm
from pydantic import BaseModel
from typing import List, Dict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import json
import uvicorn
import numpy as np
import random
import math
from PIL import Image, ImageDraw
from collections import Counter
import re

from data_clean import preprocess
from embeddings.model import get_top_words

with open('config.json') as config_file:
	config = json.load(config_file)

app = FastAPI(title="Word Cloud Generator", description="API для генерации облаков слов из строк.")


class ProcessWordsRequest(BaseModel):
	enable_trans: bool = False
	items: List[str]


def create_oval_mask(width, height):
	mask = Image.new("L", (width, height), 255)
	draw = ImageDraw.Draw(mask)
	draw.ellipse((0, 0, width, height), fill=0)

	return np.array(mask)


def create_egg_mask(width, height, egg_size):
	mask = Image.new("L", (width, height), "white")
	draw = ImageDraw.Draw(mask)

	center_x, center_y = width / 2, height / 2
	egg_points = []
	for t in range(0, 359):
		theta = math.radians(t)
		if t > 180:
			x = center_x + egg_size * (1 + 0.3 * math.cos(theta - math.radians(90)) ** 2) * math.cos(theta)
			y = center_y + egg_size * (1 + 0.3 * math.cos(theta - math.radians(90)) ** 2) * math.sin(theta)
		else:
			x = center_x + egg_size * math.cos(theta)
			y = center_y + egg_size * math.sin(theta)
		egg_points.append((x, y))
	draw.polygon(egg_points, fill=0)

	return np.array(mask)


def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
	colormap = cm.get_cmap('copper')
	rgba_color = colormap(random.random())
	rgb_color = tuple(int(255 * c) for c in rgba_color[:3])

	return rgb_color


# // TODO Нужно реализовать функцию обработки слов)
def precess_words(string_list: List[str], enable_trans: bool) -> Dict[str, float]:
	cleaned_words = preprocess(' '.join(string_list), enable_trans)
	top_words = get_top_words(cleaned_words)
	return top_words


"""
Мои функции для тестов, потом из уберемссс
-----------------------------------------------------------------------
"""


def extract_words(string_list: List[str]) -> List[str]:
	words = []
	for sentence in string_list:
		# Удаляем знаки препинания и разбиваем строку на слова
		words.extend(re.findall(r'\b\w+\b', sentence.lower()))
	return words


def my_process_word_func(string_list: List[str], enable_trans: bool) -> Dict[str, float]:
	counter = Counter(extract_words(string_list))
	max_freq: int = counter.most_common(1)[0][1]
	return {key: round(value / max_freq, 4) for key, value in counter.items()}


"""
-----------------------------------------------------------------------
"""


@app.post("/generate-wordcloud/", summary="Генерация облака слов", response_description="Изображение облака слов")
async def generate_wordcloud(req: ProcessWordsRequest):
	width = config['width']
	height = config['height']
	egg_size = config['egg_size']
	dpi = config['dpi']

	# Преобразуем пиксели в дюймы
	width_in_inches =  width / dpi
	height_in_inches = height / dpi

	mask = create_oval_mask(width, height)
	mask = create_egg_mask(width, height, egg_size)

	# TODO Здесь должна быть обработка с помощью ML
	aggregated_words = precess_words(req.items, req.enable_trans)

	wordcloud = WordCloud(
		width=width,
		height=height,
		background_color='red',
		mask=mask,
		scale=1,
		color_func=lambda *args, **kwargs: "white",
		# color_func=custom_color_func,
		font_path=config["fort_path"],
		prefer_horizontal=1
	).generate_from_frequencies(aggregated_words)

	plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')

	img_byte_array = io.BytesIO()
	plt.savefig(img_byte_array, format='png', bbox_inches="tight", pad_inches=0)
	img_byte_array.seek(0)

	plt.close()

	return Response(content=img_byte_array.getvalue(), media_type="image/png")


@app.post("/aggregate-words/", summary="Агрегация слов с их ценностью",
          response_description="Список сгенерированных слов с их ценностью")
async def aggregate_words(req: ProcessWordsRequest):
	return {"aggregated_words": my_process_word_func(req.items)}


if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=config['port'])
