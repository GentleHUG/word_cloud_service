from fastapi import FastAPI, Response
from matplotlib import cm
from pydantic import BaseModel
from typing import List
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import json
import uvicorn
import numpy as np
import random
from PIL import Image, ImageDraw

with open('config.json') as config_file:
	config = json.load(config_file)

app = FastAPI(title="Word Cloud Generator", description="API для генерации облаков слов из строк.")


class StringList(BaseModel):
	items: List[str]


def create_oval_mask(width, height):
	mask = Image.new("L", (width, height), 255)
	draw = ImageDraw.Draw(mask)
	draw.ellipse((0, 0, width, height), fill=0)

	return np.array(mask)


def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
	colormap = cm.get_cmap('plasma')
	rgba_color = colormap(random.random())
	rgb_color = tuple(int(255 * c) for c in rgba_color[:3])

	return rgb_color


def process_words(string_list: StringList) -> str:
	text = " ".join(string_list.items)
	return text


def precess_word(string_list: List[str]) -> dict[str, float]:
	return {"lol": 1.2}


@app.post("/generate-wordcloud/", summary="Генерация облака слов", response_description="Изображение облака слов")
async def generate_wordcloud(string_list: StringList):
	width = config['width']
	height = config['height']

	mask = create_oval_mask(width, height)

	# TODO Здесь должна быть обработка с помощью ML
	text = process_words(string_list)

	wordcloud = WordCloud(
		width=width,
		height=height,
		background_color='white',
		mask=mask,
		scale=1,
		color_func=custom_color_func,
		font_path=config["fort_path"],
		prefer_horizontal=1
	).generate(text)

	plt.figure(figsize=(10, 5))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')

	img_byte_array = io.BytesIO()
	plt.savefig(img_byte_array, format='png')
	img_byte_array.seek(0)

	plt.close()

	return Response(content=img_byte_array.getvalue(), media_type="image/png")


if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=config['port'])
