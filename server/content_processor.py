import json
import math
from collections import Counter
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from wordcloud import WordCloud
import CONFIG
from embeddings.model import get_top_words
from embeddings.topwords_type import Topwords


def create_egg_mask(width, height, egg_size):
	mask = Image.new("L", (width, height), 'white')
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

def generate_word_cloud(word_counts: Counter, filename: str):
	"""Generates a word cloud image from word counts."""
	mask = create_egg_mask(CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT, CONFIG.EGG_SIZE)

	wordcloud = WordCloud(
		width=CONFIG.IMAGE_WIDTH,
	    height=CONFIG.IMAGE_HEIGHT,
		background_color='#f4f4f4',
		mask=mask,
		# scale=1,
		color_func=lambda *args, **kwargs: "#ff0032",
		# font_path=CONFIG.FORT_PATH
		prefer_horizontal=1
	).generate_from_frequencies(word_counts)

	# Сохраняем изображение
	image_path = f'static/{filename}.png'
	wordcloud.to_file(image_path)
	return image_path
def process_content(strings: List[str], filename: str) -> (str, str):
	word_list = ' '.join(strings).split()
	word_counts = Counter(word_list)

	np_strs = np.array(' '.join(strings).split())
	top_words = get_top_words(np_strs)
	print(top_words)
	word_counts = Counter(top_words.words_list)

	image_path = generate_word_cloud(word_counts, filename)
	print(word_counts.most_common())
	json_data = json.dumps([(word, freq) for word, freq in word_counts.most_common()], ensure_ascii=False)

	return image_path, json_data


