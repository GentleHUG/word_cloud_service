import math
import random
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw
from wordcloud import WordCloud


color1 = (255, 0, 50)
color2 = (0, 0, 0)

class ImageProcessor:
	def _create_egg_mask(self, width: int, height: int, egg_size: int) -> np.ndarray:
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


	def __init__(self, width: int, height: int, egg_size: int=100, base_text_color: str="#ff0032", background_color: str="#f4f4f4"):
		self.width = width
		self.height = height
		self.base_text_color = base_text_color
		self.background_color = background_color
		self.mask = self._create_egg_mask(self.width, self.height, egg_size)

	# функция находит цвет на палитре градиента от одного цвета к другому со степенью alpha
	@staticmethod
	def grad_color(alpha: float, color1 = (255, 0, 50), color2 = (0, 0, 0)):
		alpha = np.random.rand()
		r = int(color1[0] + (color2[0] - color1[0]) * alpha)
		g = int(color1[1] + (color2[1] - color1[1]) * alpha)
		b = int(color1[2] + (color2[2] - color1[2]) * alpha)
		return "#{:02x}{:02x}{:02x}".format(r, g, b)

	#подаем массив с косинусными расстояниями до всех слов-'ценностей', получаем массив цветов

	def generate_word_cloud(self, words_and_weights: Dict[str, float], color_weights: Dict[str, float], filename: str) -> str:
		# norm = max(color_weights.values()) / min(color_weights.values())
		print(words_and_weights)
		print(color_weights)
		wordcloud = WordCloud(
			width=self.width,
			height=self.height,
			background_color=self.background_color,
			mask=self.mask,
			scale=1,
			color_func=lambda x, **kwargs: ImageProcessor.grad_color(color_weights[x]),
			prefer_horizontal=1
		).generate_from_frequencies(words_and_weights)

		image_path = f'static/{filename}.png'
		wordcloud.to_file(image_path)
		return image_path
