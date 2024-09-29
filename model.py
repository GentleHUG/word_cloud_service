import numpy as np

""" Модель кластеризации:
input: list[str] - очищенные данные
output: dict{str : double} - словарь с весами слов,
dict{str: list[double]} - словарь с косинусными расстоянимяи до слов-'ценностей'
"""

#функция находит цвет на палитре градиента от одного цвета к другому со степенью alpha
def grad_color(color1, color2, alpha):
  r = int(color1[0] + (color2[0] - color1[0]) * alpha)
  g = int(color1[1] + (color2[1] - color1[1]) * alpha)
  b = int(color1[2] + (color2[2] - color1[2]) * alpha)
  return (r, g, b)

#подаем массив с косинусными расстояниями до всех слов-'ценностей', получаем массив цветов
def сosine_distances_to_color (cos_dist, color1, color2):
  try:
    array = [i.mean() for i in cos_dist]
  except:
    array = cos_dist
  norm_value = np.linalg.norm(array)
  alpha = array / norm_value
  return [grad_color (color1, color2, t) for t in alpha]
