"""Очистка данных
1. Проверка орфографии
2. Делим на рус/англ
3. Лемманизация
5. Очистка bad words
6. (если нужен) перевод
7. Соединяем
8. Заменяем синонимы
"""

#исправляем орфографические ошибки и на русском, и на английском языках
from pyaspeller import YandexSpeller
def fixed_grammar(word):
  speller = YandexSpeller()
  return speller.spelled(word)
