import numpy as np
import spacy
# spacy download en_core_web_sm
import re
import pymorphy3 as pm
from pymystem3 import Mystem
# from ruwordnet import RuWordNet
# ruwordnet download
from pyaspeller import YandexSpeller
from googletrans import Translator

"""Очистка данных
1. Проверка орфографии
2. Делим на рус/англ
3. Лемманизация
5. Очистка bad words
6. (если нужен) перевод
7. Соединяем
8. Заменяем синонимы
"""

def fixed_grammar(word):
    speller = YandexSpeller()
    return speller.spelled(word)

def read_file_as_set(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split()
        word_set = set(words)
    return word_set

def translate(word):
    translator = Translator()
    translation = translator.translate(word, src='en', dest='ru')
    return translation.text


def preprocess(text, enable_trans=False):
    ru_bwords = read_file_as_set('ru_words.txt')
    en_bwords = read_file_as_set('en_words.txt')
    # wn = RuWordNet()
    # ru_morph = pm.MorphAnalyzer(lang='ru')
    ru_morph = Mystem()
    en_morph = spacy.load('en_core_web_sm')
    # make lemmas
    en_lemmas, ru_lemmas = [], []
    text = fixed_grammar(text)
    
    en_text = re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z ]+', ' ', text)).strip()
    if en_text:
        if enable_trans:
            en_lemmas = [translate(token.lemma_.lower()) for token in en_morph(en_text) if token.lemma_.lower() not in en_bwords]
        else:
            en_lemmas = [token.lemma_.lower() for token in en_morph(en_text) if token.lemma_.lower() not in en_bwords]
    
    ru_text = re.sub(r'[^А-Яа-я ]+', ' ', text)
    if ru_text:
        ru_lemmas = [lemma.strip() for lemma in ru_morph.lemmatize(ru_text) if lemma.strip() and lemma.strip() not in ru_bwords]

    return np.array(en_lemmas + ru_lemmas)
