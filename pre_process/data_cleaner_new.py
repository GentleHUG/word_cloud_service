import spacy
import re
import pymorphy3 as pm
from pyaspeller import YandexSpeller
from googletrans import Translator
import numpy as np
import itertools
import logging


class TextProcessor:
	def __init__(
			self, ru_words_path: str, en_words_path: str
	):
		logging.info("Initializing data cleaner modules.")
		self.ru_bwords = self.read_file_as_set(
			ru_words_path
		)  # Set of Russian bad words
		self.en_bwords = self.read_file_as_set(
			en_words_path
		)  # Set of English bad words
		self.en_morph = spacy.load("en_core_web_sm")  # English NLP model
		self.translator = Translator()  # Translator for language translation
		self.speller = YandexSpeller()  # Spelling checker
		self.morph = pm.MorphAnalyzer(lang="ru")  # Morphological analyzer for Russian


	def read_file_as_set(self, file_path: str) -> set:
		"""
        Reads a file and returns its contents as a set of unique words.
        """
		with open(file_path, "r", encoding="utf-8") as file:
			text = file.read()
			words = text.split()
			word_set = set(words)
		return word_set


	def fixed_grammar(self, text: str) -> str:
		"""
        Corrects the grammar of the given text using a spelling checker.
        """
		return self.speller.spelled(text)


	def translate(self, word: str) -> str:
		"""
        Translates a word from English to Russian.
        """
		return self.translator.translate(word, src="en", dest="ru").text


	def clean_answer(self, text: str, enable_trans: bool) -> str:
		"""
        Cleans and processes the input text by correcting grammar, translating words, and filtering out bad words.
        """

		en_text = re.sub(r"[^A-Za-z ]+", "", text).strip()
		en_trans_text = []
		if en_text:
			en_text = [token.lemma_.lower() for token in self.en_morph(en_text)]
			if enable_trans:
				en_trans_text = [self.translate(item) for item in en_text]
				en_text = []
			else:
				en_text = [item for item in en_text if item not in self.en_bwords]
		else:
			en_text = []

		ru_text = (re.sub(r"[^А-Яа-я ]+", " ", text).split() + en_trans_text)[::-1]

		if ru_text:
			for ind, item in enumerate(ru_text):
				p = self.morph.parse(item)[0]
				if (item or p.normal_form) not in self.ru_bwords:
					if p.tag.POS == 'NOUN':
						case, number = (p.tag.case, p.tag.number)
						if (case, number) == ("gent", "sing") or number == "plur":
							if p.inflect({"plur", "nomn"}) is not None:
								ru_text[ind] = p.inflect({"plur", "nomn"}).word
							else:
								ru_text[ind] = p.normal_form
						else:
							ru_text[ind] = p.normal_form
					elif p.tag.POS == 'ADJF':
						if ind != 0 and self.morph.parse(ru_text[ind - 1])[0].tag.POS == 'NOUN':
							gender, number = (p.tag.gender, p.tag.number)
							if number == 'plur':
								ru_text[ind - 1] = p.inflect({"plur", "nomn"}).word + ' ' + ru_text[ind - 1]
							else:
								ru_text[ind - 1] = p.inflect({number, gender, "nomn"}).word + ' ' + ru_text[ind - 1]
							ru_text[ind] = ''
						else:
							ru_text[ind] = p.normal_form
					else:
						ru_text[ind] = p.normal_form
				else:
					ru_text[ind] = ""

		ru_text = [item for item in ru_text[::-1] if item]
		return en_text + ru_text


	def forward(self, answers: np.ndarray, enable_trans: bool = False) -> np.ndarray:
		"""
        Processes a list of answers by cleaning and filtering them.
        """
		logging.info("Fixing grammar in answers.")
		# fixed_grammar_answers = [self.fixed_grammar(answer) for answer in answers]

		logging.info("Getting normalized form of answers.")
		results = [self.clean_answer(answer, enable_trans) for answer in answers]
		# results = [self.clean_answer(answer) for answer in fixed_grammar_answers]

		return np.array(list(itertools.chain.from_iterable(results)))
