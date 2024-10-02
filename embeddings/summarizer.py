import numpy as np
from gigachat import GigaChat
import logging


class Summarizer:
    def __init__(self, token: str):
        self.token: str = token
        self.c = 0;

    def main_word_from_cluster(self, claster_words: np.ndarray) -> str:
        self.c += 1
        logging.log(self)
        with GigaChat(credentials=self.token, verify_ssl_certs=False) as giga:
            response = giga.chat(f"найди 1 общее нейтральное слово для {','.join(set(claster_words))} и выведи только его")
            res = (response.choices[0].message.content).split(' ')[-1]
            return ''.join([i for i in res if i.isalpha()])
