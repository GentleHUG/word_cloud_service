import os
from typing import List

import numpy as np
from gigachat import GigaChat
import logging


class Summarizer:
    def __init__(self, token: str):
        self.token: str = token
        self.c = 0

    def summarize(self, clusters_and_words: List[List[str]]) -> List[str]:
        with GigaChat(
            credentials="ZjUzM2YyMzQtYTFiNS00M2MxLWFkOTYtNWFlY2E0NDljYTMyOjI0N2U1NDQ5LTcxMjQtNDkxNS04ZGI2LWI4OTgwYWY1NTkxYQ==",
            verify_ssl_certs=False
        ) as giga:
            return [
                (giga.chat(f"найди 1 общее нейтральное слово для {','.join(set(words))} и выведи только его").choices[0].message.content).split(' ')[-1]
                for words in clusters_and_words
            ]

