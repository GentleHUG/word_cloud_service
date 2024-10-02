# TODO: Добписать импорты
from typing import Union, List
import numpy as np
from embeddings.model import WordClusterizer    
from pre_process.data_cleaner_new import TextProcessor
from embeddings.topwords_type import TopClusters
import logging

class ContentProcessor:
    # TODO: дописать сюда в инит Леша тебе импорты которые нужны твоему классу вместо 3 точек
    def __init__(self, ru_words_path: str, en_words_path: str, enable_trans: bool = False):
        logging.info("Initializing WordClustreizer() class.")
        self.process_model = WordClusterizer()
        self.preprocess_model = TextProcessor(ru_words_path, en_words_path, enable_trans)

    def preprocess(self, input: np.ndarray) -> np.ndarray:
        logging.info("Preprocessing data.")
        result = self.preprocess_model.forward(input)
        return result

    def process(self, input: np.ndarray, num_top_words: Union[int, str] = "auto") -> List[TopClusters]:
        logging.info("Processing data")
        result = self.process_model.forward(input)
        return result