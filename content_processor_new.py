from typing import Union, List, Dict
import numpy as np
from embeddings.model import WordClusterizer    
from pre_process.data_cleaner_new import TextProcessor
from embeddings.topwords_type import TopCluster
from embeddings.summarizer import Summarizer
import logging

class ContentProcessor:
    def __init__(self, ru_words_path: str, en_words_path: str, gigachat_token: str):
        logging.info("Initializing WordClustreizer() class.")
        self.process_model = WordClusterizer()
        self.preprocess_model = TextProcessor(ru_words_path, en_words_path)
        self.summarizer = Summarizer(gigachat_token)

    def preprocess(self, input: np.ndarray, enable_trans: bool, enable_grammar: bool) -> np.ndarray:
        logging.info("Preprocessing data.")
        result = self.preprocess_model.forward(input, enable_trans, enable_grammar)
        return result

    def process(self, input: np.ndarray, num_top_words: Union[int, str] = "auto") -> List[TopCluster]:
        logging.info("Processing data")
        result = self.process_model.forward(input, num_top_words)
        return result

    def summarize(self, clusters: List[TopCluster]) -> Dict[str, float]:
        return {self.summarizer.summarize(cluster.cluster_content): cluster.cluster_weight for cluster in clusters}