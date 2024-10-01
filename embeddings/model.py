import numpy as np
from typing import Union
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer
from collections import Counter
from embeddings.topwords_type import Topwords
import logging

mts_values = [
    "партнерство",
    "результативность",
    "ответственность",
    "смелость",
    "творчество",
    "открытость",
]


class WordClusterizer:
    """
    Класс для получения эмбеддингов слов, уменьшения размерности и кластеризации.

    Этот класс использует модель SentenceTransformer для получения эмбеддингов слов,
    затем применяет UMAP для уменьшения размерности и HDBSCAN для кластеризации.
    """

    def __init__(self, random_state: int = 52):
        """
        Инициализация класса WordClusterizer.

        :param random_state: Случайное состояние для воспроизводимости. По умолчанию 52.
        """
        self.model = SentenceTransformer("cointegrated/rubert-tiny2")

    def _get_embeddings(self, word_list: np.ndarray) -> np.ndarray:
        """
        Получение эмбеддингов для списка слов.

        :param word_list: Numpy массив слов.
        :return: Numpy массив эмбеддингов.
        :raises AssertionError: Если список слов пуст.
        """
        assert len(word_list) > 0, "Word list should have at least 1 word."
        return self.model.encode(word_list)

    def _cluster_words(self, word_list: np.ndarray) -> np.ndarray:
        """
        Кластеризация слов на основе их эмбеддингов.

        :param word_list: Numpy массив слов.
        :return: Numpy массив уменьшенных эмбеддингов слов.
        :raises AssertionError: Если количество слов меньше одного.
        """
        self.num_words = len(word_list)

        assert (
            self.num_words > 0
        ), "Assertion error: Number of words should be more than one."

        self.word_list = word_list

        # Получение эмбеддингов для слов
        self.embeddings = self._get_embeddings(self.word_list)

        # Уменьшение размерности с помощью UMAP
        self.umap_model = UMAP(n_components=max(2, int(np.log(self.num_words))))

        return self.umap_model.fit_transform(self.embeddings)

    def _get_top_words(
        self,
        reduced_embeddings: np.ndarray,
        num_top_words: Union[int, str] = "auto",
    ):
        """
        Получение топ-слов из кластеров на основе уменьшенных эмбеддингов.

        :param reduced_embeddings: Уменьшенные эмбеддинги слов.
        :param num_top_words: Количество топ-слов для извлечения (int) или 'auto' для автоматического выбора.
        :return: Кортеж из массива топ-слов и словаря весов кластеров.
        """
        # Определение минимального размера кластера
        if self.num_words < 7:
            self.min_cluster_size = self.num_words
        else:
            self.min_cluster_size = max(7, int(np.log(self.num_words)))

        # Кластеризация с помощью HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=self.min_cluster_size)
        self.labels = hdbscan_model.fit_predict(reduced_embeddings)

        # Объединение эмбеддингов, лейблов и слов
        together = np.concatenate(
            (self.labels.reshape(-1, 1), self.word_list.reshape(-1, 1)), axis=1
        )

        # Отбрасываем -1 (непринадлежащие кластерам)
        self.together = together[together[:, 0] != "-1"]

        # Подсчет количества вхождений в каждый кластер
        cluster_counts = Counter(together[:, 0])

        # Получение самых больших кластеров
        top_clusters = cluster_counts.most_common(max(1, len(cluster_counts) // 2))

        # Общее количество элементов в топ-кластерах
        total_elements = sum(count for _, count in top_clusters)

        # Получение слов из самых больших кластеров
        top_clusters_words = []
        weights = {}
        for cluster_id, count in top_clusters:
            top_clusters_words.extend(together[together[:, 0] == str(cluster_id)][:, 1])
            weights[int(cluster_id)] = (
                count / total_elements if total_elements > 0 else 0
            )

        word_counts = Counter(top_clusters_words)

        # Определение количества топ-слов
        if num_top_words == "auto":
            self.num_top_words = max(1, len(word_counts) // 2)
        else:
            if num_top_words > self.num_words:
                self.num_top_words = self.num_words
            else:
                self.num_top_words = num_top_words

        # Получение самых частых слов
        _top_words = word_counts.most_common(max(1, self.num_top_words))

        top_words = []
        for word, _ in _top_words:
            top_words.append(word)

        self.weights = weights
        self.top_words = np.array(top_words)

        return np.array(top_words), weights

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Вычисление косинусного расстояния между двумя векторами.

        :param a: Первый вектор.
        :param b: Второй вектор.
        :return: Косинусное расстояние между векторами.
        """
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _get_cosines(self, top_words: np.ndarray) -> np.ndarray:
        """
        Получение средних косинусных расстояний между топ-словами и предопределенными значениями.

        :param top_words: Массив топ-слов.
        :return: Средние косинусные расстояния.
        """
        global mts_values

        # Получение эмбеддингов для топ-слов и предопределенных значений
        top_words_embds = self._get_embeddings(top_words)
        mts_values_embds = self._get_embeddings(mts_values)

        # Уменьшение размерности для топ-слов и предопределенных значений
        reduced_twe = self.umap_model.transform(top_words_embds)
        reduced_mve = self.umap_model.transform(mts_values_embds)

        # Список для хранения косинусных расстояний
        distances = []

        # Проходим по каждому вектору из reduced_mve
        for vec_mve in reduced_mve:
            # Проходим по каждому вектору из reduced_twe
            for vec_twe in reduced_twe:
                distance = self._cosine_distance(vec_mve, vec_twe)
                distances.append(distance)

        # Усредняем косинусные расстояния
        average_distance = np.mean(distances)
        return np.array(average_distance)

    def forward(
        self, words: np.ndarray, num_top_words: Union[int, str] = "auto"
    ) -> Topwords:
        """
        Основной метод для обработки списка слов, кластеризации и получения топ-слов.

        :param words: Numpy массив слов для обработки.
        :param num_top_words: Количество топ-слов для извлечения (int) или 'auto' для автоматического выбора.
        :return: Объект Topwords, содержащий список топ-слов, веса и косинусные расстояния.
        """
        # Кластеризация слов
        clustered_words = self._cluster_words(words)
        # Получение топ-слов и их весов
        top_words, weights = self._get_top_words(
            clustered_words, num_top_words=num_top_words
        )
        # Получение косинусных расстояний
        cosines = self._get_cosines(top_words)
        return Topwords(
            words_list=top_words.astype(np.str_),
            weights=weights,
            cosines=cosines,
        )
