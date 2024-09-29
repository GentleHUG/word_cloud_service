import numpy as np
from typing import List, Union
from umap import UMAP
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import fasttext
from collections import Counter
import os
import pandas as pd
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class WordClusterizer:
    """
    Класс для получения эмбеддингов слов, уменьшения размерности и кластеризации.
    """

    def __init__(self, model: str = "rubert", random_state: int = 42):
        """
        Инициализация класса.

        :param model: Название модели для получения эмбеддингов ('rubert' или 'fasttext').
        :param random_state: Случайное состояние для воспроизводимости.
        """
        assert model.lower().strip() in [
            "rubert",
            "fasttext",
        ], "Model should be 'rubert' or 'fasttext'."
        self.model_name = model.lower().strip()

        # Проверка наличия файла модели FastText
        if self.model_name == "fasttext":
            model_path = Path(os.path.abspath(__file__)) / "cc.ru.300.bin"
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"File '{model_path}' bot found in the project folder."
                )
            self.model = fasttext.load_model(model_path)
        else:
            self.model = SentenceTransformer("cointegrated/rubert-tiny2")

    def get_embeddings(self, word_list: np.ndarray) -> np.ndarray:
        """
        Получение эмбеддингов для списка слов.

        :param word_list: Numpy массив слов.
        :return: Numpy массив эмбеддингов.
        """
        assert len(word_list) > 0, "Word list should have at least 1 word."
        if self.model_name == "rubert":
            return self.model.encode(word_list)
        else:
            return np.array([self.model.get_word_vector(word) for word in word_list])

    def cluster_words(
        self,
        word_list: np.ndarray,
        num_clusters: Union[int, str] = "auto",
        num_components: Union[int, str] = "auto",
        random_state: int = 52,
    ) -> np.ndarray:
        """
        Кластеризация слов на основе их эмбеддингов.

        :param word_list: Numpy массив слов.
        :param num_clusters: Количество кластеров (int) или 'auto' для автоматического выбора.
        :return: Numpy массив слов из самых больших кластеров.
        """
        embeddings = self.get_embeddings(word_list)
        num_words = len(word_list)

        # Автоматический выбор количества кластеров и компонентов для UMAP
        if num_clusters == "auto":
            num_clusters = max(1, int(np.log2(num_words)))
        if num_components == "auto":
            num_components = max(1, int(np.log10(num_words)))

        # Уменьшение размерности с помощью UMAP
        umap_model = UMAP(n_components=num_components, random_state=random_state)
        reduced_embeddings = umap_model.fit_transform(embeddings)

        # Кластеризация с помощью KMeans
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=random_state)
        labels = kmeans_model.fit_predict(reduced_embeddings)

        # Объединение эмбеддингов, лейлов и слов
        together = np.concatenate(
            (labels.reshape(-1, 1), word_list.reshape(-1, 1)), axis=1
        )

        # Подсчет количества вхождений в каждый кластер
        cluster_counts = Counter(labels)
        top_clusters = cluster_counts.most_common(max(1, len(cluster_counts) // 2))

        # Получение слов из самых больших кластеров
        top_words = []
        for cluster_id, _ in top_clusters:
            top_words.extend(together[together[:, 0] == cluster_id][:, 1])

        return np.array(top_words)
    

def generate_word_cloud(words: Union[list, np.ndarray]):
    """
    Генерирует облако слов из списка слов и отображает его.

    :param words: Список слов для генерации облака слов.
    """
    # Объединяем список слов в одну строку
    text = ' '.join(words)

    # Создаем облако слов
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Отображаем облако слов
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Отключаем оси
    plt.show()


# Пример использования
if __name__ == "__main__":
    words = pd.read_csv(
        "/Users/fffgson/Desktop/Coding/nuclearhack2024_local/words.csv", sep=";"
    )["response"].to_numpy()
    clusterizer = WordClusterizer(model="rubert")
    top_words = clusterizer.cluster_words(words, num_clusters="auto")
    print("Слова из самых больших кластеров:", top_words)

    # generate_word_cloud(top_words)
    