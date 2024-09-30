import numpy as np
from typing import List, Union
from umap import UMAP
from sklearn.cluster import KMeans, HDBSCAN
from sentence_transformers import SentenceTransformer
import fasttext
from collections import Counter
import os
import pandas as pd
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

TOKENIZERS_PARALLELISM = False


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
        num_top_clusters: Union[int, str] = "auto",
        min_cluster_size: Union[int, str] = "auto",
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
        print(num_words)

        # Автоматический выбор количества кластеров и компонентов для UMAP
        # if num_clusters == "auto":
        #     num_clusters = max(2, int(np.log(num_words)))
        if num_components == "auto":
            num_components = max(2, int(np.log(num_words)))

        # TODO на подумать
        if min_cluster_size == "auto":
            min_cluster_size = 7
        else:
            min_cluster_size = min_cluster_size

        # Уменьшение размерности с помощью UMAP
        umap_model = UMAP(n_components=num_components, n_jobs=-1)
        reduced_embeddings = umap_model.fit_transform(embeddings)

        # Кластеризация с помощью KMeans
        # kmeans_model = KMeans(n_clusters=num_clusters)
        # labels = kmeans_model.fit_predict(reduced_embeddings)

        # Кластеризация с помощью HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size)
        labels = hdbscan_model.fit_predict(reduced_embeddings)        

        # Объединение эмбеддингов, лейлов и слов
        together = np.concatenate(
            (labels.reshape(-1, 1), word_list.reshape(-1, 1)), axis=1
        )
        # filter -1 labels
        together = together[together[:, 0] != '-1']
        
        # Подсчет количества вхождений в каждый кластер
        cluster_counts = Counter(together[:, 0])

        if num_top_clusters == "auto":
            top_clusters = cluster_counts.most_common(max(1, len(cluster_counts) // 2))
        else:
            top_clusters = cluster_counts.most_common(max(1, num_top_clusters))
        
        # Общее количество элементов в топ-кластерах
        total_elements = sum(count for _, count in top_clusters)

        # Получение слов из самых больших кластеров
        top_words = []
        weights = {}
        for cluster_id, count in top_clusters:
            top_words.extend(together[together[:, 0] == str(cluster_id)][:, 1])
            weights[cluster_id] = count / total_elements if total_elements > 0 else 0

        return np.array(top_words), weights


def generate_word_cloud(words: Union[list, np.ndarray]):
    """
    Генерирует облако слов из списка слов и отображает его.

    :param words: Список слов для генерации облака слов.
    """
    # Объединяем список слов в одну строку
    text = " ".join(words)

    # Создаем облако слов
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )

    # Отображаем облако слов
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # Отключаем оси
    plt.show()


# Пример использования
if __name__ == "__main__":
    # words = pd.read_csv(
    #     "/Users/fffgson/Desktop/Coding/nuclearhack2024_local/words.csv", sep=";"
    # )["response"].to_numpy()

    words = np.array(
        [
            "карьера",
            "офис",
            "коллега",
            "проект",
            "задача",
            "дедлайн",
            "встреча",
            "отчет",
            "команда",
            "навыки",
            "работодатель",
            "сотрудник",
            "должность",
            "обязанности",
            "профессионал",
            "развитие",
            "тренинг",
            "стратегия",
            "успех",
            "резюме",
            "интервью",
            "заработная плата",
            "бонус",
            "премия",
            "отпуск",
            "график",
            "дистанционная работа",
            "производительность",
            "мотивация",
            "лидерство",
            "коммуникация",
            "конференция",
            "собрание",
            "план",
            "анализ",
            "исследование",
            "клиент",
            "партнер",
            "сервис",
            "продвижение",
            "обучение",
            "опыт",
            "достижения",
            "цели",
            "стресс",
            "баланс",
            "возможности",
            "инновации",
            "технологии",
            "система",
            "управление",
            "ответственность",
            "планирование",
            "сроки",
            "производство",
            "согласование",
            "документация",
            "проверка",
            "обсуждение",
            "идеи",
            "конкуренция",
            "рынок",
            "стратегия",
            "разработка",
            "презентация",
            "анализ",
            "отзывы",
            "обратная связь",
            "партнерство",
            "сеть",
            "доступность",
            "профессионализм",
            "соблюдение",
            "стандарты",
            "производительность",
            "проектирование",
            "креативность",
            "инструменты",
            "ресурсы",
            "поддержка",
            "система",
            "план",
            "доклад",
            "проверка",
            "согласование",
            "поток",
            "производительность",
            "потенциал",
            "разнообразие",
            "возможности",
        ]
    )
    clusterizer = WordClusterizer(model="rubert")
    top_words, weights = clusterizer.cluster_words(words)
    print("Слова из самых больших кластеров:", top_words)
    print(weights)

    # generate_word_cloud(top_words)
