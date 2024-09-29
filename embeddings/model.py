from .constants import MODELS_DICT
import numpy as np
from typing import List
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Embedder(object):
    """
    Получи на вход список слов, прошедших модерацию и лемматизацию.
    Вычисли эмбеддинги слов.
    Вычисли косинусное расстояние между эмбеддингами и получи сходство.
    Вычисли кластеры векторов.
    Верни список из топ кластеров.

    Parameters:
    word_list (List[str]) - изначальный список слов.
    model (str) - какую модель использовать: 'fasttext' или 'rubert'. По умолчанию 'rubert'.

    Returns:
    word_cloud_list (List[str]) - итоговый список из наиболее релевантных слов.
    """

    def __init__(self, model: str = "rubert"):
        assert model.lower().strip() in [
            "rubert",
            "fasttext",
        ], "Assertion error: Model should be FastText or RuBERT"
        self.model_name = model.lower().strip()
        self.model = MODELS_DICT[model.lower().strip()]

    def forward(self, word_list: List[str]):
        assert (
            len(word_list) > 0
        ), "Assertion error: Word list should have at least 1 word."
        self.word_list = word_list
        self.length = len(word_list)
        return (
            self.model.encode(word_list)
            if self.model_name == "rubert"
            else np.array([self.model.get_word_vector(word) for word in word_list])
        )

    def clusterize(
        self,
        embds_list: np.ndarray,
        num_clusters: int = 5,
        num_components: int = 3,
        random_state: int = 42,
    ):
        self.umap_model = UMAP(n_components=num_components, random_state=random_state)
        reduced_embds = self.umap_model.fit_transform(embds_list)
        self.kmeans_model = KMeans(n_clusters=num_clusters, random_state=random_state)
        self.kmeans_model.fit(reduced_embds)
        self.labels = self.kmeans_model.labels_
        return np.concatenate(
            (
                reduced_embds,
                self.labels.reshape(-1, 1),
                self.word_list.reshape(-1, 1),
            ),
            axis=1,
        )

    def visualize3d(self, reduced_embds: np.ndarray):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            reduced_embds[:, 0],
            reduced_embds[:, 1],
            reduced_embds[:, 2],
            c=self.labels,
            cmap="viridis",
            alpha=0.6,
        )
        ax.set_title("UMAP Projection of Word Embeddings with Clusters (3D)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        cbar = plt.colorbar(scatter, ax=ax, label="Cluster Label")
        plt.show()

    def visualize2d(self, reduced_embds: np.ndarray):
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            reduced_embds[:, 0],
            reduced_embds[:, 1],
            c=self.labels,
            cmap="viridis",
            alpha=0.6,
        )
        ax.set_title("UMAP Projection of Word Embeddings with Clusters (2D)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        cbar = plt.colorbar(scatter, ax=ax, label="Cluster Label")
        plt.show()
