from dataclasses import dataclass
from typing import Union, Dict, List
import numpy as np


@dataclass
class Topwords:
    """
    Класс для хранения информации о словах, их весах и косинусных значениях.

    Атрибуты:
        words_list (Union[list, np.ndarray]):
            Список или массив, содержащий слова.
            Может быть представлен как стандартный список Python или как массив NumPy.

        weights (Dict[int, float]):
            Словарь, где ключи представляют индексы слов, а значения — их соответствующие веса.
            Используется для хранения значимости каждого слова.

        cosines (np.ndarray):
            Массив NumPy, содержащий косинусные значения, связанные со словами в `words_list`.
            Эти значения могут использоваться для оценки схожести между словами или векторов.

    Пример использования:
        topwords_instance = Topwords(
            words_list=['apple', 'banana', 'cherry'],
            weights={0: 0.5, 1: 0.3, 2: 0.2},
            cosines=np.array([0.1, 0.2, 0.3])
        )
    """

    words_list: Union[list, np.ndarray]
    weights: Dict[int, float]
    cosines: np.ndarray

# TODO: доделать тип данных
@dataclass
class TopCluster:
    cluster_id: int
    cluster_weight: int
    cluster_content: List[str]
    words_cosines: np.ndarray