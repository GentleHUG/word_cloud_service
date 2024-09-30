from dataclasses import dataclass
from typing import Union, Dict
import numpy as np


@dataclass
class Topwords:
    words_list: Union[list, np.ndarray]
    weights: Dict[int, float]
    cosines: np.ndarray
