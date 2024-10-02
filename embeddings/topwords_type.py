from dataclasses import dataclass
from typing import Union, Dict, List
import numpy as np

@dataclass
class TopCluster:
    cluster_id: int
    cluster_weight: int
    cluster_content: List[str]
    words_cosines: np.ndarray