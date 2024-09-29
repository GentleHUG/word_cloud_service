from pathlib import Path
from typing import Dict
import fasttext
from sentence_transformers import SentenceTransformer

# TODO: Надо поменять путь. 7 Гб мы никуда нормально не закинем, надо указать официальную ссылку на скачиание.
FASTTEXT_MODEL_PATH: Path = Path("/Users/fffgson/Desktop/Coding/nuclearhack2024_local")
TEST_WORDS_PATH: Path = Path("/Users/fffgson/Desktop/Coding/nuclearhack2024_local")
RUBERT_MODEL_PATH: str = "cointegrated/rubert-tiny2"

MODELS_DICT: Dict[str,] = {
    "fasttext" : fasttext.load_model('cc.ru.300.bin'),
    "rubert": SentenceTransformer('cointegrated/rubert-tiny2'),
}