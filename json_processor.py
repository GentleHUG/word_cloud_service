import json
from typing import Dict


def create_json_file(words_and_weights: Dict[str, float], filename: str) -> str:
	json_path = f'static/{filename}.json'
	with open(json_path, 'w') as json_file:
		json.dump(words_and_weights, json_file, ensure_ascii=False)
	return json_path
