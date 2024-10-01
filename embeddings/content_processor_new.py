# TODO: Добписать импорты

class ContentProcessor:
    # TODO: дописать сюда в инит Леша тебе импорты которые нужны твоему классу вместо 3 точек
    def __init__(self, ...):
        self.preprocess_model = WordClusterizer()
        self.process_model = TextProcessor(...)

    def preprocess(self, input: np.ndarray):
        result = self.preprocess_model.forward()
        return result

    def process(self, input: np.ndarray, num_top_words: Union[int, str] = "auto"):
        result = self.process_model.forward(input)
        return result