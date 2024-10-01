# TODO: Скопируй все библы
class TextProcessor:
    def __init__(
        self, ru_words_path: str, en_words_path: str, enable_trans: bool = False
    ):
        self.ru_bwords = self.read_file_as_set(
            ru_words_path
        )  # Set of Russian bad words
        self.en_bwords = self.read_file_as_set(
            en_words_path
        )  # Set of English bad words
        self.ru_morph = Mystem()  # Morphological analyzer for Russian
        self.en_morph = spacy.load("en_core_web_sm")  # English NLP model
        self.translator = Translator()  # Translator for language translation
        self.speller = YandexSpeller()  # Spelling checker
        self.morph = pm.MorphAnalyzer(lang="ru")
        self.enable_trans = enable_trans

    def read_file_as_set(self, file_path: str) -> set:
        """
        Reads a file and returns its contents as a set of unique words.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            words = text.split()
            word_set = set(words)
        return word_set

    def fixed_grammar(self, text: str) -> str:
        """
        Corrects the grammar of the given text using a spelling checker.
        """
        return self.speller.spelled(text)

    def translate(self, word: str) -> str:
        """
        Translates a word from English to Russian.
        """
        return self.translator.translate(word, src="en", dest="ru").text

    def clean_answer(self, text: str) -> str:
        """
        Cleans and processes the input text by correcting grammar, translating words, and filtering out bad words.
        """
        text = self.fixed_grammar(text)

        en_text = re.sub(r"[^A-Za-z ]+", "", text).strip()
        if en_text:
            en_text = [token.lemma_.lower() for token in self.en_morph(en_text)]
            if self.enable_trans:
                en_text = [self.translate(item) for item in en_text]
                en_text = [item for item in en_text if item not in self.ru_bwords]
            else:
                en_text = [item for item in en_text if item not in self.en_bwords]
        en_text = " ".join(en_text).strip()

        ru_text = re.sub(r"[^А-Яа-я ]+", "", text).split()
        if ru_text:
            for ind, item in enumerate(ru_text):
                p = self.morph.parse(item)[0]
                norma = self.morph.parse(p.normal_form)[0]
                if norma.word not in self.ru_bwords:
                    pos, case, gender, number = (
                        p.tag.POS,
                        p.tag.case,
                        p.tag.gender,
                        p.tag.number,
                    )
                    if (case, number) == ("gent", "sing") and pos == "NOUN":
                        try:
                            ru_text[ind] = norma.inflect({"plur", "nomn"}).word
                        except AttributeError:
                            ru_text[ind] = item
                    elif pos == "NOUN":
                        if number == "plur":
                            ru_text[ind] = norma.inflect({"plur", "nomn"}).word
                        else:
                            ru_text[ind] = norma.inflect({"sing", "nomn"}).word
                    elif pos == "ADJF":
                        if ind + 1 < len(ru_text):
                            p_next = self.morph.parse(ru_text[ind + 1])[0]
                            norma_next = self.morph.parse(p.normal_form)[0]
                            pos_next, case_next, gender_next, number_next = (
                                p_next.tag.POS,
                                p_next.tag.case,
                                p_next.tag.gender,
                                p_next.tag.number,
                            )
                            if pos_next == "NOUN":
                                if number_next == "plur":
                                    ru_text[ind] = norma.inflect(
                                        {number_next, "nomn"}
                                    ).word
                                else:
                                    if (case_next, number_next) == ("gent", "sing"):
                                        ru_text[ind] = norma.inflect(
                                            {"plur", "nomn"}
                                        ).word
                                    else:
                                        ru_text[ind] = norma.inflect(
                                            {number_next, gender_next, "nomn"}
                                        ).word
                            else:
                                ru_text[ind] = norma.word
                        else:
                            ru_text[ind] = norma.word
                    else:
                        ru_text[ind] = norma.word
                else:
                    ru_text[ind] = ""

        ru_text = re.sub(r"\s+", " ", " ".join(ru_text))
        clean_answer = " ".join([ru_text, en_text]).strip()
        if clean_answer != "":
            return clean_answer
        return None

    def forward(self, answers) -> np.ndarray:
        """
        Processes a list of answers by cleaning and filtering them.
        """
        results = np.array([self.clean_answer(answer) for answer in np.array(answers)])
        results = results[results != np.array(None)]  # Filter out None results
        return results
