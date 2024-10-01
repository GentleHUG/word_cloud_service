def read_file_as_set(file_path: str) -> set:
    """
    Reads a file and returns its contents as a set of unique words.

    Parameters:
        file_path (str): The path to the file to be read.

    Returns:
        set: A set containing unique words from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split()
        word_set = set(words)
    return word_set


def fixed_grammar(text: str, speller) -> str:
    """
    Corrects the grammar of the given text using a spelling checker.

    Parameters:
        text (str): The text to be checked for spelling errors.
        speller: An instance of a spelling checker.

    Returns:
        str: The corrected text.
    """
    return speller.spelled(text)


def translate(word: str, translator) -> str:
    """
    Translates a word from English to Russian.

    Parameters:
        word (str): The word to be translated.
        translator: An instance of a translator.

    Returns:
        str: The translated word in Russian.
    """
    return translator.translate(word, src='en', dest='ru').text


def clean_answer(text: str, ru_bwords: set, en_bwords: set, ru_morph, en_morph, translator, speller, morph, enable_trans: bool = False) -> str:
    """
    Cleans and processes the input text by correcting grammar, translating words, and filtering out bad words.

    Parameters:
        text (str): The input text to be cleaned.
        ru_bwords (set): A set of Russian bad words to filter out.
        en_bwords (set): A set of English bad words to filter out.
        ru_morph: A morphological analyzer for Russian.
        en_morph: A morphological analyzer for English.
        translator: An instance of a translator for language translation.
        speller: An instance of a spelling checker.
        morph: A morphological analyzer for Russian.
        enable_trans (bool): Flag to enable translation of English words to Russian.

    Returns:
        str: The cleaned and processed text, or None if the result is empty.
    """
    text = fixed_grammar(text, speller)
    
    en_text = re.sub(r'[^A-Za-z ]+', '', text).strip()

    if en_text:
        en_text = [token.lemma_.lower() for token in en_morph(en_text)]
        if enable_trans:
            en_text = [translate(item, translator) for item in en_text]
            en_text = [item for item in en_text if item not in ru_bwords]
        else:
          en_text = [item for item in en_text if item not in en_bwords]
    en_text = " ".join(en_text).strip()

    ru_text = re.sub(r'[^А-Яа-я ]+', '', text).split()
    if ru_text:
        for ind, item in enumerate(ru_text):
            p = morph.parse(item)[0]
            norma = morph.parse(p.normal_form)[0]
            if norma.word not in ru_bwords:
                pos, case, genger, number = p.tag.POS, p.tag.case, p.tag.gender, p.tag.number
                if (case, number) == ('gent', 'sing') and pos == 'NOUN':
                    ru_text[ind] = norma.inflect({'plur', 'nomn'}).word
                elif pos == 'NOUN':
                    if (number) == ('plur'):
                        ru_text[ind] = norma.inflect({'plur', 'nomn'}).word
                    else:
                        ru_text[ind] = norma.inflect({'sing', 'nomn'}).word
                elif pos == 'ADJF':
                    if ind + 1 < len(ru_text):
                        p_next = morph.parse(ru_text[ind+1])[0]
                        norma_next = morph.parse(p.normal_form)[0]
                        pos_next, case_next, genger_next, number_next = p_next.tag.POS, p_next.tag.case, p_next.tag.gender, p_next.tag.number
                        if pos_next == 'NOUN':
                            if number_next == 'plur':
                                ru_text[ind] = norma.inflect({number_next, 'nomn'}).word
                            else:
                                if (case_next, number_next) == ('gent', 'sing'):
                                    ru_text[ind] = norma.inflect({'plur', 'nomn'}).word
                                else:
                                    ru_text[ind] = norma.inflect({number_next, genger_next, 'nomn'}).word
                        else:
                            ru_text[ind] = norma.word
                    else:
                        ru_text[ind] = norma.word
                else:
                    ru_text[ind] = norma.word
            else:
                ru_text[ind] = ''

    ru_text = re.sub(r'\s+', ' ', " ".join(ru_text))
    print(ru_text)
    clean_answer = " ".join([ru_text, en_text]).strip()
    if clean_answer != '':
        return clean_answer

def preprocess(answers, enable_trans=False):

    ru_bwords = read_file_as_set('/content/ru_words.txt')  # Set of Russian words
    en_bwords = read_file_as_set('/content/en_words.txt')  # Set of English words
    ru_morph = Mystem()  # Morphological analyzer for Russian
    en_morph = spacy.load('en_core_web_sm')  # English NLP model
    translator = Translator()  # Translator for language translation
    speller = YandexSpeller()  # Spelling checker
    morph = pm.MorphAnalyzer(lang='ru')

    # results = Parallel(n_jobs=-1)(delayed(clean_answer)(answer, ru_bwords, en_bwords, ru_morph, en_morph, translator, speller, morph, enable_trans) for answer in np.array(answers))
    results = np.array([clean_answer(answer, ru_bwords, en_bwords, ru_morph, en_morph, translator, speller, morph, enable_trans) for answer in np.array(answers)])
    return results[results != np.array(None)]

