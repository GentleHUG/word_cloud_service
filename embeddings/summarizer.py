from gigachat import GigaChat

class Summarizer:
    def __init__(self, name):
        self.name = name
    def main_word_from_claster(self, claster_words, token):
    with GigaChat(credentials = token, verify_ssl_certs=False) as giga:
        response = giga.chat(f"найди 1 общее нейтральное слово для {','.join(set(claster_words))} и выведи только его")
        res = (response.choices[0].message.content).split(' ')[-1]
        return ''.join([i for i in res if i.isalpha()])


