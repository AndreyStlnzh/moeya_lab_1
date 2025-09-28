import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

class TextCleaner:
    """
    Класс для предобработки текста новостей: удаление дубликатов, очистка текста,
    фильтрация данных, обработка дат.
    """
    # Компилируем регулярки заранее (ускоряет работу)
    _whitespace_re = re.compile(r"\s+")
    _non_alnum_re = re.compile(r"[^a-zA-Z0-9]")

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.data = None


    def clean_text(
        self, 
        text: str
    ) -> str:
        """
        Функция очистки текста
        Удаляет лишние пробелы и символы переноса строк и табуляции
        Оставляет только буквы и цифры

        Args:
            text (str): Исходный текст

        Returns:
            str: Очищенный текст
        """

        text = self._whitespace_re.sub(" ", text) # удаляем символы табуляции и переносов строк
        text = self._non_alnum_re.sub(" ", text)  # оставляем только буквы и цифры
        
        text = text.lower()
        words = word_tokenize(text)  # Токенизация
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]  # Лемматизация
        return " ".join(words)
    