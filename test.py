import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def read_documents_from_text(raw_text: str, separator: str = "###"):
    docs = [doc.strip() for doc in raw_text.split(separator)]
    return [d for d in docs if d]

text_path = "lab_2/data/text.txt"

try:
    with open(text_path, "r") as f:
        text = f.read()
except FileNotFoundError as e:
    print("Файл не найден. Проверьте путь")
    raise e

documents = read_documents_from_text(text)
print(f"Количество документов: {len(documents)}")
print(documents[0])

# =============================
# 3. Очистка текста (твоя функция)
# =============================

class TextCleaner:
    def __init__(self):
        self._whitespace_re = re.compile(r'\s+')
        self._non_alnum_re = re.compile(r'[^a-zA-Z0-9 ]')
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        text = self._whitespace_re.sub(" ", text)
        text = self._non_alnum_re.sub(" ", text)
        text = text.lower()
        words = word_tokenize(text)
        words = [
            self.lemmatizer.lemmatize(word)
            for word in words if word not in self.stop_words
        ]
        return " ".join(words)

cleaner = TextCleaner()
cleaned_docs = [cleaner.clean_text(doc) for doc in documents]

# =============================
# 4. TF-IDF и LSA
# =============================

vectorizer = TfidfVectorizer(max_features=4000, max_df=0.9, min_df=1)
tfidf_matrix = vectorizer.fit_transform(cleaned_docs)

lsa = TruncatedSVD(n_components=6, random_state=42)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# =============================
# 5. Косинусное сходство
# =============================

similarity = cosine_similarity(lsa_matrix)

# =============================
# 6. Визуализация
# =============================

plt.figure(figsize=(10, 8))
sns.heatmap(similarity, annot=False, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Семантическая близость текстов (LSA + TF-IDF)", fontsize=14)
plt.show()
