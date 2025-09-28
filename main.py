from src.sentiment_analysis import SentimentAnalysis
from src.text_cleaner import TextCleaner

text_cleaner: TextCleaner = TextCleaner()
sentiment_analysis: SentimentAnalysis = SentimentAnalysis()

text_path = "data/text.txt"

with open(text_path, "r") as f:
    text = f.read()


cleaned_text = text_cleaner.clean_text(text)
sentiment = sentiment_analysis.get_text_sentiment(cleaned_text)

print(f"\nРезультат: Тональность текста с файла {text_path} - {sentiment}")