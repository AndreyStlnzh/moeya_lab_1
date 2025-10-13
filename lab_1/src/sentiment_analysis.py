import nltk

from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')


class SentimentAnalysis:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()


    def get_text_sentiment(
        self,
        text: str
    ) -> str:
        """Функция определения тональности текста

        Args:
            text (str): исходный текст

        Returns:
            str: тональность (neg, neu, pos, compound)
        """
        
        sentiment_scores = self.sia.polarity_scores(text)
        return max(sentiment_scores, key=sentiment_scores.get)
    
