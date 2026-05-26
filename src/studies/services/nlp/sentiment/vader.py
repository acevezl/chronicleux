from nltk.sentiment import SentimentIntensityAnalyzer
from studies.services.nlp.contracts import BaseSentimentAnalyzer, SentimentResult

class VaderSentimentAnalyzer(BaseSentimentAnalyzer):
    method_name = "vader"

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        scores = self.analyzer.polarity_scores(text or "")
        compound = scores["compound"]

        if compound <= -0.6:
            label = "VERY_NEGATIVE"
        elif compound < -0.05:
            label = "NEGATIVE"
        elif compound < 0.05:
            label = "NEUTRAL"
        elif compound < 0.6:
            label = "POSITIVE"
        else:            
            label = "VERY_POSITIVE"

        return SentimentResult(
            score=compound,
            label=label,
            method=self.method_name,
            metadata= {
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "positive": scores["pos"],
                "compound": compound,
            }
        )
