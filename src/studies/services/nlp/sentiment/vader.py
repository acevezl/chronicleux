from nltk.sentiment import SentimentIntensityAnalyzer
from studies.services.contracts import BaseSentimentAnalyzer, SentimentResult
from studies.services.nlp.sentiment._thresholds import map_sentiment_score_to_label


class VaderSentimentAnalyzer(BaseSentimentAnalyzer):
    method_name = "vader"

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        scores = self.analyzer.polarity_scores(text or "")
        compound = scores["compound"]

        label = map_sentiment_score_to_label(compound)

        return SentimentResult(
            score=compound,
            label=label,
            method=self.method_name,
            metadata={
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "positive": scores["pos"],
                "compound": compound,
            },
        )
