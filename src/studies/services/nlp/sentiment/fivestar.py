### 5-STAR 
### A.K.A. 5-STAR WITH BERT BASE MULTI-LINGUAL

from transformers import pipeline

from studies.models import SentimentCategory
from studies.services.nlp.contracts import BaseSentimentAnalyzer, SentimentResult


MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"


class StarRatingSentimentAnalyzer(BaseSentimentAnalyzer):
    method_name = "star_rating"

    def __init__(self):
        self.analyzer = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
        )

    def analyze(self, text: str) -> SentimentResult:
        cleaned_text = text or ""

        if not cleaned_text.strip():
            return SentimentResult(
                score=0.0,
                label=SentimentCategory.NEUTRAL,
                method=self.method_name,
                metadata={
                    "model": MODEL_NAME,
                    "raw_label": None,
                    "confidence": None,
                    "stars": None,
                },
            )

        result = self.analyzer(cleaned_text, truncation=True)[0]

        raw_label = result["label"]
        confidence = float(result["score"])

        stars = int(raw_label.split()[0])

        if stars == 1:
            sentiment_score = -1.0
            label = SentimentCategory.VERY_NEGATIVE
        elif stars == 2:
            sentiment_score = -0.5
            label = SentimentCategory.NEGATIVE
        elif stars == 3:
            sentiment_score = 0.0
            label = SentimentCategory.NEUTRAL
        elif stars == 4:
            sentiment_score = 0.5
            label = SentimentCategory.POSITIVE
        else:
            sentiment_score = 1.0
            label = SentimentCategory.VERY_POSITIVE

        return SentimentResult(
            score=sentiment_score,
            label=label,
            method=self.method_name,
            metadata={
                "model": MODEL_NAME,
                "raw_label": raw_label,
                "confidence": confidence,
                "stars": stars,
            },
        )