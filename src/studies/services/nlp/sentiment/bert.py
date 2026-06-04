### BINARY BERT
### A.K.A. DISTILBERT

from transformers import pipeline

from studies.models import SentimentCategory
from studies.services.nlp.contracts import BaseSentimentAnalyzer, SentimentResult
from studies.services.nlp.sentiment._thresholds import map_sentiment_score_to_label

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

class BertSentimentAnalyzer(BaseSentimentAnalyzer):
    method_name = "bert"

    def __init__(self):
        self.analyzer = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
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
                },
            )

        result = self.analyzer(cleaned_text, truncation=True)[0]

        raw_label = result["label"]
        confidence = float(result["score"])

        if raw_label == "POSITIVE":
            sentiment_score = confidence
        elif raw_label == "NEGATIVE":
            sentiment_score = -confidence
        else:
            sentiment_score = 0.0

        label = map_sentiment_score_to_label(sentiment_score)

        return SentimentResult(
            score=sentiment_score,
            label=label,
            method=self.method_name,
            metadata={
                "model": MODEL_NAME,
                "raw_label": raw_label,
                "confidence": confidence,
            },
        )