from transformers import pipeline

from studies.models import SentimentCategory
from studies.services.nlp.contracts import BaseSentimentAnalyzer, SentimentResult
from studies.services.nlp.sentiment._thresholds import map_sentiment_score_to_label

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class RobertaSentimentAnalyzer(BaseSentimentAnalyzer):
    method_name = "roberta"

    def __init__(self):
        self.analyzer = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            top_k=None,
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
                    "probabilities": {},
                },
            )

        raw_results = self.analyzer(cleaned_text, truncation=True)[0]

        probabilities = {}

        for item in raw_results:
            raw_label = item["label"].lower()
            score = float(item["score"])

            if raw_label in {"label_0", "negative"}:
                probabilities["negative"] = score
            elif raw_label in {"label_1", "neutral"}:
                probabilities["neutral"] = score
            elif raw_label in {"label_2", "positive"}:
                probabilities["positive"] = score

        negative_score = probabilities.get("negative", 0.0)
        neutral_score = probabilities.get("neutral", 0.0)
        positive_score = probabilities.get("positive", 0.0)

        sentiment_score = positive_score - negative_score
        label = map_sentiment_score_to_label(sentiment_score)

        dominant_label = max(
            probabilities,
            key=probabilities.get,
            default="neutral",
        )

        confidence = probabilities.get(dominant_label)

        return SentimentResult(
            score=sentiment_score,
            label=label,
            method=self.method_name,
            metadata={
                "model": MODEL_NAME,
                "raw_label": dominant_label,
                "confidence": confidence,
                "probabilities": probabilities,
                "positive_probability": positive_score,
                "neutral_probability": neutral_score,
                "negative_probability": negative_score,
            },
        )