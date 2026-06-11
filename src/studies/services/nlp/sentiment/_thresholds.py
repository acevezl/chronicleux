from studies.models import SentimentCategory, SENTIMENT_SCORE_THRESHOLDS, BinarySentimentCategory, BINARY_SENTIMENT_SCORE_THRESHOLDS

def map_sentiment_score_to_label(score: float | None) -> str | None:
    if score is None:
        return None

    for lower, upper, label in SENTIMENT_SCORE_THRESHOLDS:
        if lower <= score < upper:
            return label

    if score == 1.0:
        return SentimentCategory.VERY_POSITIVE

    if score == -1.0:
        return SentimentCategory.VERY_NEGATIVE

    return SentimentCategory.NEUTRAL


def binary_map_sentiment_score_to_label(score: float | None) -> str | None:
    if score is None:
        return None

    for lower, upper, label in BINARY_SENTIMENT_SCORE_THRESHOLDS:
        if lower < score < upper:
            return label

    if score == 1.0:
        return BinarySentimentCategory.NOT_NEGATIVE

    if score == -1.0:
        return BinarySentimentCategory.NEGATIVE

    return SentimentCategory.NOT_NEGATIVE