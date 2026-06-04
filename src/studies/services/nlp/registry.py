# This file defines the registry for NLP analysis methods in the context of studies.
# Given a method name like "vader" or "tfidf_nmf", the registry can be used to look up the corresponding function that performs the analysis.

from studies.services.nlp.sentiment.vader import VaderSentimentAnalyzer
from studies.services.nlp.sentiment.bert import BertSentimentAnalyzer
from studies.services.nlp.sentiment.roberta import RobertaSentimentAnalyzer
from studies.services.nlp.sentiment.fivestar import StarRatingSentimentAnalyzer

from studies.services.nlp.themes.tfidf_nmf import TfidfNmfThemeExtractor
from studies.services.nlp.themes.tfidf_lda import TfidfLdaThemeExtractor

SENTIMENT_ANALYZERS = {
    "vader": {
        "class": VaderSentimentAnalyzer,
        "label": "VADER",
        "description": "Lexicon-based sentiment analyzer. Fast, lightweight, and useful as a baseline.",
        "long_description": "Use VADER for fast baseline analysis when transparency matters. Best for clear sentiment cues, but weaker with sarcasm, mixed feelings, or context-heavy diary entries.",
    },
    "bert": {
        "class": BertSentimentAnalyzer,
        "label": "BERT",
        "description": "Binary transformer sentiment classifier. Returns positive or negative with confidence.",
        "long_description": "Use BERT for broad positive vs. negative classification with contextual language understanding. Best when neutral nuance is less important.",
    },
    "roberta": {
        "class": RobertaSentimentAnalyzer,
        "label": "RoBERTa",
        "description": "Transformer sentiment classifier with positive, neutral, and negative probabilities.",
        "long_description": "Use RoBERTa when neutral sentiment matters. Best for diary entries with moderate, factual, or mixed emotional tone. Better suited than BERT when neutral sentiment matters.",
    },
    "star_rating": {
        "class": StarRatingSentimentAnalyzer,
        "label": "Star Rating",
        "description": "Five-class transformer model that maps text into 1-to-5 star sentiment ratings (where 1-star = Very Negative, and 5-star = Very Positive).",
        "long_description": "Use Star Rating when evaluators need a simple five-level sentiment scale. Best for quick comparison across entries, not deep interpretation.",
    },
}

THEME_EXTRACTORS = {
    "tfidf_nmf": {
        "class": TfidfNmfThemeExtractor,
        "label": "TF-IDF + NMF",
        "description": "Extracts recurring themes using TF-IDF features and non-negative matrix factorization.",
        "long_description": "Use TF-IDF + NMF as a fast, interpretable baseline for recurring vocabulary patterns. Best when participants describe similar experiences with similar words.",
    },
    "tfidf_lda": {
        "class": TfidfLdaThemeExtractor,
        "label": "TF-IDF + LDA",
        "description": "Extracts recurring themes using TF-IDF features and latent Dirichlet allocation.",
        "long_description": "Use as TF-IDF + LDA a probabilistic topic-modeling comparison. Best for larger entry sets with enough repeated vocabulary.",
    },
}

# Note to self, how do we allow users to add their own custom analyzers? 
# Maybe I can add a function like `register_sentiment_analyzer(name: str, analyzer_class: Type[SentimentAnalyzer])` that adds to the SENTIMENT_ANALYZERS dict, and similarly for theme analyzers... but the user will need to know the correct analysizer classes.

def get_sentiment_analyzer(method: str):
    try:
        return SENTIMENT_ANALYZERS[method]["class"]()
    except KeyError:
        available_methods = ", ".join(SENTIMENT_ANALYZERS.keys())
        raise ValueError(
            f"Unknown sentiment analysis method '{method}'. "
            f"Available methods: {available_methods}"
        )


def get_theme_extractor(method: str):
    try:
        return THEME_EXTRACTORS[method]["class"]()
    except KeyError:
        available_methods = ", ".join(THEME_EXTRACTORS.keys())
        raise ValueError(
            f"Unknown theme extraction method '{method}'. "
            f"Available methods: {available_methods}"
        )


def get_available_sentiment_methods():
    return [
        {
            "value": method,
            "label": config["label"],
            "description": config["description"],
            "long_description": config["long_description"],
        }
        for method, config in SENTIMENT_ANALYZERS.items()
    ]


def get_available_theme_methods():
    return [
        {
            "value": method,
            "label": config["label"],
            "description": config["description"],
            "long_description": config["long_description"],
        }
        for method, config in THEME_EXTRACTORS.items()
    ]


def get_available_sentiment_method_values():
    return list(SENTIMENT_ANALYZERS.keys())


def get_available_theme_method_values():
    return list(THEME_EXTRACTORS.keys())