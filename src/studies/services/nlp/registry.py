# This file defines the registry for NLP analysis methods in the context of studies.
# Given a method name like "vader" or "tfidf_nmf", the registry can be used to look up the corresponding function that performs the analysis.

from studies.services.nlp.sentiment.vader import VaderSentimentAnalyzer
from studies.services.nlp.sentiment.bert import BertSentimentAnalyzer
from studies.services.nlp.themes.tfidf_nmf import TfidfNmfThemeExtractor

SENTIMENT_ANALYZERS = {
    "vader": VaderSentimentAnalyzer,
    "bert": BertSentimentAnalyzer
    # Note to self: Add future sentiment analyzers here
}

THEME_EXTRACTORS = {
    "tfidf_nmf": TfidfNmfThemeExtractor,
    # Note to self: Add future theme analyzers here
}

# Note to self, how do we allow users to add their own custom analyzers? 
# Maybe I can add a function like `register_sentiment_analyzer(name: str, analyzer_class: Type[SentimentAnalyzer])` that adds to the SENTIMENT_ANALYZERS dict, and similarly for theme analyzers... but the user will need to know the correct analysizer classes.

def get_sentiment_analyzer(method: str):
    try:
        return SENTIMENT_ANALYZERS[method]()
    except KeyError:
        available_methods = ", ".join(SENTIMENT_ANALYZERS.keys())
        raise ValueError (
            f"Unknown sentiment analysis method '{method}'." 
            f"Available methods: {available_methods}"
        )

def get_theme_extractor(method: str):
    try:
        return THEME_EXTRACTORS[method]()
    except KeyError:
        available_methods = ", ".join(THEME_EXTRACTORS.keys())
        raise ValueError (
            f"Unknown theme extraction method '{method}'." 
            f"Available methods: {available_methods}"
        )