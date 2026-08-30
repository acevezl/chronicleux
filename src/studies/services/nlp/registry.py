# This file defines the registry for NLP analysis methods in the context of studies.
# Given a method name like "vader" or "tfidf_nmf", the registry can be used to look up the corresponding function that performs the analysis.

# Sentiment Analyzers
from studies.services.nlp.sentiment.vader import VaderSentimentAnalyzer
from studies.services.nlp.sentiment.bert import BertSentimentAnalyzer
from studies.services.nlp.sentiment.roberta import RobertaSentimentAnalyzer
from studies.services.nlp.sentiment.fivestar import StarRatingSentimentAnalyzer

# Theme Extractors
from studies.services.nlp.themes.tfidf_nmf import TfidfNmfThemeExtractor
from studies.services.nlp.themes.tfidf_lda import CountLdaThemeExtractor
from studies.services.nlp.themes.bertopic import BertopicThemeExtractor

# Issue Detectors
from studies.services.nlp.issues.tfidf import TfidfIssueDetector

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
        "description": "Binary transformer sentiment classifier. Returns negative vs. non-negative with confidence.",
        "long_description": "Use BERT for broad negative vs. non-negative classification with contextual language understanding. Best when neutral nuance is less important.",
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
        "description": "Five-class BERT transformer model that maps text into 1-to-5 star sentiment ratings (where 1-star = Very Negative, and 5-star = Very Positive).",
        "long_description": "Use Star Rating when evaluators need a simple five-level sentiment scale. Best for quick comparison across entries, not deep interpretation.",
    },
}

THEME_EXTRACTORS = {
    "tfidf_nmf": {
        "class": TfidfNmfThemeExtractor,
        "label": "TF-IDF + NMF",
        "description": "Matches entries against the canonical theme catalog, then uses NMF to suggest missing themes when catalog matches are weak.",
        "long_description": "Use TF-IDF + NMF as a topic-modeling comparison. The analyzer first maps diary entries to approved canonical themes, then runs NMF only on weakly matched entries to suggest possible new catalog themes.",
    },
    "tfidf_lda": {
        "class": CountLdaThemeExtractor,
        "label": "Count Vectorization + LDA",
        "description": "Uses LDA to discover themes, matches them against the canonical theme catalog, and suggests new themes when matches are weak.",
        "long_description": "Use count vectorization and Latent Dirichlet Allocation as a probabilistic topic-modeling approach. The analyzer discovers topics across the diary entries, compares each topic and its supporting entries against approved canonical themes using TF-IDF similarity, and creates suggested catalog themes only when the available evidence does not support a reliable canonical match.",
    },
    "bertopic": {
        "class": BertopicThemeExtractor,
        "label": "BERTopic",
        "description": "Discovers semantic topics with BERTopic, then resolves them against the canonical theme catalog.",
        "long_description": "Use BERTopic when participants describe similar experiences using different words. The analyzer first discovers semantic topics using transformer embeddings and c-TF-IDF, then matches each generated topic to the canonical theme catalog. If no strong catalog match is found, it creates a suggested canonical theme for evaluator review.",
    },
}

ISSUE_DETECTORS = {
    "tfidf": {
        "class": TfidfIssueDetector,
        "label": "Keyword Match + TF-IDF",
        "description": "Detects UX issues by comparing diary entries against canonical issue definitions using keyword match and TF-IDF similarity.",
        "long_description": "Use Keyword Match + TF-IDF issue detection as a transparent NLP baseline. It works best when diary entries share vocabulary with issue names, aliases, descriptions, or examples.",
    }
}

# Analyzer / Extractor / Detector getters
def get_sentiment_analyzer(method: str):
    try:
        return SENTIMENT_ANALYZERS[method]["class"]()
    except KeyError:
        available_methods = ", ".join(SENTIMENT_ANALYZERS.keys())
        raise ValueError(
            f"Unknown sentiment-analysis method '{method}'. "
            f"Available methods: {available_methods}"
        )


def get_theme_extractor(method: str):
    try:
        return THEME_EXTRACTORS[method]["class"]()
    except KeyError:
        available_methods = ", ".join(THEME_EXTRACTORS.keys())
        raise ValueError(
            f"Unknown theme-extraction method '{method}'. "
            f"Available methods: {available_methods}"
        )


def get_issue_detector(method: str):
    try:
        return ISSUE_DETECTORS[method]["class"]()
    except KeyError:
        available_methods = ", ".join(ISSUE_DETECTORS.keys())
        raise ValueError(
            f"Unknown issue-detection method '{method}'. "
            f"Available methods: {available_methods}"
        )

# Method listers
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

def get_available_issue_methods():
    return [
        {
            "value": method,
            "label": config["label"],
            "description": config["description"],
            "long_description": config["long_description"],
        }
        for method, config in ISSUE_DETECTORS.items()
    ]


# Method values listers
def get_available_sentiment_method_values():
    return list(SENTIMENT_ANALYZERS.keys())


def get_available_theme_method_values():
    return list(THEME_EXTRACTORS.keys())


def get_available_issue_method_values():
    return list(ISSUE_DETECTORS.keys())