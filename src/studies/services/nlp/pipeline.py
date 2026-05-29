# This file defines the main NLP pipeline for analyzing study entries
# Steps:
# 1. Receive a study_id and list/query for entries to analyze
# 2. For each entry, run sentiment analysis and theme extraction
# 3. Return one StudyAnalysisResult containing all the entry-level results

from collections import Counter

from studies.services.nlp.contracts import EntryAnalysisResult, StudyAnalysisResult, SentimentResult, ThemeResult
from studies.services.nlp.registry import get_sentiment_analyzer, get_theme_extractor
from studies.models import SentimentCategory, SENTIMENT_SCORE_THRESHOLDS

def analyze_study_entries(study_id: int, entries: list[dict], sentiment_method: str = "vader", theme_method: str = "tfidf_nmf") -> StudyAnalysisResult:
    # First, ensure the entries are in list form
    entry_list = list(entries)  

    # Get the sentiment analyzer and the theme extractor from the params
    sentiment_analyzer = get_sentiment_analyzer(sentiment_method)
    theme_extractor = get_theme_extractor(theme_method)

    documents = [entry.content or "" for entry in entry_list]

    themes, assignments = theme_extractor.extract(documents)

    themes_by_id = {
        theme.theme_id: theme
        for theme in themes
    }

    assignments_by_document_index = {
        assignment["document_index"]: assignment
        for assignment in assignments
    }

    entry_analysis_results = []

    for document_index, entry in enumerate(entry_list):
        entry_text = entry.content or ""

        sentiment_result = sentiment_analyzer.analyze(entry_text)

        assignment = assignments_by_document_index.get(document_index)
        theme_result = None
        theme_weight = None

        if assignment:
            theme_id = assignment["theme_id"]
            theme_result = themes_by_id.get(theme_id)
            theme_weight = assignment.get("theme_weight")

            if theme_result and theme_weight is not None:
                theme_result = ThemeResult(
                    theme_id = theme_result.theme_id,
                    weight = theme_weight,
                    label = theme_result.label,
                    keywords = theme_result.keywords,
                    method = theme_result.method,
                    metadata= {
                        **theme_result.metadata,
                    },
                )
        
        entry_analysis_result = EntryAnalysisResult(
            entry_id=entry.id,
            sentiment=sentiment_result,
            theme=theme_result,
            metadata={
                "word_count": len(entry_text.split()),
                "language": "en",
            }
        )

        entry_analysis_results.append(entry_analysis_result)
    
    return build_study_analysis_result(
        study_id=study_id,
        entry_analysis_results=entry_analysis_results,
        themes=themes,
        sentiment_method=sentiment_method,
        theme_method=theme_method,
    )

def build_study_analysis_result(study_id: int, entry_analysis_results: list[EntryAnalysisResult], themes: list, sentiment_method: str, theme_method: str) -> StudyAnalysisResult:
    
    sentiment_labels = [
        result.sentiment.label
        for result in entry_analysis_results
        if result.sentiment and result.sentiment.label
    ]

    sentiment_scores = [
        result.sentiment.score
        for result in entry_analysis_results
        if result.sentiment and result.sentiment.score is not None
    ]

    sentiment_distribution = dict(Counter(sentiment_labels))

    average_sentiment_score = None
    if sentiment_scores:
        average_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

    average_sentiment_label = None
    if average_sentiment_score is not None:
        for lower, upper, label in SENTIMENT_SCORE_THRESHOLDS:
            if lower <= average_sentiment_score < upper:
                average_sentiment_label = label
                break

        if average_sentiment_score == 1.0:
            average_sentiment_label = SentimentCategory.VERY_POSITIVE

        if average_sentiment_score == -1.0:
            average_sentiment_label = SentimentCategory.VERY_NEGATIVE
        

    dominant_sentiment_label = None
    if sentiment_distribution:
        dominant_sentiment_label = max(
                sentiment_distribution, 
                key=sentiment_distribution.get,
                )
        
    dominant_sentiment_score = None
    if dominant_sentiment_label:
        dominant_sentiment_scores = [
            result.sentiment.score
            for result in entry_analysis_results
            if (
                result.sentiment
                and result.sentiment.label == dominant_sentiment_label
                and result.sentiment.score is not None
            )
        ]

        if dominant_sentiment_scores:
            dominant_sentiment_score = (
                sum(dominant_sentiment_scores) / len(dominant_sentiment_scores)
            )

    theme_labels = [
        result.theme.label
        for result in entry_analysis_results
        if result.theme and result.theme.label
    ]

    theme_distribution = dict(Counter(theme_labels))  

    dominant_theme_label = None
    if theme_distribution:
        dominant_theme_label = max(
            theme_distribution, 
            key=theme_distribution.get,
            )
        
    dominant_theme_weights = []
    if dominant_theme_label:
        for result in entry_analysis_results:
            if result.theme and result.theme.label == dominant_theme_label:
                theme_weight = result.metadata.get("theme_weight")
                if theme_weight is not None:
                    dominant_theme_weights.append(theme_weight)

    dominant_theme_weight = None

    return StudyAnalysisResult(
        study_id=study_id,

        average_sentiment_label=average_sentiment_label,
        average_sentiment_score=average_sentiment_score,

        dominant_sentiment_label=dominant_sentiment_label,
        dominant_sentiment_score=dominant_sentiment_score,
        sentiment_distribution=sentiment_distribution,

        dominant_theme_label=dominant_theme_label,
        dominant_theme_weight=dominant_theme_weight,
        theme_distribution=theme_distribution,

        entry_analysis_results=entry_analysis_results,
        total_entries = len(entry_analysis_results),
        total_themes = len(themes),

        methods={
            "sentiment": sentiment_method,
            "theme": theme_method,
        },
        metadata={
            "pipeline": "v1",
        }
    )
