# This file defines the main NLP pipeline for analyzing study entries
# Steps:
# 1. Receive a study_id and list/query for entries to analyze
# 2. For each entry, run sentiment analysis and theme extraction
# 3. Return one StudyAnalysisResult containing all the entry-level results

from collections import Counter

from studies.services.nlp.contracts import EntryAnalysisResult, StudyAnalysisResult
from studies.services.nlp.registry import get_sentiment_analyzer, get_theme_extractor

def analyze_study_entries(study_id: int, entries: list[dict], sentiment_method: str = "vader", theme_method: str = "tfidf_nmf") -> StudyAnalysisResult:
    entry_list = list(entries)  # Ensure it's a list if it's a query result or other iterable

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
        sentiment_result = sentiment_analyzer.analyze(entry.content or "")

        assignment = assignments_by_document_index.get(document_index)
        theme_result = None
        theme_weight = None

        if assignment:
            theme_id = assignment["theme_id"]
            theme_result = themes_by_id.get(theme_id)
            theme_weight = assignment.get("theme_weight")
        
        entry_analysis_result = EntryAnalysisResult(
            entry_id=entry.id,
            sentiment=sentiment_result,
            theme=theme_result,
            metadata={
                "theme_weight": theme_weight,
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
    
    sentiment_scores = [
        result.sentiment.score
        for result in entry_analysis_results
        if result.sentiment and result.sentiment.score is not None
    ]

    sentiment_labels = [
        result.sentiment.label
        for result in entry_analysis_results
        if result.sentiment and result.sentiment.label
    ]

    study_sentiment_distribution = dict(Counter(sentiment_labels))

    dominant_study_sentiment_label = None
    if study_sentiment_distribution:
        dominant_study_sentiment_label = max(
                study_sentiment_distribution, 
                key=study_sentiment_distribution.get,
                )
        
    average_study_sentiment_score = None
    if sentiment_scores:
        average_study_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

    theme_labels = [
        result.theme.label
        for result in entry_analysis_results
        if result.theme and result.theme.label
    ]

    study_theme_distribution = dict(Counter(theme_labels))  

    dominant_study_theme_label = None
    if study_theme_distribution:
        dominant_study_theme_label = max(
            study_theme_distribution, 
            key=study_theme_distribution.get,
            )
        
    dominant_theme_weights = []
    if dominant_study_theme_label:
        for result in entry_analysis_results:
            if result.theme and result.theme.label == dominant_study_theme_label:
                theme_weight = result.metadata.get("theme_weight")
                if theme_weight is not None:
                    dominant_theme_weights.append(theme_weight)

    average_study_theme_weight = None

    if dominant_theme_weights:
        average_study_theme_weight = sum(dominant_theme_weights) / len(dominant_theme_weights)

    return StudyAnalysisResult(
        study_id=study_id,
        dominant_study_sentiment_label=dominant_study_sentiment_label,
        average_study_sentiment_score=average_study_sentiment_score,
        study_sentiment_distribution=study_sentiment_distribution,
        dominant_study_theme_label=dominant_study_theme_label,
        average_study_theme_weight=average_study_theme_weight,
        study_theme_distribution=study_theme_distribution,
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
