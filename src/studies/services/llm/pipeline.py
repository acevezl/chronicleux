# This file defines the main LLM pipeline for analyzing study entries
# Steps:
# 1. Receive a study_id and list/query for entries to analyze
# 2. Explore study-level themes using an LLM and all entries (1ST PASS)
# 3. Analyze each entry with an LLM to obtain sentiment and theme assignment (2ND PASS)
# 4. Return one StudyAnalysisResult containing all the entry-level results

############################
####### LLM PIPELINE #######
############################

from collections import Counter

from studies.services.contracts import EntryAnalysisResult, StudyAnalysisResult, ThemeResult
from studies.services.llm.entry_analyzer import analyze_entry_with_llm
from studies.services.llm.theme_explorer import explore_themes_with_llm
from studies.services.nlp.sentiment._thresholds import map_sentiment_score_to_label


def analyze_study_entries_with_llm(
    study_id: int,
    entries,
    provider: str,
    model: str | None = None,
    max_themes: int = 8,
) -> StudyAnalysisResult:

    entry_list = list(entries)

    if not entry_list:
        raise RuntimeError("No diary entries available for LLM analysis.")

    # FIRST PASS: Explore shared study-level themes
    themes = explore_themes_with_llm(
        entries=entry_list,
        provider=provider,
        model=model,
        max_themes=max_themes,
    )

    if not themes:
        raise RuntimeError("LLM theme exploration did not return any themes.")

    # SECOND PASS: Analyze each entry using the discovered theme catalog
    entry_analysis_results = []

    for entry in entry_list:
        entry_analysis_result = analyze_entry_with_llm(
            entry=entry,
            theme_catalog=themes,
            provider=provider,
            model=model,
        )

        entry_analysis_results.append(entry_analysis_result)

    return build_study_analysis_result(
        study_id=study_id,
        entry_analysis_results=entry_analysis_results,
        themes=themes,
        provider=provider,
        model=model,
    )


def build_study_analysis_result(
    study_id: int,
    entry_analysis_results: list[EntryAnalysisResult],
    themes: list[ThemeResult],
    provider: str,
    model: str | None = None,
) -> StudyAnalysisResult:

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
        average_sentiment_label = map_sentiment_score_to_label(
            average_sentiment_score
        )

    dominant_sentiment_label = None
    if sentiment_distribution:
        dominant_sentiment_label = max(
            sentiment_distribution,
            key=sentiment_distribution.get,
        )

    dominant_sentiment_score = None
    dominant_sentiment_scores = []

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

    dominant_theme_weight = None
    dominant_theme_weights = []

    if dominant_theme_label:
        dominant_theme_weights = [
            result.theme.weight
            for result in entry_analysis_results
            if (
                result.theme
                and result.theme.label == dominant_theme_label
                and result.theme.weight is not None
            )
        ]

        if dominant_theme_weights:
            dominant_theme_weight = (
                sum(dominant_theme_weights) / len(dominant_theme_weights)
            )

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
        total_entries=len(entry_analysis_results),
        total_themes=len(themes),

        methods={
            "sentiment": "llm",
            "theme": "llm",
            "provider": provider,
            "model": model,
        },
        metadata={
            "pipeline": "llm_v1",
            "provider": provider,
            "model": model,
        },
    )