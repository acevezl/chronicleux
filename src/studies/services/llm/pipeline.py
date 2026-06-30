# This file defines the main LLM pipeline for analyzing study entries
# Steps:
# 1. Receive a study_id and list/query for entries to analyze
# 2. Explore study-level themes and issues using an LLM and all entries (1ST PASS)
# 3. Analyze each entry with an LLM to obtain sentiment, theme, and issue assignment (2ND PASS)
# 4. Return one StudyAnalysisResult containing all the entry-level results

############################
####### LLM PIPELINE #######
############################

from collections import Counter

from studies.services.contracts import (
    EntryAnalysisResult,
    IssueResult,
    StudyAnalysisResult,
    ThemeResult,
)
from studies.services.llm.entry_analyzer import analyze_entry_with_llm
from studies.services.llm.issue_explorer import explore_issues_with_llm
from studies.services.llm.theme_explorer import explore_themes_with_llm
from studies.services.nlp.sentiment._thresholds import map_sentiment_score_to_label


def analyze_study_entries_with_llm(
    study_id: int,
    entries,
    provider: str,
    model: str | None = None,
    max_themes: int = 8,
    max_issues: int = 8,
) -> StudyAnalysisResult:

    entry_list = list(entries)

    if not entry_list:
        raise RuntimeError("No diary entries available for LLM analysis.")

    # FIRST PASS: Build the study-level theme catalog
    # This returns canonical themes + LLM-suggested themes.
    themes = explore_themes_with_llm(
        entries=entry_list,
        provider=provider,
        model=model,
        max_themes=max_themes,
    )

    if not themes:
        raise RuntimeError("LLM theme exploration did not return any themes.")

    # FIRST PASS: Build the study-level issue catalog
    # This returns canonical issues + LLM-suggested issues.
    issues = explore_issues_with_llm(
        entries=entry_list,
        provider=provider,
        model=model,
        max_issues=max_issues,
    )

    # SECOND PASS: Analyze each entry using both catalogs.
    entry_analysis_results = []

    for entry in entry_list:
        entry_analysis_result = analyze_entry_with_llm(
            entry=entry,
            theme_catalog=themes,
            issue_catalog=issues,
            provider=provider,
            model=model,
        )

        entry_analysis_results.append(entry_analysis_result)


    return build_study_analysis_result(
        study_id=study_id,
        entry_analysis_results=entry_analysis_results,
        themes=themes,
        issues=issues,
        provider=provider,
        model=model,
    )


def build_study_analysis_result(
    study_id: int,
    entry_analysis_results: list[EntryAnalysisResult],
    themes: list[ThemeResult],
    issues: list[IssueResult],
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

    entry_themes = [
        theme
        for result in entry_analysis_results
        for theme in (result.themes or [])
        if theme and theme.theme_id is not None
    ]

    theme_distribution = dict(
        Counter(str(theme.theme_id) for theme in entry_themes)
    )

    dominant_theme = None
    if theme_distribution:
        dominant_theme_id = max(
            theme_distribution,
            key=theme_distribution.get,
        )

        matching_theme_results = [
            theme
            for theme in entry_themes
            if str(theme.theme_id) == dominant_theme_id
        ]

        if matching_theme_results:
            average_weight = None

            weights = [
                theme.weight
                for theme in matching_theme_results
                if theme.weight is not None
            ]

            if weights:
                average_weight = sum(weights) / len(weights)

            dominant_theme = matching_theme_results[0]
            dominant_theme.weight = average_weight

    entry_issues = [
        issue
        for result in entry_analysis_results
        for issue in (result.issues or [])
        if issue and issue.issue_id is not None
    ]

    issue_distribution = dict(
        Counter(str(issue.issue_id) for issue in entry_issues)
    )

    return StudyAnalysisResult(
        study_id=study_id,

        average_sentiment_label=average_sentiment_label,
        average_sentiment_score=average_sentiment_score,
        dominant_sentiment_label=dominant_sentiment_label,
        dominant_sentiment_score=dominant_sentiment_score,
        sentiment_distribution=sentiment_distribution,

        dominant_theme=dominant_theme,
        theme_distribution=theme_distribution,
        total_themes=len(entry_themes),

        issues=entry_issues,
        issue_distribution=issue_distribution,
        total_issues=len(entry_issues),

        entry_analysis_results=entry_analysis_results,
        total_entries=len(entry_analysis_results),

        methods={
            "sentiment": "llm",
            "theme": "llm",
            "issue": "llm",
            "provider": provider,
            "model": model,
        },
        metadata={
            "pipeline": "llm_v2",
            "provider": provider,
            "model": model,
            "theme_catalog_size": len(themes),
            "issue_catalog_size": len(issues),
        },
    )
