# This file defines the main NLP pipeline for analyzing study entries
# Steps:
# 1. Receive a study_id and list/query for entries to analyze
# 2. For each entry, run sentiment analysis, theme extraction, and issue detection
# 3. Return one StudyAnalysisResult containing all the entry-level results

############################
####### NLP PIPELINE #######
############################

from collections import Counter

from studies.services.contracts import (
    EntryAnalysisResult,
    StudyAnalysisResult,
    ThemeResult,
    IssueResult,
)

from studies.services.nlp.registry import (
    get_sentiment_analyzer,
    get_theme_extractor,
    get_issue_detector,
)

from studies.services.nlp.sentiment._thresholds import map_sentiment_score_to_label


def analyze_study_entries(
    study_id: int,
    entries: list[dict],
    sentiment_method: str = "vader",
    theme_method: str = "tfidf_nmf",
    issue_method: str = "tfidf",
) -> StudyAnalysisResult:
    entry_list = list(entries)

    sentiment_analyzer = get_sentiment_analyzer(sentiment_method)
    theme_extractor = get_theme_extractor(theme_method)
    issue_detector = get_issue_detector(issue_method)

    documents = [entry.content or "" for entry in entry_list]

    themes, theme_assignments = theme_extractor.extract(documents)
    issues, issue_assignments = issue_detector.detect(documents)

    themes_by_id = {
        theme.theme_id: theme
        for theme in themes
    }

    issues_by_id = {
        issue.issue_id: issue
        for issue in issues
    }

    theme_assignments_by_document_index = {
        assignment["document_index"]: assignment
        for assignment in theme_assignments
    }

    issue_assignments_by_document_index = {
        assignment["document_index"]: assignment
        for assignment in issue_assignments
    }

    entry_analysis_results = []

    for document_index, entry in enumerate(entry_list):
        entry_text = entry.content or ""

        sentiment_result = sentiment_analyzer.analyze(entry_text)

        theme_assignment = theme_assignments_by_document_index.get(document_index)
        theme_result = None

        if theme_assignment:
            theme_id = theme_assignment["theme_id"]
            base_theme_result = themes_by_id.get(theme_id)
            theme_weight = theme_assignment.get("theme_weight")

            if base_theme_result:
                theme_result = ThemeResult(
                    theme_id=base_theme_result.theme_id,
                    weight=(
                        theme_weight
                        if theme_weight is not None
                        else base_theme_result.weight
                    ),
                    label=base_theme_result.label,
                    keywords=base_theme_result.keywords,
                    method=base_theme_result.method,
                    metadata={
                        **base_theme_result.metadata,
                        "assignment": {
                            key: value
                            for key, value in theme_assignment.items()
                            if key not in {"document_index", "theme_id"}
                        },
                    },
                )

        issue_assignment = issue_assignments_by_document_index.get(document_index)
        issue_result = None

        if issue_assignment:
            issue_id = issue_assignment["issue_id"]
            base_issue_result = issues_by_id.get(issue_id)
            issue_weight = issue_assignment.get("issue_weight")

            if base_issue_result:
                issue_result = IssueResult(
                    issue_id=base_issue_result.issue_id,
                    weight=(
                        issue_weight
                        if issue_weight is not None
                        else base_issue_result.weight
                    ),
                    label=base_issue_result.label,
                    keywords=base_issue_result.keywords,
                    method=base_issue_result.method,
                    metadata={
                        **base_issue_result.metadata,
                        "assignment": {
                            key: value
                            for key, value in issue_assignment.items()
                            if key not in {"document_index", "issue_id"}
                        },
                    },
                )

        entry_analysis_result = EntryAnalysisResult(
            entry_id=entry.id,
            sentiment=sentiment_result,
            theme=theme_result,
            issue=issue_result,
            metadata={
                "word_count": len(entry_text.split()),
                "language": "en",
            },
        )

        entry_analysis_results.append(entry_analysis_result)

    return build_study_analysis_result(
        study_id=study_id,
        entry_analysis_results=entry_analysis_results,
        themes=themes,
        issues=issues,
        sentiment_method=sentiment_method,
        theme_method=theme_method,
        issue_method=issue_method,
    )


def build_study_analysis_result(
    study_id: int,
    entry_analysis_results: list[EntryAnalysisResult],
    themes: list,
    issues: list,
    sentiment_method: str,
    theme_method: str,
    issue_method: str,
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

    issue_labels = [
        result.issue.label
        for result in entry_analysis_results
        if result.issue and result.issue.label
    ]

    issue_distribution = dict(Counter(issue_labels))

    dominant_issue_label = None
    if issue_distribution:
        dominant_issue_label = max(
            issue_distribution,
            key=issue_distribution.get,
        )

    dominant_issue_weight = None
    dominant_issue_weights = []

    if dominant_issue_label:
        dominant_issue_weights = [
            result.issue.weight
            for result in entry_analysis_results
            if (
                result.issue
                and result.issue.label == dominant_issue_label
                and result.issue.weight is not None
            )
        ]

        if dominant_issue_weights:
            dominant_issue_weight = (
                sum(dominant_issue_weights) / len(dominant_issue_weights)
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

        dominant_issue_label=dominant_issue_label,
        dominant_issue_weight=dominant_issue_weight,
        issue_distribution=issue_distribution,


        entry_analysis_results=entry_analysis_results,
        total_entries=len(entry_analysis_results),
        total_themes=len(themes),
        total_issues=len(issues),

        methods={
            "sentiment": sentiment_method,
            "theme": theme_method,
            "issue": issue_method,
        },
        metadata={
            "pipeline": "v1",
        },
    )