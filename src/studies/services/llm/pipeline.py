# This file defines the main LLM pipeline for analyzing study entries
# Steps:
# 1. Receive a study_id and list/query for entries to analyze
# 2. Explore study-level themes and issues using an LLM and all entries (1ST PASS)
# 3. Analyze each entry with an LLM to obtain sentiment, theme, and issue assignment (2ND PASS)
# 4. Return one StudyAnalysisResult containing all the entry-level results

############################
####### LLM PIPELINE #######
############################

### DEBUG STUFF ###

from datetime import datetime
from pathlib import Path
from pprint import pformat
import random
import traceback

def _write_first_pass_debug(
    *,
    study_id: int,
    provider: str,
    model: str | None,
    themes: list[ThemeResult],
    issues: list[IssueResult],
) -> Path:
    debug_directory = Path("debug")
    debug_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    filename_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    debug_file = debug_directory / f"llm_first_pass_study_{study_id}_{filename_timestamp}.log"

    theme_ids = [
        theme.theme_id
        for theme in themes
        if theme and theme.theme_id is not None
    ]

    issue_ids = [
        issue.issue_id
        for issue in issues
        if issue and issue.issue_id is not None
    ]

    with debug_file.open("a", encoding="utf-8") as file:
        file.write("\n")
        file.write("=" * 100)
        file.write("\n")
        file.write(f"TIMESTAMP: {timestamp}\n")
        file.write(f"STUDY ID: {study_id}\n")
        file.write(f"PROVIDER: {provider}\n")
        file.write(f"MODEL: {model or 'DEFAULT'}\n")
        file.write("=" * 100)
        file.write("\n\n")

        file.write("1ST PASS THEME IDENTIFICATION\n")
        file.write("-" * 100)
        file.write("\n")
        file.write(f"THEME COUNT: {len(themes)}\n")
        file.write(f"VALID THEME IDS: {theme_ids}\n\n")
        file.write("THEMES:\n\n")
        file.write(pformat(themes, width=140, sort_dicts=False))
        file.write("\n\n")

        file.write("1ST PASS ISSUE IDENTIFICATION\n")
        file.write("-" * 100)
        file.write("\n")
        file.write(f"ISSUE COUNT: {len(issues)}\n")
        file.write(f"VALID ISSUE IDS: {issue_ids}\n\n")
        file.write("ISSUES:\n\n")
        file.write(pformat(issues, width=140, sort_dicts=False))
        file.write("\n\n")

    return debug_file

def _write_second_pass_debug(
    *,
    debug_file: Path,
    entry,
    status: str,
    result: EntryAnalysisResult | None = None,
    error: Exception | None = None,
) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    entry_id = getattr(entry, "id", None)
    entry_text = getattr(entry, "content", None)

    with debug_file.open("a", encoding="utf-8") as file:
        file.write("\n")
        file.write("=" * 100)
        file.write("\n")
        file.write("2ND PASS ENTRY ANALYSIS\n")
        file.write("-" * 100)
        file.write("\n")
        file.write(f"TIMESTAMP: {timestamp}\n")
        file.write(f"STATUS: {status}\n")
        file.write(f"ENTRY ID: {entry_id}\n")
        file.write(f"ENTRY TEXT:\n{entry_text}\n\n")

        if result is not None:
            file.write("RESULT:\n\n")
            file.write(pformat(result, width=140, sort_dicts=False))
            file.write("\n")

        if error is not None:
            file.write("ERROR TYPE:\n")
            file.write(f"{type(error).__name__}\n\n")

            file.write("ERROR MESSAGE:\n")
            file.write(f"{error}\n\n")

            file.write("TRACEBACK:\n")
            file.write(traceback.format_exc())
            file.write("\n")

        file.write("=" * 100)
        file.write("\n")

### DEBUG STUFF ###

from collections import Counter

from studies.services.contracts import (
    EntryAnalysisResult,
    IssueResult,
    StudyAnalysisResult,
    ThemeResult,
)
from studies.services.llm.entry_analyzer_split import analyze_entry_with_llm
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

    # shuffle entries
    random.Random(666).shuffle(entry_list)

    if not entry_list:
        raise RuntimeError("No diary entries available for LLM analysis.")

    # FIRST PASS: Build the study-level theme catalog.
    themes = explore_themes_with_llm(
        entries=entry_list,
        provider=provider,
        model=model,
        max_themes=max_themes,
    )

    if not themes:
        raise RuntimeError("LLM theme exploration did not return any themes.")

    # FIRST PASS: Build the study-level issue catalog.
    issues = explore_issues_with_llm(
        entries=entry_list,
        provider=provider,
        model=model,
        max_issues=max_issues,
    )

    debug_file = _write_first_pass_debug(
        study_id=study_id,
        provider=provider,
        model=model,
        themes=themes,
        issues=issues,
    )

    print("\n1ST PASS THEME IDENTIFICATION")
    print("--------------------------------")
    print(f"THEMES: {len(themes)}")
    print(
        "VALID THEME IDS:",
        [
            theme.theme_id
            for theme in themes
            if theme and theme.theme_id is not None
        ],
    )

    print("\n1ST PASS ISSUE IDENTIFICATION")
    print("--------------------------------")
    print(f"ISSUES: {len(issues)}")
    print(
        "VALID ISSUE IDS:",
        [
            issue.issue_id
            for issue in issues
            if issue and issue.issue_id is not None
        ],
    )

    print(f"\nFull first-pass debug written to: {debug_file.resolve()}")

    # SECOND PASS: Analyze each entry using both catalogs.
    entry_analysis_results = []

    for entry in entry_list:
        try:
            entry_analysis_result = analyze_entry_with_llm(
                entry=entry,
                theme_catalog=themes,
                issue_catalog=issues,
                provider=provider,
                model=model,
            )

            entry_analysis_results.append(entry_analysis_result)

            _write_second_pass_debug(
                debug_file=debug_file,
                entry=entry,
                status="SUCCESS",
                result=entry_analysis_result,
            )

        except Exception as error:
            _write_second_pass_debug(
                debug_file=debug_file,
                entry=entry,
                status="FAILED",
                error=error,
            )

            print(
                f"\nSecond-pass analysis failed for entry "
                f"{getattr(entry, 'id', 'UNKNOWN')}."
            )
            print(f"Debug written to: {debug_file.resolve()}")

            raise


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
