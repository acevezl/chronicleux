from __future__ import annotations

from collections import Counter
import re
from typing import Any

from django.db import transaction
from django.utils import timezone

from studies.models import (
    Study,
    StudyStatus,
    StudyAnalysisRun,
    DiaryEntry,
    DiaryEntryAnalysis,
    AnalysisRunStatus,
)

from studies.services.nlp.pipeline import analyze_study_entries

DEFAULT_THEME_COUNT = 3
DEFAULT_THEME_TERMS = 6
MAX_SUMMARY_LENGTH = 180

ISSUE_KEYWORDS = {
    "bug": ["bug", "error", "crash", "broken", "glitch", "failed", "failure"],
    "confusion": ["confusing", "unclear", "lost", "didn't understand", "not sure", "uncertain"],
    "performance": ["slow", "lag", "laggy", "delay", "delayed", "loading", "freeze", "frozen"],
    "usability": ["hard", "difficult", "awkward", "annoying", "frustrating", "frustration"],
}

# ---------------------
# Helper functions 
# ---------------------

def normalize_entry_text(text: str) -> str:
    """
    Light normalization of entries for better analysis results. 
    I'm preserving as much of the original text as possible, since VADER relies on punctuation and casing cues for sentiment analysis. 
    This function primarily collapses excessive whitespace and ensures we have a string to work with.
    (In other words, clean and trim, but not over-clean, to preserve sentiment cues.)
    """
    return " ".join((text or "").split())


def make_entry_summary(text: str, max_length: int = MAX_SUMMARY_LENGTH) -> str:
    """
    Create a summary of the entry text, truncated to a maximum length.
    """
    normalized = normalize_entry_text(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length].rstrip() + "..."


def detect_issue_tags(text: str) -> list[str]:
    """
    Detect potential issue tags in the entry text based on keyword matching.
    This helps identify common problems users might be mentioning, which can be useful for highlighting recurring issues in the study analysis.
    """
    lowered = (text or "").lower()
    tags: list[str] = []

    for tag, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(tag)

    return tags

def build_sentiment_distribution(entry_results: list[dict]) -> dict:
    """
    Build a distribution of sentiment categories from the entry results, including both COUNTS and PERCENTAGES.
    Useful for understanding the sentiment landscape of the study.
    """
    counts = Counter(
        result["sentiment_category"]
        for result in entry_results
        if result.get("sentiment_category")
    )
    total = sum(counts.values())

    percentages = {}
    if total:
        percentages = {
            key: round(value / total, 4)
            for key, value in counts.items()
        }

    return {
        "counts": dict(counts),
        "percentages": percentages,
        "total_entries": total,
    }

def build_recurring_issues(entry_results: list[dict]) -> list[dict]:
    """
    Build a list of recurring issues based on detected issue tags in the entry results, including both COUNTS and PERCENTAGES.
    This helps identify common problems users are mentioning across entries in the study.
    """
    counter = Counter()
    total_issue_entries = 0

    for result in entry_results:
        tags = result.get("issue_tags", [])
        if tags:
            total_issue_entries += 1
            counter.update(tags)

    output = []
    for tag, count in counter.most_common():
        percentage = round(count / total_issue_entries, 4) if total_issue_entries else 0
        output.append({
            "tag": tag,
            "count": count,
            "percentage": percentage,
        })
    return output

def build_recurring_themes(entry_results: list[dict]) -> list[dict]:
    """
    Build a list of recurring themes based on the themes detected in the entry results, including both COUNTS and PERCENTAGES.
    This helps identify common themes that are emerging across entries in the study.
    """
    counter = Counter()
    for result in entry_results:
        counter.update(result.get("themes", []))

    return [
        {"theme": theme, "count": count}
        for theme, count in counter.most_common()
    ]

def build_evolution_over_time(entries_with_results: list[tuple]) -> list[dict]:
    """
    Build a time series of average sentiment and sentiment distribution over time (by day) based on the entry results.
    This helps visualize how sentiment is evolving throughout the study period.
    """
    grouped = {}

    for entry, result in entries_with_results:
        day = entry.created_at.date().isoformat()
        grouped.setdefault(day, []).append(result)

    output = []
    for day in sorted(grouped.keys()):
        day_results = grouped[day]
        sentiments = [r["sentiment"] for r in day_results if r.get("sentiment") is not None]
        avg_sentiment = round(sum(sentiments) / len(sentiments), 4) if sentiments else None

        output.append({
            "date": day,
            "entry_count": len(day_results),
            "avg_sentiment": avg_sentiment,
            "sentiment_distribution": build_sentiment_distribution(day_results),
            "top_themes": build_recurring_themes(day_results)[:3],
        })

    return output

def build_top_representative_quotes(entries_with_results: list[tuple]) -> list[dict]:
    """
    Build a list of top representative quotes from entries with the most extreme sentiment scores, including their sentiment category and themes.
    This helps surface specific user feedback that is strongly positive or negative, along with the context ofm themes they mention.
    """
    selected = []

    ranked = sorted(
        entries_with_results,
        key=lambda pair: abs(pair[1].get("sentiment") or 0),
        reverse=True,
    )

    for entry, result in ranked[:5]:
        quote = (entry.content or "").strip()
        if len(quote) > 240:
            quote = quote[:240] + "..."

        selected.append({
            "entry_id": entry.id,
            "quote": quote,
            "sentiment_category": result.get("sentiment_category"),
            "themes": result.get("themes", []),
        })

    return selected

#---------------------
# Main analysis runner function
#---------------------
@transaction.atomic
def run_study_analysis(study_id: int) -> StudyAnalysisRun:
	study = Study.objects.get(pk=study_id)

	sentiment_method = "vader"
	theme_method = "tfidf_nmf"

	run = StudyAnalysisRun.objects.create(
		study=study,
		status=AnalysisRunStatus.RUNNING,
		analysis_model=f"{sentiment_method}_{theme_method}",
		analysis_version="v3",
	)

	study.status = StudyStatus.MACHINE_ANALYSIS
	study.save(update_fields=["status"])

	entry_results = []
	entries_with_results = []

	try:
		entries = list(study.entries.all().order_by("created_at"))
		now = timezone.now()

		study_analysis_result = analyze_study_entries(
			study_id=study.id,
			entries=entries,
			sentiment_method=sentiment_method,
			theme_method=theme_method,
		)

		for entry_analysis_result in study_analysis_result.entry_analysis_results:
			entry = next(
				(entry for entry in entries if entry.id == entry_analysis_result.entry_id),
				None,
			)

			if entry is None:
				continue

			sentiment_score = None
			sentiment_category = None
			sentiment_metadata = {}

			if entry_analysis_result.sentiment:
				sentiment_score = round(entry_analysis_result.sentiment.score, 4)
				sentiment_category = entry_analysis_result.sentiment.label
				sentiment_metadata = entry_analysis_result.sentiment.metadata

			theme_labels = []
			theme_metadata = {}

			if entry_analysis_result.theme:
				theme_labels = [entry_analysis_result.theme.label]
				theme_metadata = {
					"theme_id": entry_analysis_result.theme.theme_id,
					"keywords": entry_analysis_result.theme.keywords,
					"method": entry_analysis_result.theme.method,
					"metadata": entry_analysis_result.theme.metadata,
					"theme_weight": entry_analysis_result.metadata.get("theme_weight"),
				}

			normalized_content = normalize_entry_text(entry.content)
			issue_tags = detect_issue_tags(normalized_content)

			result = {
				"sentiment": sentiment_score,
				"sentiment_category": sentiment_category,
				"issue_detected": bool(issue_tags),
				"issue_tags": issue_tags,
				"themes": theme_labels,
				"entry_summary": make_entry_summary(normalized_content),
				"raw_response": {
					"analysis_model": run.analysis_model,
					"analysis_version": run.analysis_version,
					"sentiment_method": sentiment_method,
					"theme_method": theme_method,
					"sentiment_metadata": sentiment_metadata,
					"theme_metadata": theme_metadata,
				},
			}

			DiaryEntryAnalysis.objects.create(
				run=run,
				entry=entry,
				sentiment=result["sentiment"],
				sentiment_category=result["sentiment_category"],
				issue_detected=result["issue_detected"],
				issue_tags=result["issue_tags"],
				themes=result["themes"],
				entry_summary=result["entry_summary"],
				raw_response=result["raw_response"],
			)

			entry.sentiment = result["sentiment"]
			entry.sentiment_category = result["sentiment_category"]
			entry.analysis_issue_detected = result["issue_detected"]
			entry.analysis_issue_tags = result["issue_tags"]
			entry.analysis_themes = result["themes"]
			entry.entry_summary = result["entry_summary"]
			entry.analysis_model = run.analysis_model
			entry.analysis_version = run.analysis_version
			entry.analyzed_at = now

			entry.save(
				update_fields=[
					"sentiment",
					"sentiment_category",
					"analysis_issue_detected",
					"analysis_issue_tags",
					"analysis_themes",
					"entry_summary",
					"analysis_model",
					"analysis_version",
					"analyzed_at",
				]
			)

			entry_results.append(result)
			entries_with_results.append((entry, result))

		avg_sentiment = (
			round(study_analysis_result.average_study_sentiment_score, 4)
			if study_analysis_result.average_study_sentiment_score is not None
			else None
		)

		sentiment_category = study_analysis_result.dominant_study_sentiment_label

		sentiment_distribution = {
			"counts": study_analysis_result.study_sentiment_distribution,
			"average_score": avg_sentiment,
			"dominant_label": study_analysis_result.dominant_study_sentiment_label,
			"total_entries": study_analysis_result.total_entries,
		}

		recurring_issues = build_recurring_issues(entry_results)

		recurring_themes = [
			{
				"theme": theme,
				"count": count,
			}
			for theme, count in study_analysis_result.study_theme_distribution.items()
		]

		evolution_over_time = build_evolution_over_time(entries_with_results)
		top_representative_quotes = build_top_representative_quotes(entries_with_results)

		now = timezone.now()

		run.sentiment_distribution = sentiment_distribution
		run.recurring_issues = recurring_issues
		run.recurring_themes = recurring_themes
		run.evolution_over_time = evolution_over_time
		run.top_representative_quotes = top_representative_quotes
		run.status = AnalysisRunStatus.COMPLETED
		run.completed_at = now

		run.save(
			update_fields=[
				"sentiment_distribution",
				"recurring_issues",
				"recurring_themes",
				"evolution_over_time",
				"top_representative_quotes",
				"status",
				"completed_at",
			]
		)

		study.avg_sentiment = avg_sentiment
		study.sentiment_category = sentiment_category
		study.sentiment_distribution = sentiment_distribution
		study.recurring_issues = recurring_issues
		study.recurring_themes = recurring_themes
		study.evolution_over_time = evolution_over_time
		study.top_representative_quotes = top_representative_quotes
		study.analysis_model = run.analysis_model
		study.analysis_version = run.analysis_version
		study.analyzed_at = now
		study.status = StudyStatus.HUMAN_ANALYSIS

		study.save(
			update_fields=[
				"avg_sentiment",
				"sentiment_category",
				"sentiment_distribution",
				"recurring_issues",
				"recurring_themes",
				"evolution_over_time",
				"top_representative_quotes",
				"analysis_model",
				"analysis_version",
				"analyzed_at",
				"status",
			]
		)

		return run

	except Exception as exc:
		run.status = AnalysisRunStatus.FAILED
		run.error_message = str(exc)
		run.completed_at = timezone.now()
		run.save(update_fields=["status", "error_message", "completed_at"])
		raise