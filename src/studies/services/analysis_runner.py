from __future__ import annotations

from collections import Counter
import re
from typing import Any

from django.db import transaction
from django.utils import timezone

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from studies.models import (
    Study,
    StudyStatus,
    StudyAnalysisRun,
    DiaryEntry,
    DiaryEntryAnalysis,
    AnalysisRunStatus,
)


DEFAULT_THEME_COUNT = 3
DEFAULT_THEME_TERMS = 6
MAX_SUMMARY_LENGTH = 180

ISSUE_KEYWORDS = {
    "bug": ["bug", "error", "crash", "broken", "glitch", "failed", "failure"],
    "confusion": ["confusing", "unclear", "lost", "didn't understand", "not sure", "uncertain"],
    "performance": ["slow", "lag", "laggy", "delay", "delayed", "loading", "freeze", "frozen"],
    "usability": ["hard", "difficult", "awkward", "annoying", "frustrating", "frustration"],
}


def normalize_entry_text(text: str) -> str:
    """
    Light normalization only.
    Do not over-clean, because VADER benefits from punctuation/casing cues.
    """
    return " ".join((text or "").split())


def make_entry_summary(text: str, max_length: int = MAX_SUMMARY_LENGTH) -> str:
    normalized = normalize_entry_text(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length].rstrip() + "..."


def detect_issue_tags(text: str) -> list[str]:
    lowered = (text or "").lower()
    tags: list[str] = []

    for tag, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(tag)

    return tags


def analyze_entry_sentiment(content: str, sia: SentimentIntensityAnalyzer) -> dict[str, Any]:
    normalized = normalize_entry_text(content)
    scores = sia.polarity_scores(normalized)
    sentiment = round(scores["compound"], 4)
    sentiment_category = DiaryEntry.map_sentiment_to_category(sentiment)
    issue_tags = detect_issue_tags(normalized)

    return {
        "sentiment": sentiment,
        "sentiment_category": sentiment_category,
        "issue_detected": bool(issue_tags),
        "issue_tags": issue_tags,
        "entry_summary": make_entry_summary(normalized),
        "raw_response": {
            "analyzer": "vader_tfidf_nmf",
            "sentiment_engine": "nltk_vader",
            "vader_scores": scores,
        },
    }


def build_theme_label(terms: list[str]) -> str:
    """
    Simple human-readable label from top NMF terms.
    You can later replace this with a manual labeling table if you want nicer labels.
    """
    cleaned_terms = [term.replace("_", " ") for term in terms if term]
    return " / ".join(cleaned_terms[:3]) if cleaned_terms else "general experience"


def extract_themes_for_entries(
    entries: list[DiaryEntry],
    n_components: int = DEFAULT_THEME_COUNT,
    top_terms_per_theme: int = DEFAULT_THEME_TERMS,
) -> tuple[dict[int, list[str]], list[dict[str, Any]]]:
    """
    Returns:
      - entry_id -> assigned theme labels
      - recurring themes metadata for study/run aggregates
    """
    if not entries:
        return {}, []

    entry_texts = [normalize_entry_text(entry.content) for entry in entries]
    non_empty_entries = [(entry, text) for entry, text in zip(entries, entry_texts) if text]

    if len(non_empty_entries) < 2:
        fallback = {
            entry.id: ["general experience"]
            for entry, text in non_empty_entries
        }
        recurring = [{"theme": "general experience", "count": len(non_empty_entries)}]
        return fallback, recurring

    filtered_entries = [entry for entry, _ in non_empty_entries]
    filtered_texts = [text for _, text in non_empty_entries]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=1 if len(filtered_texts) < 5 else 2,
        max_features=1000,
    )
    tfidf_matrix = vectorizer.fit_transform(filtered_texts)

    if tfidf_matrix.shape[1] == 0:
        fallback = {entry.id: ["general experience"] for entry in filtered_entries}
        recurring = [{"theme": "general experience", "count": len(filtered_entries)}]
        return fallback, recurring

    actual_components = max(
        1,
        min(n_components, tfidf_matrix.shape[0], tfidf_matrix.shape[1]),
    )

    nmf = NMF(
        n_components=actual_components,
        init="nndsvda",
        random_state=42,
        max_iter=400,
    )
    doc_topic_matrix = nmf.fit_transform(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()

    theme_labels: list[str] = []
    recurring_themes: list[dict[str, Any]] = []

    for topic_idx, topic_weights in enumerate(nmf.components_):
        top_indices = topic_weights.argsort()[::-1][:top_terms_per_theme]
        top_terms = [feature_names[i] for i in top_indices]
        label = build_theme_label(top_terms)
        theme_labels.append(label)

        recurring_themes.append({
            "theme": label,
            "keywords": top_terms,
            "topic_index": topic_idx,
        })

    entry_theme_map: dict[int, list[str]] = {}
    theme_counter = Counter()

    for row_idx, entry in enumerate(filtered_entries):
        topic_scores = doc_topic_matrix[row_idx]
        if len(topic_scores) == 0:
            assigned = ["general experience"]
        else:
            dominant_topic_idx = int(topic_scores.argmax())
            assigned = [theme_labels[dominant_topic_idx]]
            theme_counter.update(assigned)

        entry_theme_map[entry.id] = assigned

    recurring_output = []
    for theme, count in theme_counter.most_common():
        metadata = next((item for item in recurring_themes if item["theme"] == theme), {})
        recurring_output.append({
            "theme": theme,
            "count": count,
            "keywords": metadata.get("keywords", []),
        })

    # Give empty-text entries a fallback
    for entry in entries:
        entry_theme_map.setdefault(entry.id, ["general experience"])

    return entry_theme_map, recurring_output


def build_sentiment_distribution(entry_results: list[dict]) -> dict:
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
    counter = Counter()
    for result in entry_results:
        counter.update(result.get("themes", []))

    return [
        {"theme": theme, "count": count}
        for theme, count in counter.most_common()
    ]


def build_evolution_over_time(entries_with_results: list[tuple]) -> list[dict]:
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


@transaction.atomic
def run_study_analysis(study_id: int) -> StudyAnalysisRun:
    study = Study.objects.get(pk=study_id)

    run = StudyAnalysisRun.objects.create(
        study=study,
        status=AnalysisRunStatus.RUNNING,
        analysis_model="vader_tfidf_nmf",
        analysis_version="v2",
    )

    study.status = StudyStatus.ANALYZING
    study.save(update_fields=["status"])

    entry_results = []
    entries_with_results = []

    try:
        entries = list(study.entries.all().order_by("created_at"))
        now = timezone.now()

        sia = SentimentIntensityAnalyzer()
        entry_theme_map, recurring_themes = extract_themes_for_entries(entries)

        for entry in entries:
            result = analyze_entry_sentiment(entry.content, sia)
            result["themes"] = entry_theme_map.get(entry.id, ["general experience"])
            result["raw_response"]["theme_engine"] = "sklearn_tfidf_nmf"

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

        sentiments = [
            result["sentiment"]
            for result in entry_results
            if result.get("sentiment") is not None
        ]
        avg_sentiment = round(sum(sentiments) / len(sentiments), 4) if sentiments else None
        sentiment_category = Study.map_sentiment_to_category(avg_sentiment)

        sentiment_distribution = build_sentiment_distribution(entry_results)
        recurring_issues = build_recurring_issues(entry_results)
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
        run.save(update_fields=[
            "sentiment_distribution",
            "recurring_issues",
            "recurring_themes",
            "evolution_over_time",
            "top_representative_quotes",
            "status",
            "completed_at",
        ])

        now = timezone.now()
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
            ]
        )

        return run

    except Exception as exc:
        run.status = AnalysisRunStatus.FAILED
        run.error_message = str(exc)
        run.completed_at = timezone.now()
        run.save(update_fields=["status", "error_message", "completed_at"])
        raise