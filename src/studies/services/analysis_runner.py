from __future__ import annotations

from collections import Counter
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


def normalize_entry_text(text: str) -> str:
    return " ".join((text or "").split())


def analyze_entry(content: str) -> dict:
    normalized = normalize_entry_text(content)

    # stub implementation for now
    sentiment = None
    if any(word in normalized.lower() for word in ["love", "great", "easy", "good"]):
        sentiment = 0.5
    elif any(word in normalized.lower() for word in ["bad", "frustrating", "confusing", "annoying"]):
        sentiment = -0.5

    sentiment_category = DiaryEntry.map_sentiment_to_category(sentiment)

    issue_detected = any(
        word in normalized.lower()
        for word in ["issue", "problem", "bug", "frustrating", "confusing", "error"]
    )

    issue_tags = []
    if "bug" in normalized.lower() or "error" in normalized.lower():
        issue_tags.append("bug")
    if "confusing" in normalized.lower():
        issue_tags.append("confusion")
    if "slow" in normalized.lower():
        issue_tags.append("performance")

    themes = []
    if "onboarding" in normalized.lower():
        themes.append("onboarding")
    if "navigation" in normalized.lower():
        themes.append("navigation")
    if not themes:
        themes.append("general experience")

    summary = normalized[:180]
    if len(normalized) > 180:
        summary += "..."

    return {
        "sentiment": sentiment,
        "sentiment_category": sentiment_category,
        "issue_detected": issue_detected,
        "issue_tags": issue_tags,
        "themes": themes,
        "entry_summary": summary,
        "raw_response": {
            "analyzer": "stub_v1"
        },
    }


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

    for entry, result in entries_with_results[:5]:
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
        analysis_model="stub",
        analysis_version="v1",
    )

    study.status = StudyStatus.ANALYZING
    study.save(update_fields=["status"])

    entry_results = []
    entries_with_results = []

    try:
        entries = list(
            study.entries.all().order_by("created_at")
        )

        now = timezone.now()

        for entry in entries:
            result = analyze_entry(entry.content)

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
        recurring_themes = build_recurring_themes(entry_results)
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
        study.avg_sentiment  = avg_sentiment
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