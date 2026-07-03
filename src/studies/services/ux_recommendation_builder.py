from collections import defaultdict

from studies.models import (
    CanonicalIssueToFrameworkMapping,
    CanonicalThemeToFrameworkMapping,
    EntryAnalysisIssue,
    EntryAnalysisTheme,
    StudyAnalysis,
    UXFrameworkMappingStatus,
)


def build_ux_recommendation_report(run: StudyAnalysis) -> dict:
    approved_issue_mappings = CanonicalIssueToFrameworkMapping.objects.filter(
        status=UXFrameworkMappingStatus.APPROVED,
        criterion__is_active=True,
        issue__is_active=True,
    ).select_related(
        "issue",
        "criterion",
        "criterion__framework",
    )

    approved_theme_mappings = CanonicalThemeToFrameworkMapping.objects.filter(
        status=UXFrameworkMappingStatus.APPROVED,
        criterion__is_active=True,
        theme__is_active=True,
    ).select_related(
        "theme",
        "criterion",
        "criterion__framework",
    )

    issue_mappings_by_issue_id = defaultdict(list)
    for mapping in approved_issue_mappings:
        issue_mappings_by_issue_id[mapping.issue_id].append(mapping)

    theme_mappings_by_theme_id = defaultdict(list)
    for mapping in approved_theme_mappings:
        theme_mappings_by_theme_id[mapping.theme_id].append(mapping)

    report_sections = {}

    for issue_summary in run.issue_summaries.select_related("issue").all():
        issue = issue_summary.issue
        mappings = issue_mappings_by_issue_id.get(issue.id, [])

        for mapping in mappings:
            criterion = mapping.criterion
            key = criterion.id

            section = report_sections.setdefault(
                key,
                {
                    "criterion_id": criterion.id,
                    "framework": criterion.framework.name,
                    "criterion_code": criterion.code,
                    "criterion_name": criterion.name,
                    "criterion_description": criterion.description,
                    "recommendation_guidance": criterion.recommendation_guidance,
                    "evaluation_questions": criterion.evaluation_questions,
                    "issues": [],
                    "themes": [],
                    "entry_count": 0,
                    "average_sentiment_score": None,
                    "priority_score": 0,
                    "evidence": [],
                },
            )

            section["issues"].append(
                {
                    "id": issue.id,
                    "name": issue.name,
                    "entry_count": issue_summary.entry_count,
                    "average_confidence_score": issue_summary.average_confidence_score,
                    "average_sentiment_score": issue_summary.average_sentiment_score,
                    "mapping_rationale": mapping.rationale,
                }
            )

            section["entry_count"] += issue_summary.entry_count
            section["priority_score"] += calculate_priority_score(
                entry_count=issue_summary.entry_count,
                average_sentiment_score=issue_summary.average_sentiment_score,
            )

    for theme_summary in run.theme_summaries.select_related("theme").all():
        theme = theme_summary.theme
        mappings = theme_mappings_by_theme_id.get(theme.id, [])

        for mapping in mappings:
            criterion = mapping.criterion
            key = criterion.id

            section = report_sections.setdefault(
                key,
                {
                    "criterion_id": criterion.id,
                    "framework": criterion.framework.name,
                    "criterion_code": criterion.code,
                    "criterion_name": criterion.name,
                    "criterion_description": criterion.description,
                    "recommendation_guidance": criterion.recommendation_guidance,
                    "evaluation_questions": criterion.evaluation_questions,
                    "issues": [],
                    "themes": [],
                    "entry_count": 0,
                    "average_sentiment_score": None,
                    "priority_score": 0,
                    "evidence": [],
                },
            )

            section["themes"].append(
                {
                    "id": theme.id,
                    "name": theme.name,
                    "entry_count": theme_summary.entry_count,
                    "average_confidence_score": theme_summary.average_confidence_score,
                    "average_sentiment_score": theme_summary.average_sentiment_score,
                    "mapping_rationale": mapping.rationale,
                }
            )

            section["priority_score"] += calculate_priority_score(
                entry_count=theme_summary.entry_count,
                average_sentiment_score=theme_summary.average_sentiment_score,
                theme_weight=True,
            )

    sections = list(report_sections.values())

    for section in sections:
        section["evidence"] = get_representative_evidence(
            run=run,
            issue_ids=[issue["id"] for issue in section["issues"]],
            theme_ids=[theme["id"] for theme in section["themes"]],
        )

    sections.sort(
        key=lambda section: section["priority_score"],
        reverse=True,
    )

    return {
        "run_id": run.id,
        "study_id": run.study_id,
        "study_title": run.study.title,
        "sections": sections,
    }


def calculate_priority_score(entry_count, average_sentiment_score, theme_weight=False):
    sentiment_score = average_sentiment_score if average_sentiment_score is not None else 0

    negative_weight = abs(min(sentiment_score, 0))
    base_score = entry_count * (1 + negative_weight)

    if theme_weight:
        return base_score * 0.5

    return base_score


def get_representative_evidence(run, issue_ids=None, theme_ids=None, limit=3):
    issue_ids = issue_ids or []
    theme_ids = theme_ids or []

    evidence = []

    if issue_ids:
        issue_assignments = (
            EntryAnalysisIssue.objects
            .filter(
                entry_analysis__run=run,
                issue_id__in=issue_ids,
            )
            .select_related(
                "issue",
                "entry_analysis",
                "entry_analysis__entry",
            )
            .order_by("entry_analysis__sentiment_score")[:limit]
        )

        for assignment in issue_assignments:
            evidence.append(
                {
                    "type": "issue",
                    "label": assignment.issue.name,
                    "entry_id": assignment.entry_analysis.entry_id,
                    "sentiment_score": assignment.entry_analysis.sentiment_score,
                    "excerpt": assignment.entry_analysis.entry.content[:500],
                    "rationale": assignment.rationale,
                }
            )

    if len(evidence) < limit and theme_ids:
        remaining = limit - len(evidence)

        theme_assignments = (
            EntryAnalysisTheme.objects
            .filter(
                entry_analysis__run=run,
                theme_id__in=theme_ids,
            )
            .select_related(
                "theme",
                "entry_analysis",
                "entry_analysis__entry",
            )
            .order_by("entry_analysis__sentiment_score")[:remaining]
        )

        for assignment in theme_assignments:
            evidence.append(
                {
                    "type": "theme",
                    "label": assignment.theme.name,
                    "entry_id": assignment.entry_analysis.entry_id,
                    "sentiment_score": assignment.entry_analysis.sentiment_score,
                    "excerpt": assignment.entry_analysis.entry.content[:500],
                    "rationale": assignment.rationale,
                }
            )

    return evidence