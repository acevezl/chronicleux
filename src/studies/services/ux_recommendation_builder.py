from collections import defaultdict

from studies.models import (
    CanonicalIssueToFrameworkMapping,
    CanonicalThemeToFrameworkMapping,
    EntryAnalysisIssue,
    EntryAnalysisTheme,
    StudyAnalysis,
    UXFrameworkMappingStatus,
)


THEME_PRIORITY_WEIGHT = 0.5
EVIDENCE_LIMIT = 3


def build_ux_recommendation_report(run: StudyAnalysis) -> dict:
    issue_mappings = get_approved_mappings_by_source_id(
        CanonicalIssueToFrameworkMapping,
        source_id_field="issue_id",
        source_active_field="issue__is_active",
        source_related_field="issue",
    )

    theme_mappings = get_approved_mappings_by_source_id(
        CanonicalThemeToFrameworkMapping,
        source_id_field="theme_id",
        source_active_field="theme__is_active",
        source_related_field="theme",
    )

    report_sections = {}

    add_issue_summaries_to_sections(
        run=run,
        report_sections=report_sections,
        mappings_by_issue_id=issue_mappings,
    )

    add_theme_summaries_to_sections(
        run=run,
        report_sections=report_sections,
        mappings_by_theme_id=theme_mappings,
    )

    sections = list(report_sections.values())

    add_evidence_to_sections(
        run=run,
        sections=sections,
    )

    normalize_priority_scores(sections)

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


def get_approved_mappings_by_source_id(
    mapping_model,
    source_id_field,
    source_active_field,
    source_related_field,
):
    mappings = (
        mapping_model.objects
        .filter(
            status=UXFrameworkMappingStatus.APPROVED,
            criterion__is_active=True,
            **{source_active_field: True},
        )
        .select_related(
            source_related_field,
            "criterion",
            "criterion__framework",
        )
    )

    mappings_by_source_id = defaultdict(list)

    for mapping in mappings:
        source_id = getattr(mapping, source_id_field)
        mappings_by_source_id[source_id].append(mapping)

    return mappings_by_source_id


def add_issue_summaries_to_sections(
    run,
    report_sections,
    mappings_by_issue_id,
):
    issue_summaries = run.issue_summaries.select_related("issue").all()

    for issue_summary in issue_summaries:
        issue = issue_summary.issue
        mappings = mappings_by_issue_id.get(issue.id, [])

        for mapping in mappings:
            section = get_or_create_report_section(
                report_sections=report_sections,
                criterion=mapping.criterion,
            )

            section["issues"].append(
                build_summary_item(
                    source=issue,
                    summary=issue_summary,
                    mapping=mapping,
                )
            )

            section["entry_count"] += issue_summary.entry_count

            section["priority_score"] += calculate_priority_score(
                entry_count=issue_summary.entry_count,
                average_sentiment_score=(
                    issue_summary.average_sentiment_score
                ),
            )


def add_theme_summaries_to_sections(
    run,
    report_sections,
    mappings_by_theme_id,
):
    theme_summaries = run.theme_summaries.select_related("theme").all()

    for theme_summary in theme_summaries:
        theme = theme_summary.theme
        mappings = mappings_by_theme_id.get(theme.id, [])

        for mapping in mappings:
            section = get_or_create_report_section(
                report_sections=report_sections,
                criterion=mapping.criterion,
            )

            section["themes"].append(
                build_summary_item(
                    source=theme,
                    summary=theme_summary,
                    mapping=mapping,
                )
            )

            section["priority_score"] += calculate_priority_score(
                entry_count=theme_summary.entry_count,
                average_sentiment_score=(
                    theme_summary.average_sentiment_score
                ),
                weight=THEME_PRIORITY_WEIGHT,
            )


def get_or_create_report_section(
    report_sections,
    criterion,
):
    return report_sections.setdefault(
        criterion.id,
        {
            "criterion_id": criterion.id,
            "framework": criterion.framework.name,
            "criterion_code": criterion.code,
            "criterion_name": criterion.name,
            "criterion_description": criterion.description,
            "recommendation_guidance": (
                criterion.recommendation_guidance
            ),
            "evaluation_questions": criterion.evaluation_questions,
            "issues": [],
            "themes": [],
            "entry_count": 0,
            "average_sentiment_score": None,
            "raw_priority_score": 0,
            "priority_score": 0,
            "evidence": [],
        },
    )


def build_summary_item(
    source,
    summary,
    mapping,
):
    return {
        "id": source.id,
        "name": source.name,
        "entry_count": summary.entry_count,
        "average_confidence_score": (
            summary.average_confidence_score
        ),
        "average_sentiment_score": (
            summary.average_sentiment_score
        ),
        "mapping_rationale": mapping.rationale,
    }


def calculate_priority_score(
    entry_count,
    average_sentiment_score,
    weight=1.0,
):
    sentiment_score = (
        average_sentiment_score
        if average_sentiment_score is not None
        else 0
    )

    negative_weight = abs(min(sentiment_score, 0))
    base_score = entry_count * (1 + negative_weight)

    return base_score * weight


def add_evidence_to_sections(
    run,
    sections,
):
    for section in sections:
        section["evidence"] = get_representative_evidence(
            run=run,
            issue_ids=[
                issue["id"]
                for issue in section["issues"]
            ],
            theme_ids=[
                theme["id"]
                for theme in section["themes"]
            ],
            limit=EVIDENCE_LIMIT,
        )


def normalize_priority_scores(sections):
    max_raw_score = max(
        (
            section["priority_score"]
            for section in sections
        ),
        default=0,
    )

    for section in sections:
        raw_score = section["priority_score"]

        section["raw_priority_score"] = round(
            raw_score,
            2,
        )

        section["priority_score"] = (
            round((raw_score / max_raw_score) * 100)
            if max_raw_score > 0
            else 0
        )


def get_representative_evidence(
    run,
    issue_ids=None,
    theme_ids=None,
    limit=3,
):
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
            .order_by(
                "entry_analysis__sentiment_score"
            )[:limit]
        )

        for assignment in issue_assignments:
            evidence.append(
                build_evidence_item(
                    assignment=assignment,
                    evidence_type="issue",
                    label=assignment.issue.name,
                )
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
            .order_by(
                "entry_analysis__sentiment_score"
            )[:remaining]
        )

        for assignment in theme_assignments:
            evidence.append(
                build_evidence_item(
                    assignment=assignment,
                    evidence_type="theme",
                    label=assignment.theme.name,
                )
            )

    return evidence


def build_evidence_item(
    assignment,
    evidence_type,
    label,
):
    entry_analysis = assignment.entry_analysis

    return {
        "type": evidence_type,
        "label": label,
        "entry_id": entry_analysis.entry_id,
        "sentiment_score": entry_analysis.sentiment_score,
        "excerpt": entry_analysis.entry.content[:500],
        "rationale": assignment.rationale,
    }