from django.db import transaction
from django.db.models import Count

from studies.models import (
	Study,
	EntryEvaluationTheme,
	EntryEvaluationIssue,
	StudyEvaluationTheme,
	StudyEvaluationIssue,
)


@transaction.atomic
def consolidate_themes_on_study(study):
	if isinstance(study, int):
		study = Study.objects.get(pk=study)

	theme_rows = (
		EntryEvaluationTheme.objects
		.filter(entry_evaluation__entry__study=study)
		.values("theme_id")
		.annotate(entry_count=Count("entry_evaluation__entry_id", distinct=True))
	)

	seen_theme_ids = set()

	for row in theme_rows:
		theme_id = row["theme_id"]
		seen_theme_ids.add(theme_id)

		StudyEvaluationTheme.objects.update_or_create(
			study=study,
			theme_id=theme_id,
			defaults={
				"entry_count": row["entry_count"],
				"average_confidence_score": None,
			},
		)

	StudyEvaluationTheme.objects.filter(study=study).exclude(
		theme_id__in=seen_theme_ids
	).delete()

	issue_rows = (
		EntryEvaluationIssue.objects
		.filter(entry_evaluation__entry__study=study)
		.values("issue_id")
		.annotate(entry_count=Count("entry_evaluation__entry_id", distinct=True))
	)

	seen_issue_ids = set()

	for row in issue_rows:
		issue_id = row["issue_id"]
		seen_issue_ids.add(issue_id)

		StudyEvaluationIssue.objects.update_or_create(
			study=study,
			issue_id=issue_id,
			defaults={
				"entry_count": row["entry_count"],
				"average_confidence_score": None,
			},
		)

	StudyEvaluationIssue.objects.filter(study=study).exclude(
		issue_id__in=seen_issue_ids
	).delete()

	return study