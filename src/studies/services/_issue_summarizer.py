from django.db import transaction
from django.db.models import Count

from studies.models import (
	Study,
	EntryEvaluationIssue,
	StudyEvaluationIssue,
)


@transaction.atomic
def consolidate_issues_on_study(study):
	if isinstance(study, int):
		study = Study.objects.get(pk=study)

	issue_rows = (
		EntryEvaluationIssue.objects
		.filter(entry_evaluation__entry__study=study)
		.values("issue_id")
		.annotate(
			entry_count=Count("entry_evaluation__entry_id", distinct=True),
		)
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
