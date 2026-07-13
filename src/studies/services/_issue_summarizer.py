from collections import defaultdict

from django.db import transaction
from django.db.models import Count

from studies.models import (
	Study,
	EntryEvaluationIssue,
	StudyEvaluationIssue,
)


EVALUATOR_SENTIMENT_SCORES = {
	"VERY_NEGATIVE": -1.0,
	"NEGATIVE": -0.5,
	"NEUTRAL": 0.0,
	"POSITIVE": 0.5,
	"VERY_POSITIVE": 1.0,
	"NOT_NEGATIVE": 0.5,
}


@transaction.atomic
def consolidate_issues_on_study(study):
	if isinstance(study, int):
		study = Study.objects.get(pk=study)

	assignments = (
		EntryEvaluationIssue.objects
		.filter(entry_evaluation__entry__study=study)
		.select_related(
			"entry_evaluation",
			"issue",
		)
	)

	grouped = defaultdict(
		lambda: {
			"entry_ids": set(),
			"sentiment_scores": [],
		}
	)

	for assignment in assignments:
		issue_id = assignment.issue_id
		entry_evaluation = assignment.entry_evaluation

		grouped[issue_id]["entry_ids"].add(
			entry_evaluation.entry_id
		)

		sentiment_label = entry_evaluation.evaluator_sentiment_label
		sentiment_score = EVALUATOR_SENTIMENT_SCORES.get(
			sentiment_label
		)

		if sentiment_score is not None:
			grouped[issue_id]["sentiment_scores"].append(
				sentiment_score
			)

	seen_issue_ids = set()

	for issue_id, data in grouped.items():
		seen_issue_ids.add(issue_id)

		sentiment_scores = data["sentiment_scores"]

		average_sentiment_score = (
			round(
				sum(sentiment_scores) / len(sentiment_scores),
				4,
			)
			if sentiment_scores
			else None
		)

		StudyEvaluationIssue.objects.update_or_create(
			study=study,
			issue_id=issue_id,
			defaults={
				"entry_count": len(data["entry_ids"]),
				"average_confidence_score": None,
				"average_sentiment_score": average_sentiment_score,
			},
		)

	StudyEvaluationIssue.objects.filter(
		study=study,
	).exclude(
		issue_id__in=seen_issue_ids,
	).delete()

	return study