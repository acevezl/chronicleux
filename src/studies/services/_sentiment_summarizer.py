from collections import Counter

from django.db import transaction

from studies.models import (
	BinarySentimentCategory,
	BINARY_SENTIMENT_SCORE_THRESHOLDS,
	SENTIMENT_SCORE_THRESHOLDS,
	Study,
	Entry,
	EntryEvaluation,
)


STUDY_SENTIMENT_UPDATE_FIELDS = [
	"participant_reported_average_sentiment_score",
	"participant_reported_average_sentiment_label",
	"participant_reported_dominant_sentiment_score",
	"participant_reported_dominant_sentiment_label",
	"participant_sentiment_distribution",

	"evaluator_average_sentiment_score",
	"evaluator_average_sentiment_label",
	"evaluator_dominant_sentiment_score",
	"evaluator_dominant_sentiment_label",
	"evaluator_sentiment_distribution",

	"updated_at",
]


def _get_thresholds_for_labels(labels):
	if BinarySentimentCategory.NOT_NEGATIVE in labels:
		return BINARY_SENTIMENT_SCORE_THRESHOLDS

	return SENTIMENT_SCORE_THRESHOLDS


def _label_to_score(label, thresholds):
	for min_score, max_score, threshold_label in thresholds:
		if threshold_label == label:
			return round((min_score + max_score) / 2, 4)

	return None


def _score_to_label(score, thresholds):
	if score is None:
		return None

	for min_score, max_score, label in thresholds:
		if min_score <= score <= max_score:
			return label

	return None


def _build_sentiment_distribution(labels):
	labels = [label for label in labels if label]
	total = len(labels)

	if total == 0:
		return []

	counts = Counter(labels)

	return [
		{
			"label": label,
			"count": count,
			"percentage": round(count / total, 4),
		}
		for label, count in counts.most_common()
	]


def _summarize_sentiment_labels(labels):
	labels = [label for label in labels if label]

	if not labels:
		return {
			"average_score": None,
			"average_label": None,
			"dominant_score": None,
			"dominant_label": None,
		}

	thresholds = _get_thresholds_for_labels(labels)

	scores = [
		_label_to_score(label, thresholds)
		for label in labels
	]

	scores = [
		score
		for score in scores
		if score is not None
	]

	if not scores:
		return {
			"average_score": None,
			"average_label": None,
			"dominant_score": None,
			"dominant_label": None,
		}

	average_score = round(sum(scores) / len(scores), 4)
	average_label = _score_to_label(average_score, thresholds)

	dominant_label = Counter(labels).most_common(1)[0][0]
	dominant_score = _label_to_score(dominant_label, thresholds)

	return {
		"average_score": average_score,
		"average_label": average_label,
		"dominant_score": dominant_score,
		"dominant_label": dominant_label,
	}


@transaction.atomic
def summarize_study_sentiment(study):
	if isinstance(study, int):
		study = Study.objects.get(pk=study)

	participant_labels = list(
		Entry.objects
		.filter(study=study)
		.exclude(sentiment_self_report__isnull=True)
		.exclude(sentiment_self_report="")
		.values_list("sentiment_self_report", flat=True)
	)

	evaluator_labels = list(
		EntryEvaluation.objects
		.filter(entry__study=study)
		.exclude(evaluator_sentiment_label__isnull=True)
		.exclude(evaluator_sentiment_label="")
		.values_list("evaluator_sentiment_label", flat=True)
	)

	participant_summary = _summarize_sentiment_labels(participant_labels)
	evaluator_summary = _summarize_sentiment_labels(evaluator_labels)

	study.participant_reported_average_sentiment_score = participant_summary["average_score"]
	study.participant_reported_average_sentiment_label = participant_summary["average_label"]
	study.participant_reported_dominant_sentiment_score = participant_summary["dominant_score"]
	study.participant_reported_dominant_sentiment_label = participant_summary["dominant_label"]
	study.participant_sentiment_distribution = _build_sentiment_distribution(participant_labels)

	study.evaluator_average_sentiment_score = evaluator_summary["average_score"]
	study.evaluator_average_sentiment_label = evaluator_summary["average_label"]
	study.evaluator_dominant_sentiment_score = evaluator_summary["dominant_score"]
	study.evaluator_dominant_sentiment_label = evaluator_summary["dominant_label"]
	study.evaluator_sentiment_distribution = _build_sentiment_distribution(evaluator_labels)

	study.save(update_fields=STUDY_SENTIMENT_UPDATE_FIELDS)

	return study


# In case I want to summarize all sentiments on all studies
def summarize_all_sentiments_on_all_studies():
	study_ids = (
		Study.objects
		.filter(entries__isnull=False)
		.values_list("pk", flat=True)
		.distinct()
	)

	for study_id in study_ids:
		summarize_study_sentiment(study_id)