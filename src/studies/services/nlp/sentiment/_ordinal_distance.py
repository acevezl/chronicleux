from __future__ import annotations

from collections import Counter
from typing import Iterable

from studies.models import DiaryEntryAnalysis, SentimentCategory, StudyAnalysisRun
from studies.services.nlp.sentiment._thresholds import map_sentiment_score_to_label


SENTIMENT_LABEL_SCORES = {
	SentimentCategory.VERY_NEGATIVE: -0.85,
	SentimentCategory.NEGATIVE: -0.45,
	SentimentCategory.NEUTRAL: 0.0,
	SentimentCategory.POSITIVE: 0.45,
	SentimentCategory.VERY_POSITIVE: 0.85,
}


SENTIMENT_LABEL_ORDINALS = {
	SentimentCategory.VERY_NEGATIVE: 1,
	SentimentCategory.NEGATIVE: 2,
	SentimentCategory.NEUTRAL: 3,
	SentimentCategory.POSITIVE: 4,
	SentimentCategory.VERY_POSITIVE: 5,
}


def sentiment_label_to_score(label: str | None) -> float | None:
	if not label:
		return None

	return SENTIMENT_LABEL_SCORES.get(label)


def sentiment_label_to_ordinal(label: str | None) -> int | None:
	if not label:
		return None

	return SENTIMENT_LABEL_ORDINALS.get(label)


def average_sentiment_score(labels: Iterable[str | None]) -> float | None:
	values = [
		sentiment_label_to_score(label)
		for label in labels
	]

	values = [
		value
		for value in values
		if value is not None
	]

	if not values:
		return None

	return round(sum(values) / len(values), 4)


def dominant_sentiment_label(labels: Iterable[str | None]) -> str | None:
	valid_labels = [
		label
		for label in labels
		if sentiment_label_to_score(label) is not None
	]

	if not valid_labels:
		return None

	return Counter(valid_labels).most_common(1)[0][0]


def absolute_ordinal_distance(
	reference_label: str | None,
	detected_label: str | None,
) -> int | None:
	reference_ordinal = sentiment_label_to_ordinal(reference_label)
	detected_ordinal = sentiment_label_to_ordinal(detected_label)

	if reference_ordinal is None or detected_ordinal is None:
		return None

	return abs(reference_ordinal - detected_ordinal)


def calculate_reference_sentiment_summary(labels: Iterable[str | None]) -> dict:
	labels = list(labels)

	average_score = average_sentiment_score(labels)

	average_label = (
		map_sentiment_score_to_label(average_score)
		if average_score is not None
		else None
	)

	dominant_label = dominant_sentiment_label(labels)
	dominant_score = sentiment_label_to_score(dominant_label)

	return {
		"average_score": average_score,
		"average_label": average_label,
		"dominant_score": dominant_score,
		"dominant_label": dominant_label,
	}


def refresh_sentiment_ordinal_distance_for_run(
	run: StudyAnalysisRun,
) -> StudyAnalysisRun:
	entry_analyses = (
		DiaryEntryAnalysis.objects
		.filter(run=run)
		.select_related("entry")
	)

	participant_labels = [
		entry_analysis.entry.sentiment_self_report
		for entry_analysis in entry_analyses
		if (
			entry_analysis.entry
			and entry_analysis.entry.sentiment_self_report
		)
	]

	evaluator_labels = [
		entry_analysis.evaluator_sentiment_label
		for entry_analysis in entry_analyses
		if entry_analysis.evaluator_sentiment_label
	]

	participant_summary = calculate_reference_sentiment_summary(
		participant_labels
	)

	evaluator_summary = calculate_reference_sentiment_summary(
		evaluator_labels
	)

	participant_distribution = build_sentiment_distribution_from_labels(
		participant_labels
	)

	evaluator_distribution = build_sentiment_distribution_from_labels(
		evaluator_labels
	)

	run.participant_average_sentiment_score = participant_summary["average_score"]
	run.participant_average_sentiment_label = participant_summary["average_label"]
	run.participant_dominant_sentiment_score = participant_summary["dominant_score"]
	run.participant_dominant_sentiment_label = participant_summary["dominant_label"]
	run.participant_abs_distance_average_sentiment = absolute_ordinal_distance(
		participant_summary["average_label"],
		run.average_sentiment_label,
	)
	run.participant_abs_distance_dominant_sentiment = absolute_ordinal_distance(
		participant_summary["dominant_label"],
		run.dominant_sentiment_label,
	)

	run.evaluator_average_sentiment_score = evaluator_summary["average_score"]
	run.evaluator_average_sentiment_label = evaluator_summary["average_label"]
	run.evaluator_dominant_sentiment_score = evaluator_summary["dominant_score"]
	run.evaluator_dominant_sentiment_label = evaluator_summary["dominant_label"]
	run.evaluator_abs_distance_average_sentiment = absolute_ordinal_distance(
		evaluator_summary["average_label"],
		run.average_sentiment_label,
	)
	run.evaluator_abs_distance_dominant_sentiment = absolute_ordinal_distance(
		evaluator_summary["dominant_label"],
		run.dominant_sentiment_label,
	)

	run.participant_sentiment_distribution = participant_distribution
	run.evaluator_sentiment_distribution = evaluator_distribution

	run.save(
		update_fields=[
			"participant_average_sentiment_score",
			"participant_average_sentiment_label",
			"participant_dominant_sentiment_score",
			"participant_dominant_sentiment_label",
			"participant_abs_distance_average_sentiment",
			"participant_abs_distance_dominant_sentiment",

			"evaluator_average_sentiment_score",
			"evaluator_average_sentiment_label",
			"evaluator_dominant_sentiment_score",
			"evaluator_dominant_sentiment_label",
			"evaluator_abs_distance_average_sentiment",
			"evaluator_abs_distance_dominant_sentiment",

			"participant_sentiment_distribution",
			"evaluator_sentiment_distribution",
		]
	)

	return run

def build_sentiment_distribution_from_labels(labels: Iterable[str | None]) -> list[dict]:
	"""
	Build sentiment label counts and percentages using the same shape as
	StudyAnalysisRun.sentiment_distribution.

	Output:
	[
		{"label": "POSITIVE", "count": 10, "percentage": 0.625},
		...
	]
	"""
	counts = Counter(
		label
		for label in labels
		if sentiment_label_to_score(label) is not None
	)

	total = sum(counts.values())

	return [
		{
			"label": label,
			"count": count,
			"percentage": round(count / total, 4) if total else 0,
		}
		for label, count in counts.most_common()
	]