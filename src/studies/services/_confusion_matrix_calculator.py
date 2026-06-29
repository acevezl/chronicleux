from dataclasses import dataclass

from django.db import transaction
from django.db.models import Prefetch

from studies.models import (
	BinarySentimentCategory,
	ConfusionMatrixOutcome,
	EntryAnalysis,
	EntryEvaluation,
	SENTIMENT_CHOICES,
	SentimentCategory,
	StudyAnalysis,
	Study,
)


# Target class for the binary confusion matrix:
# "negative" vs "not negative".
NEGATIVE_SENTIMENTS = {
	SentimentCategory.VERY_NEGATIVE,
	SentimentCategory.NEGATIVE,
	BinarySentimentCategory.NEGATIVE,
}


SENTIMENT_LABELS = [
	label
	for label, _display_name in SENTIMENT_CHOICES
]


@dataclass
class BinaryMetrics:
	TPs: int = 0
	FPs: int = 0
	TNs: int = 0
	FNs: int = 0
	accuracy: float | None = None
	precision: float | None = None
	recall: float | None = None
	f1: float | None = None


STUDY_ANALYSIS_CONFUSION_MATRIX_UPDATE_FIELDS = [
	"participant_sentiment_true_positives",
	"participant_sentiment_false_positives",
	"participant_sentiment_true_negatives",
	"participant_sentiment_false_negatives",
	"run_accuracy_v_participant",
	"run_precision_v_participant",
	"run_recall_v_participant",
	"run_f1_v_participant",
	"participant_sentiment_metrics_by_category",

	"evaluator_sentiment_true_positives",
	"evaluator_sentiment_false_positives",
	"evaluator_sentiment_true_negatives",
	"evaluator_sentiment_false_negatives",
	"run_accuracy_v_evaluator",
	"run_precision_v_evaluator",
	"run_recall_v_evaluator",
	"run_f1_v_evaluator",
	"evaluator_sentiment_metrics_by_category",
]


ENTRY_ANALYSIS_CONFUSION_MATRIX_UPDATE_FIELDS = [
	"participant_sentiment_label",
	"participant_confusion_matrix_outcome",
	"evaluator_confusion_matrix_outcome",
]


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
	if denominator == 0:
		return None

	return round(numerator / denominator, 4)


def calculate_binary_metrics(
	true_positives: int,
	false_positives: int,
	true_negatives: int,
	false_negatives: int,
) -> BinaryMetrics:
	total = true_positives + false_positives + true_negatives + false_negatives

	accuracy = safe_divide(true_positives + true_negatives, total)
	precision = safe_divide(true_positives, true_positives + false_positives)
	recall = safe_divide(true_positives, true_positives + false_negatives)

	if precision is None or recall is None or (precision + recall) == 0:
		f1 = None
	else:
		f1 = round(2 * precision * recall / (precision + recall), 4)

	return BinaryMetrics(
		TPs=true_positives,
		FPs=false_positives,
		TNs=true_negatives,
		FNs=false_negatives,
		accuracy=accuracy,
		precision=precision,
		recall=recall,
		f1=f1,
	)


def calculate_metrics_from_outcomes(outcomes: list[str]) -> BinaryMetrics:
	true_positives = outcomes.count(ConfusionMatrixOutcome.TRUE_POSITIVE)
	false_positives = outcomes.count(ConfusionMatrixOutcome.FALSE_POSITIVE)
	true_negatives = outcomes.count(ConfusionMatrixOutcome.TRUE_NEGATIVE)
	false_negatives = outcomes.count(ConfusionMatrixOutcome.FALSE_NEGATIVE)

	return calculate_binary_metrics(
		true_positives=true_positives,
		false_positives=false_positives,
		true_negatives=true_negatives,
		false_negatives=false_negatives,
	)


def is_negative_sentiment(label: str | None) -> bool | None:
	if not label:
		return None

	return label in NEGATIVE_SENTIMENTS


def get_confusion_matrix_outcome(
	predicted_label: str | None,
	reference_label: str | None,
) -> str:
	predicted_is_negative = is_negative_sentiment(predicted_label)
	reference_is_negative = is_negative_sentiment(reference_label)

	if predicted_is_negative is None or reference_is_negative is None:
		return ConfusionMatrixOutcome.NOT_AVAILABLE

	if predicted_is_negative and reference_is_negative:
		return ConfusionMatrixOutcome.TRUE_POSITIVE

	if predicted_is_negative and not reference_is_negative:
		return ConfusionMatrixOutcome.FALSE_POSITIVE

	if not predicted_is_negative and reference_is_negative:
		return ConfusionMatrixOutcome.FALSE_NEGATIVE

	return ConfusionMatrixOutcome.TRUE_NEGATIVE


def get_category_confusion_matrix_outcome(
	predicted_label: str | None,
	reference_label: str | None,
	target_label: str,
) -> str:
	if not predicted_label or not reference_label:
		return ConfusionMatrixOutcome.NOT_AVAILABLE

	predicted_is_target = predicted_label == target_label
	reference_is_target = reference_label == target_label

	if predicted_is_target and reference_is_target:
		return ConfusionMatrixOutcome.TRUE_POSITIVE

	if predicted_is_target and not reference_is_target:
		return ConfusionMatrixOutcome.FALSE_POSITIVE

	if not predicted_is_target and reference_is_target:
		return ConfusionMatrixOutcome.FALSE_NEGATIVE

	return ConfusionMatrixOutcome.TRUE_NEGATIVE


def calculate_metrics_by_sentiment_category(
	label_pairs: list[tuple[str | None, str | None]],
) -> dict:
	metrics_by_category = {}

	for target_label in SENTIMENT_LABELS:
		category_outcomes = [
			get_category_confusion_matrix_outcome(
				predicted_label=predicted_label,
				reference_label=reference_label,
				target_label=target_label,
			)
			for predicted_label, reference_label in label_pairs
		]

		category_metrics = calculate_metrics_from_outcomes(category_outcomes)

		metrics_by_category[target_label] = {
			"tp": category_metrics.TPs,
			"fp": category_metrics.FPs,
			"tn": category_metrics.TNs,
			"fn": category_metrics.FNs,
			"accuracy": category_metrics.accuracy,
			"precision": category_metrics.precision,
			"recall": category_metrics.recall,
			"f1": category_metrics.f1,
		}

	return metrics_by_category


def get_prefetched_entry_evaluations(entry) -> list[EntryEvaluation]:
	prefetched_evaluations = getattr(
		entry,
		"prefetched_entry_evaluations",
		None,
	)

	if prefetched_evaluations is not None:
		return list(prefetched_evaluations)

	return list(
		entry.entry_evaluations
		.order_by("-updated_at", "-created_at")
	)


def get_latest_evaluator_outcome(
	machine_sentiment_label: str | None,
	entry,
) -> str:
	entry_evaluations = get_prefetched_entry_evaluations(entry)

	if not entry_evaluations:
		return ConfusionMatrixOutcome.NOT_AVAILABLE

	latest_entry_evaluation = entry_evaluations[0]

	return get_confusion_matrix_outcome(
		predicted_label=machine_sentiment_label,
		reference_label=latest_entry_evaluation.evaluator_sentiment_label,
	)


@transaction.atomic
def refresh_sentiment_confusion_matrix_for_run(study_analysis: StudyAnalysis):

	if isinstance(study_analysis, int):
		study_analysis = StudyAnalysis.objects.get(pk=study_analysis)

	entry_analyses = (
		EntryAnalysis.objects
		.select_related("entry")
		.prefetch_related(
			Prefetch(
				"entry__entry_evaluations",
				queryset=EntryEvaluation.objects.order_by(
					"-updated_at",
					"-created_at",
				),
				to_attr="prefetched_entry_evaluations",
			)
		)
		.filter(run=study_analysis)
	)

	participant_outcomes = []
	evaluator_outcomes = []

	participant_label_pairs = []
	evaluator_label_pairs = []

	entry_analyses_to_update = []

	for entry_analysis in entry_analyses:
		machine_sentiment_label = entry_analysis.sentiment_label

		participant_reference_label = entry_analysis.entry.sentiment_self_report

		participant_outcome = get_confusion_matrix_outcome(
			predicted_label=machine_sentiment_label,
			reference_label=participant_reference_label,
		)

		entry_analysis.participant_sentiment_label = participant_reference_label
		entry_analysis.participant_confusion_matrix_outcome = participant_outcome

		participant_outcomes.append(participant_outcome)
		participant_label_pairs.append(
			(machine_sentiment_label, participant_reference_label)
		)

		entry_evaluations = get_prefetched_entry_evaluations(entry_analysis.entry)

		for entry_evaluation in entry_evaluations:
			evaluator_reference_label = entry_evaluation.evaluator_sentiment_label

			evaluator_outcome = get_confusion_matrix_outcome(
				predicted_label=machine_sentiment_label,
				reference_label=evaluator_reference_label,
			)

			evaluator_outcomes.append(evaluator_outcome)
			evaluator_label_pairs.append(
				(machine_sentiment_label, evaluator_reference_label)
			)

		entry_analysis.evaluator_confusion_matrix_outcome = get_latest_evaluator_outcome(
			machine_sentiment_label=machine_sentiment_label,
			entry=entry_analysis.entry,
		)

		entry_analyses_to_update.append(entry_analysis)

	if entry_analyses_to_update:
		EntryAnalysis.objects.bulk_update(
			entry_analyses_to_update,
			ENTRY_ANALYSIS_CONFUSION_MATRIX_UPDATE_FIELDS,
		)

	participant_metrics = calculate_metrics_from_outcomes(participant_outcomes)
	participant_metrics_by_category = calculate_metrics_by_sentiment_category(
		participant_label_pairs
	)

	study_analysis.participant_sentiment_true_positives = participant_metrics.TPs
	study_analysis.participant_sentiment_false_positives = participant_metrics.FPs
	study_analysis.participant_sentiment_true_negatives = participant_metrics.TNs
	study_analysis.participant_sentiment_false_negatives = participant_metrics.FNs
	study_analysis.run_accuracy_v_participant = participant_metrics.accuracy
	study_analysis.run_precision_v_participant = participant_metrics.precision
	study_analysis.run_recall_v_participant = participant_metrics.recall
	study_analysis.run_f1_v_participant = participant_metrics.f1
	study_analysis.participant_sentiment_metrics_by_category = participant_metrics_by_category

	evaluator_metrics = calculate_metrics_from_outcomes(evaluator_outcomes)
	evaluator_metrics_by_category = calculate_metrics_by_sentiment_category(
		evaluator_label_pairs
	)

	study_analysis.evaluator_sentiment_true_positives = evaluator_metrics.TPs
	study_analysis.evaluator_sentiment_false_positives = evaluator_metrics.FPs
	study_analysis.evaluator_sentiment_true_negatives = evaluator_metrics.TNs
	study_analysis.evaluator_sentiment_false_negatives = evaluator_metrics.FNs
	study_analysis.run_accuracy_v_evaluator = evaluator_metrics.accuracy
	study_analysis.run_precision_v_evaluator = evaluator_metrics.precision
	study_analysis.run_recall_v_evaluator = evaluator_metrics.recall
	study_analysis.run_f1_v_evaluator = evaluator_metrics.f1
	study_analysis.evaluator_sentiment_metrics_by_category = evaluator_metrics_by_category

	study_analysis.save(
		update_fields=STUDY_ANALYSIS_CONFUSION_MATRIX_UPDATE_FIELDS
	)

	return {
		"participant": participant_metrics,
		"participant_by_category": participant_metrics_by_category,
		"evaluator": evaluator_metrics,
		"evaluator_by_category": evaluator_metrics_by_category,
	}