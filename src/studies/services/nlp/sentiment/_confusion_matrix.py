from dataclasses import dataclass
from django.db import transaction
from django.db.models import Prefetch

from studies.models import (
    EntryAnalysis,
    EntryEvaluation,
    ConfusionMatrixOutcome,
    SentimentCategory,
)

# My target label to identify is "SENTIMENT=NEGATIVE"
NEGATIVE_SENTIMENTS = {
    SentimentCategory.NEGATIVE,
    SentimentCategory.VERY_NEGATIVE
}

# Metrics
@dataclass
class BinaryMetrics:
    TPs: int = 0 # count of true positives
    FPs: int = 0 # ... of false positives
    TNs: int = 0 # ... of true negatives
    FNs: int = 0 # ... of false negatives
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

# Check if a sentiment label is negative (So I don't have to do many ifs later)
def is_negative_sentiment(label: str | None) -> bool | None:
    if not label:
        return None
    
    return label in NEGATIVE_SENTIMENTS

def get_confusion_matrix_outcome (
        predicted_label: str | None,
        reference_label: str | None,
) -> str:

    predicted_is_negative = is_negative_sentiment(predicted_label)
    reference_is_negative = is_negative_sentiment(reference_label)

    if predicted_is_negative is None or reference_is_negative is None:
        return ConfusionMatrixOutcome.NOT_AVAILABLE
    
    # If SENTIMENT is NEGATIVE on both prediction and Reference, result is a True Positive.
    if predicted_is_negative and reference_is_negative:
        return ConfusionMatrixOutcome.TRUE_POSITIVE
    
    # If SENTIMENT is NEGATIVE on prediction, but POSITIVE or NEUTRAL on Reference, result is a False Positive.
    if predicted_is_negative and not reference_is_negative:
        return ConfusionMatrixOutcome.FALSE_POSITIVE
    
    # If SENTIMENT is POSITIVE or NEUTRAL on prediction, but NEGATIVE on Reference, result is a False Negative
    if not predicted_is_negative and reference_is_negative:
        return ConfusionMatrixOutcome.FALSE_NEGATIVE
    
    # If SENTIMENT is POSITIVE or NEUTRAL on both prediction and Reference
    return ConfusionMatrixOutcome.TRUE_NEGATIVE
    

def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator

def calculate_binary_metrics(
        true_positives:int, false_positives:int, 
        true_negatives:int, false_negatives:int
        )->BinaryMetrics:
    
    try: 

        total = true_positives + false_positives + true_negatives + false_negatives

        # acc = (TP + TN) / TOTAL
        accuracy = safe_divide (true_positives + true_negatives, total)

        # p = TP / (TP + FP)
        precision = safe_divide (true_positives, true_positives + false_positives)

        # recall = TP / (TP + FN)
        recall = safe_divide (true_positives, true_positives + false_negatives)

        # f1 = 2 x ( (precision x recall) / (precision + recall))
        f1 = (
            safe_divide(2 * precision * recall, precision + recall)
            if precision is not None and recall is not None
            else None
        )

    except Exception as e:
        print (e)

    return BinaryMetrics (
        TPs = true_positives,
        FPs = false_positives,
        TNs = true_negatives,
        FNs = false_negatives,
        accuracy = accuracy,
        precision = precision,
        recall = recall,
        f1 = f1,
    )

def calculate_metrics_from_outcomes (outcomes: list[str]) -> BinaryMetrics:
    TPs = outcomes.count(ConfusionMatrixOutcome.TRUE_POSITIVE)
    FPs = outcomes.count(ConfusionMatrixOutcome.FALSE_POSITIVE)
    TNs = outcomes.count(ConfusionMatrixOutcome.TRUE_NEGATIVE)
    FNs = outcomes.count(ConfusionMatrixOutcome.FALSE_NEGATIVE)
    
    return calculate_binary_metrics (
        true_positives=TPs,
        false_positives=FPs,
        true_negatives=TNs,
        false_negatives=FNs,
    )

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


def get_latest_entry_evaluation(entry):
    prefetched_evaluations = getattr(
        entry,
        "prefetched_entry_evaluations",
        None,
    )

    if prefetched_evaluations is not None:
        return prefetched_evaluations[0] if prefetched_evaluations else None

    return (
        entry.entry_evaluations
        .order_by("-updated_at", "-created_at")
        .first()
    )


def get_evaluator_reference_label(entry) -> str | None:
    entry_evaluation = get_latest_entry_evaluation(entry)

    if not entry_evaluation:
        return None

    return entry_evaluation.evaluator_sentiment_label


def calculate_metrics_by_sentiment_category(
        label_pairs: list[tuple[str | None, str | None]]
) -> dict:
    metrics_by_category = {}

    for sentiment_category in SentimentCategory:
        target_label = sentiment_category.value

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

@transaction.atomic
def refresh_sentiment_confusion_matrix_for_run (study_analysis_run):

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
        .filter(run=study_analysis_run)
    )

    participant_outcomes = []
    evaluator_outcomes = []

    participant_label_pairs = []
    evaluator_label_pairs = []

    for entry_analysis in entry_analyses:
        machine_sentiment_label = entry_analysis.sentiment_label

        participant_reference_label = entry_analysis.entry.sentiment_self_report
        evaluator_reference_label = get_evaluator_reference_label(entry_analysis.entry)

        participant_outcome = get_confusion_matrix_outcome(
            predicted_label=machine_sentiment_label,
            reference_label=participant_reference_label,
        )

        evaluator_outcome = get_confusion_matrix_outcome(
            predicted_label=machine_sentiment_label,
            reference_label=evaluator_reference_label,
        )

        entry_analysis.participant_confusion_matrix_outcome = participant_outcome
        entry_analysis.evaluator_confusion_matrix_outcome = evaluator_outcome

        entry_analysis.save(
            update_fields=[
                "participant_confusion_matrix_outcome",
                "evaluator_confusion_matrix_outcome",
            ]
        )

        # Carry all outcomes for calc at study level
        participant_outcomes.append(participant_outcome)
        evaluator_outcomes.append(evaluator_outcome)

        # And carry the pairs of machine sentiment vs reference
        # participant
        participant_label_pairs.append(
            (machine_sentiment_label, participant_reference_label)
        )
        # evaluator
        evaluator_label_pairs.append(
            (machine_sentiment_label, evaluator_reference_label)
        )

    # Calculate metrics from all outcomes - Participant
    participant_metrics = calculate_metrics_from_outcomes (participant_outcomes)

    study_analysis_run.participant_sentiment_true_positives = participant_metrics.TPs
    study_analysis_run.participant_sentiment_false_positives = participant_metrics.FPs
    study_analysis_run.participant_sentiment_true_negatives = participant_metrics.TNs
    study_analysis_run.participant_sentiment_false_negatives = participant_metrics.FNs
    study_analysis_run.run_accuracy_v_participant = participant_metrics.accuracy
    study_analysis_run.run_precision_v_participant = participant_metrics.precision
    study_analysis_run.run_recall_v_participant = participant_metrics.recall
    study_analysis_run.run_f1_v_participant = participant_metrics.f1

    # and for all categories
    participant_metrics_by_category = calculate_metrics_by_sentiment_category(
        participant_label_pairs
    )

    study_analysis_run.participant_sentiment_metrics_by_category = participant_metrics_by_category

    # Calculate metrics from all outcomes - Evaluator
    evaluator_metrics = calculate_metrics_from_outcomes (evaluator_outcomes)

    study_analysis_run.evaluator_sentiment_true_positives = evaluator_metrics.TPs
    study_analysis_run.evaluator_sentiment_false_positives = evaluator_metrics.FPs
    study_analysis_run.evaluator_sentiment_true_negatives = evaluator_metrics.TNs
    study_analysis_run.evaluator_sentiment_false_negatives = evaluator_metrics.FNs
    study_analysis_run.run_accuracy_v_evaluator = evaluator_metrics.accuracy
    study_analysis_run.run_precision_v_evaluator = evaluator_metrics.precision
    study_analysis_run.run_recall_v_evaluator = evaluator_metrics.recall
    study_analysis_run.run_f1_v_evaluator = evaluator_metrics.f1

    # and also for all categories - Evaluator
    evaluator_metrics_by_category = calculate_metrics_by_sentiment_category(
        evaluator_label_pairs
    )

    study_analysis_run.evaluator_sentiment_metrics_by_category = evaluator_metrics_by_category

    # Now savesies
    study_analysis_run.save(
        update_fields=[

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
    )

    return {
        "participant": participant_metrics,
        "participant_by_category": participant_metrics_by_category,
        "evaluator": evaluator_metrics,
        "evaluator_by_category": evaluator_metrics_by_category,
    }


    