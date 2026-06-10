from dataclasses import dataclass
from django.db import transaction

from studies.models import DiaryEntryAnalysis, ConfusionMatrixOutcome, SentimentCategory

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

@transaction.atomic
def refresh_sentiment_confusion_matrix_for_run (study_analysis_run):

    entry_analyses = (
        DiaryEntryAnalysis.objects
        .select_related("entry")
        .filter(run=study_analysis_run)
    )

    participant_outcomes = []
    evaluator_outcomes = []

    for entry_analysis in entry_analyses:
        machine_sentiment_label = entry_analysis.sentiment_label

        participant_outcome = get_confusion_matrix_outcome(
            predicted_label=machine_sentiment_label,
            reference_label=entry_analysis.entry.sentiment_self_report
        )

        evaluator_outcome = get_confusion_matrix_outcome(
            predicted_label=machine_sentiment_label,
            reference_label=entry_analysis.evaluator_sentiment_label
        )

        entry_analysis.participant_confusion_matrix_outcome = participant_outcome
        entry_analysis.evaluator_confusion_matrix_outcome = evaluator_outcome

        ### NOTE TO SELF: REMOVE THIS LINE AFTER ANALYZING ALL STUDIES... 
        # THIS FIXES THE GAP FOR OLD STUDIES
        # ALSO --> MAKE SURE I ADDED IT TO analysis runner so future runs have this field
        # THEN UPDATE THE LINE ABOVE AND DON'T PICK THE PARTICIPANT REFERENCE FROM THE ENTRY
        # IMPORTANT IMPORTANT IMPORTANT IMPORTANT VERY IMPORTANT
        entry_analysis.participant_sentiment_label = entry_analysis.entry.sentiment_self_report
        ### DID I STUTTER??? I SAID IMPORTANT #FIXLATER

        entry_analysis.save(
            update_fields=[
                "participant_confusion_matrix_outcome",
                "evaluator_confusion_matrix_outcome",
                "participant_sentiment_label", # ALSO REMOVE THIS LINE #FIXLATER
            ]
        )

        # Carry all outcomes for calc at study level
        participant_outcomes.append(participant_outcome)
        evaluator_outcomes.append(evaluator_outcome)

    participant_metrics = calculate_metrics_from_outcomes (participant_outcomes)

    study_analysis_run.participant_sentiment_true_positives = participant_metrics.TPs
    study_analysis_run.participant_sentiment_false_positives = participant_metrics.FPs
    study_analysis_run.participant_sentiment_true_negatives = participant_metrics.TNs
    study_analysis_run.participant_sentiment_false_negatives = participant_metrics.FNs
    study_analysis_run.run_accuracy_v_participant = participant_metrics.accuracy
    study_analysis_run.run_precision_v_participant = participant_metrics.precision
    study_analysis_run.run_recall_v_participant = participant_metrics.recall
    study_analysis_run.run_f1_v_participant = participant_metrics.f1

    evaluator_metrics = calculate_metrics_from_outcomes (evaluator_outcomes)
    
    study_analysis_run.evaluator_sentiment_true_positives = evaluator_metrics.TPs
    study_analysis_run.evaluator_sentiment_false_positives = evaluator_metrics.FPs
    study_analysis_run.evaluator_sentiment_true_negatives = evaluator_metrics.TNs
    study_analysis_run.evaluator_sentiment_false_negatives = evaluator_metrics.FNs
    study_analysis_run.run_accuracy_v_evaluator = evaluator_metrics.accuracy
    study_analysis_run.run_precision_v_evaluator = evaluator_metrics.precision
    study_analysis_run.run_recall_v_evaluator = evaluator_metrics.recall
    study_analysis_run.run_f1_v_evaluator = evaluator_metrics.f1

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

            "evaluator_sentiment_true_positives",
            "evaluator_sentiment_false_positives",
            "evaluator_sentiment_true_negatives",
            "evaluator_sentiment_false_negatives",
            "run_accuracy_v_evaluator",
            "run_precision_v_evaluator",
            "run_recall_v_evaluator",
            "run_f1_v_evaluator",

        ]
    )

    return {
        "participant": participant_metrics,
        "evaluator": evaluator_metrics,
    }


    