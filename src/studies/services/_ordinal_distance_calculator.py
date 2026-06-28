from django.db import transaction

from studies.models import (
	BinarySentimentCategory,
	SentimentCategory,
	StudyAnalysis,
)


SENTIMENT_LABEL_ORDINALS = {
	SentimentCategory.VERY_NEGATIVE: 1,
	SentimentCategory.NEGATIVE: 2,
	SentimentCategory.NEUTRAL: 3,
	SentimentCategory.POSITIVE: 4,
	SentimentCategory.VERY_POSITIVE: 5,
}


BINARY_SENTIMENT_LABEL_ORDINALS = {
	BinarySentimentCategory.NEGATIVE: 1,
	BinarySentimentCategory.NOT_NEGATIVE: 2,
}


STUDY_ORDINAL_DISTANCE_UPDATE_FIELDS = [
	"participant_abs_distance_average_sentiment",
	"participant_abs_distance_dominant_sentiment",
	"evaluator_abs_distance_average_sentiment",
	"evaluator_abs_distance_dominant_sentiment",
]


def _absolute_ordinal_distance(detected_label, reference_label):
	if not detected_label or not reference_label:
		return None

	if (
		detected_label in BINARY_SENTIMENT_LABEL_ORDINALS
		or reference_label in BINARY_SENTIMENT_LABEL_ORDINALS
	):
		detected_ordinal = BINARY_SENTIMENT_LABEL_ORDINALS.get(detected_label)
		reference_ordinal = BINARY_SENTIMENT_LABEL_ORDINALS.get(reference_label)

		if detected_ordinal is None or reference_ordinal is None:
			return None

		return abs(detected_ordinal - reference_ordinal)

	detected_ordinal = SENTIMENT_LABEL_ORDINALS.get(detected_label)
	reference_ordinal = SENTIMENT_LABEL_ORDINALS.get(reference_label)

	if detected_ordinal is None or reference_ordinal is None:
		return None

	return abs(detected_ordinal - reference_ordinal)


@transaction.atomic
def refresh_sentiment_ordinal_distance_for_run(study_analysis):

	if isinstance(study_analysis, int):
		study_analysis = StudyAnalysis.objects.get(pk=study_analysis)

	study_analysis.participant_abs_distance_average_sentiment = _absolute_ordinal_distance(
		study_analysis.average_sentiment_label,
		study_analysis.participant_average_sentiment_label,
	)

	study_analysis.participant_abs_distance_dominant_sentiment = _absolute_ordinal_distance(
		study_analysis.dominant_sentiment_label,
		study_analysis.participant_dominant_sentiment_label,
	)

	study_analysis.evaluator_abs_distance_average_sentiment = _absolute_ordinal_distance(
		study_analysis.average_sentiment_label,
		study_analysis.evaluator_average_sentiment_label,
	)

	study_analysis.evaluator_abs_distance_dominant_sentiment = _absolute_ordinal_distance(
		study_analysis.dominant_sentiment_label,
		study_analysis.evaluator_dominant_sentiment_label,
	)

	study_analysis.save(update_fields=STUDY_ORDINAL_DISTANCE_UPDATE_FIELDS)

	return study_analysis