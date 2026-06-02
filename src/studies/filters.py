from django.core.paginator import Paginator
from django.db.models import Q

from .models import DiaryEntry


ALLOWED_ENTRY_SORTS = {
	"created_at",
	"-created_at",
	"participant_display_name",
	"-participant_display_name",
	"sentiment_self_report",
	"-sentiment_self_report",
	"machine_sentiment_label",
	"-machine_sentiment_label",
}


def filter_diary_entries(request, study, run=None):
	q = request.GET.get("q", "").strip()
	participant = request.GET.get("participant", "").strip()
	date_from = request.GET.get("date_from", "").strip()
	date_to = request.GET.get("date_to", "").strip()
	reported_sentiment = request.GET.get("reported_sentiment", "").strip()
	detected_sentiment = request.GET.get("detected_sentiment", "").strip()
	theme = request.GET.get("theme", "").strip()
	issue = request.GET.get("issue", "").strip()
	sort = request.GET.get("sort", "-created_at")

	entries = (
		DiaryEntry.objects
		.filter(study=study)
		.select_related("participant")
	)

	# General search
	if q:
		entries = entries.filter(
			Q(content__icontains=q)
			| Q(participant_display_name__icontains=q)
			| Q(participant_external_id__icontains=q)
			| Q(participant_email__icontains=q)
			| Q(source__icontains=q)
			| Q(sentiment_self_report__icontains=q)
			| Q(machine_sentiment_label__icontains=q)
			| Q(machine_theme_label__icontains=q)
			| Q(entry_summary__icontains=q)
			| Q(analysis_issue_tags__icontains=q)
		)

	# Participant search
	if participant:
		entries = entries.filter(
			Q(participant_display_name__icontains=participant)
			| Q(participant_external_id__icontains=participant)
			| Q(participant_email__icontains=participant)
			| Q(participant__username__icontains=participant)
			| Q(participant__first_name__icontains=participant)
			| Q(participant__last_name__icontains=participant)
			| Q(participant__email__icontains=participant)
		)

	# Date range
	if date_from:
		entries = entries.filter(created_at__date__gte=date_from)

	if date_to:
		entries = entries.filter(created_at__date__lte=date_to)

	# Self-reported sentiment
	if reported_sentiment:
		entries = entries.filter(sentiment_self_report=reported_sentiment)

	# Latest machine sentiment cached on DiaryEntry
	if detected_sentiment:
		entries = entries.filter(machine_sentiment_label=detected_sentiment)

	# Latest machine theme cached on DiaryEntry
	if theme:
		entries = entries.filter(machine_theme_label=theme)

	# Latest issue detection cached on DiaryEntry
	if issue == "true":
		entries = entries.filter(analysis_issue_detected=True)

	elif issue == "false":
		entries = entries.filter(
			Q(analysis_issue_detected=False)
			| Q(analysis_issue_detected__isnull=True)
		)

	# Sorting
	if sort not in ALLOWED_ENTRY_SORTS:
		sort = "-created_at"

	entries = entries.order_by(sort)

	# Theme dropdown options
	theme_options = (
		DiaryEntry.objects
		.filter(study=study)
		.exclude(machine_theme_label__isnull=True)
		.exclude(machine_theme_label="")
		.order_by("machine_theme_label")
		.values_list("machine_theme_label", flat=True)
		.distinct()
	)

	# Pagination
	paginator = Paginator(entries, 10)
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	page_params = request.GET.copy()
	page_params.pop("page", None)
	page_params.pop("sort", None)

	# Display # enrties out of total
	displayed_entries_count = page_obj.object_list.count()
	total_filtered_entries_count = paginator.count
	total_study_entries_count = (
		DiaryEntry.objects
		.filter(study=study)
		.count()
	)

	return {
		"diary_entries": page_obj.object_list,
		"page_obj": page_obj,
		"page_params": page_params.urlencode(),
		"q": q,

		"participant": participant,
		"date_from": date_from,
		"date_to": date_to,

		"reported_sentiment": reported_sentiment,
		"detected_sentiment": detected_sentiment,
		
		"theme": theme,
		"theme_options": theme_options,
		
		"issue": issue,
		"sort": sort,

		"displayed_entries_count": displayed_entries_count,
		"total_filtered_entries_count": total_filtered_entries_count,
		"total_study_entries_count": total_study_entries_count,
	}