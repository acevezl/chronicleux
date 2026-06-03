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
	"selected_entry_run__sentiment_label",
	"-selected_entry_run__sentiment_label",
	"selected_entry_run__theme_label",
	"-selected_entry_run__theme_label",
	"selected_entry_run__issues",
	"-selected_entry_run__issues",
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
		.select_related("participant", "selected_entry_run")
	)

	# General search
	if q:
		entries = entries.filter(
			Q(content__icontains=q)
			| Q(participant_display_name__icontains=q)
			| Q(participant_external_id__icontains=q)
			| Q(participant_email__icontains=q)
			| Q(source__icontains=q)
			| Q(content__icontains=q)
			| Q(sentiment_self_report__icontains=q)
			| Q(selected_entry_run__sentiment_label__icontains=q)
			| Q(selected_entry_run__theme_label__icontains=q)
			| Q(selected_entry_run__issues__icontains=q)
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
		entries = entries.filter(selected_entry_run__sentiment_label=detected_sentiment)

	# Latest machine theme cached on DiaryEntry
	if theme:
		entries = entries.filter(selected_entry_run__theme_label=theme)

	# Sorting
	if sort not in ALLOWED_ENTRY_SORTS:
		sort = "-created_at"

	entries = entries.order_by(sort)

	# Latest issue detection cached on DiaryEntry
	if issue:
		entries = [
			entry for entry in entries
			if entry.selected_entry_run
			and issue in (entry.selected_entry_run.issues or [])
		]

	# Theme dropdown options
	theme_options = (
		DiaryEntry.objects
		.filter(study=study)
		.exclude(selected_entry_run__theme_label__isnull=True)
		.exclude(selected_entry_run__theme_label="")
		.order_by("selected_entry_run__theme_label")
		.values_list("selected_entry_run__theme_label", flat=True)
		.distinct()
	)

	# Issue dropdown options
	issue_values = (
		DiaryEntry.objects
		.filter(study=study)
		.exclude(selected_entry_run__issues__isnull=True)
		.values_list("selected_entry_run__issues", flat=True)
	)

	issue_options = sorted({
		issue
		for issues in issue_values
		for issue in issues
		if issue
	})

	# Pagination
	paginator = Paginator(entries, 10)
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	page_params = request.GET.copy()
	page_params.pop("page", None)
	page_params.pop("sort", None)

	# Display # enrties out of total
	displayed_entries_count = len(page_obj.object_list)
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
		"issue_options": issue_options,

		"sort": sort,

		"displayed_entries_count": displayed_entries_count,
		"total_filtered_entries_count": total_filtered_entries_count,
		"total_study_entries_count": total_study_entries_count,
	}