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
	"issue_encountered",
	"-issue_encountered",
}

def filter_diary_entries(request, study, run=None):
	q = request.GET.get("q", "").strip()
	date_from = request.GET.get("date_from", "").strip()
	date_to = request.GET.get("date_to", "").strip()
	reported_sentiment = request.GET.get("reported_sentiment", "").strip()
	issue_encountered = request.GET.get("issue_encountered", "").strip()
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
		)

	# Date range
	if date_from:
		entries = entries.filter(created_at__date__gte=date_from)

	if date_to:
		entries = entries.filter(created_at__date__lte=date_to)

	# Self-reported sentiment
	if reported_sentiment:
		entries = entries.filter(sentiment_self_report=reported_sentiment)

	if issue_encountered == "yes":
		entries = entries.filter(issue_encountered=True)
	elif issue_encountered == "no":
		entries = entries.filter(issue_encountered=False)

	# Sorting
	if sort not in ALLOWED_ENTRY_SORTS:
		sort = "-created_at"

	entries = entries.order_by(sort)

	# Pagination
	paginator = Paginator(entries, 10)
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	page_params = request.GET.copy()
	page_params.pop("page", None)

	sort_params = request.GET.copy()
	sort_params.pop("page", None)
	sort_params.pop("sort", None)

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
		"sort_params": sort_params.urlencode(),

		"q": q,
		"date_from": date_from,
		"date_to": date_to,
		"reported_sentiment": reported_sentiment,
		"issue_encountered": issue_encountered,
		"sort": sort,

		"displayed_entries_count": displayed_entries_count,
		"total_filtered_entries_count": total_filtered_entries_count,
		"total_study_entries_count": total_study_entries_count,
	}