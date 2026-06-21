from django.core.paginator import Paginator
from django.db.models import Q, Count

from .models import DiaryEntry, DiaryEntryAnalysis, CanonicalTheme, CanonicalIssue, ThemeAndIssueSource, ThemeAndIssueStatus


# -------- FILTER FOR ENTRY LIST PAGE -------- #

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


# -------- FILTER FOR STUDY ANALYSIS WITHIN MACHINE ANALYSIS DEETS PAGE -------- #

ALLOWED_ANALYSIS_ENTRY_SORTS = {
	"entry__created_at",
	"-entry__created_at",
	"entry__participant_display_name",
	"-entry__participant_display_name",
	"sentiment_label",
	"-sentiment_label",
	"themes__name",
	"-themes__name",
	"issues__name",
	"-issues__name",
}

def filter_analysis_entries(request, study, run):
	q = request.GET.get("q", "").strip()
	sentiment = request.GET.get("sentiment", "").strip()
	theme = request.GET.get("theme", "").strip()
	issue_detected = request.GET.get("issue_detected", "").strip()
	issue_tag = request.GET.get("issue_tag", "").strip()
	human_evaluation = request.GET.get("human_evaluation", "").strip()
	sort = request.GET.get("sort", "-entry__created_at")

	entry_analyses = (
		DiaryEntryAnalysis.objects
		.filter(run=run, entry__study=study)
		.select_related("entry", "entry__participant")
		.prefetch_related(
			"themes",
			"issues",
			"evaluator_themes",
			"evaluator_issues",
		)
	)

	if q:
		entry_analyses = entry_analyses.filter(
			Q(entry__content__icontains=q)
			| Q(entry__participant_display_name__icontains=q)
			| Q(entry__participant_external_id__icontains=q)
			| Q(entry__participant_email__icontains=q)
			| Q(entry_summary__icontains=q)
			| Q(sentiment_label__icontains=q)
			| Q(themes__name__icontains=q)
			| Q(themes__description__icontains=q)
			| Q(themes__aliases__icontains=q)
			| Q(issues__name__icontains=q)
			| Q(issues__description__icontains=q)
			| Q(issues__aliases__icontains=q)
		)

	if sentiment:
		entry_analyses = entry_analyses.filter(sentiment_label=sentiment)

	if theme:
		entry_analyses = entry_analyses.filter(
			Q(themes__name__iexact=theme)
			| Q(themes__aliases__icontains=theme)
		)

	if issue_detected == "yes":
		entry_analyses = entry_analyses.filter(issues__isnull=False)
	elif issue_detected == "no":
		entry_analyses = entry_analyses.filter(issues__isnull=True)

	if issue_tag:
		entry_analyses = entry_analyses.filter(
			Q(issues__name__iexact=issue_tag)
			| Q(issues__aliases__icontains=issue_tag)
		)

	entry_analyses = entry_analyses.annotate(
		evaluator_theme_count=Count("evaluator_themes", distinct=True)
	)

	if human_evaluation == "pending":
		entry_analyses = entry_analyses.filter(
			Q(evaluator_sentiment_label__isnull=True)
			| Q(evaluator_sentiment_label="")
			| Q(evaluator_theme_count=0)
		)

	elif human_evaluation == "completed":
		entry_analyses = entry_analyses.exclude(
			Q(evaluator_sentiment_label__isnull=True)
			| Q(evaluator_sentiment_label="")
			| Q(evaluator_theme_count=0)
		)

	if sort not in ALLOWED_ANALYSIS_ENTRY_SORTS:
		sort = "-entry__created_at"

	entry_analyses = entry_analyses.distinct().order_by(sort)

	sort_params = request.GET.copy()
	sort_params.pop("sort", None)

	total_filtered_entry_analyses_count = entry_analyses.count()
	total_run_entry_analyses_count = (
		DiaryEntryAnalysis.objects
		.filter(run=run, entry__study=study)
		.count()
	)

	return {
		"entry_analyses": entry_analyses,

		"sort_params": sort_params.urlencode(),

		"q": q,
		"sentiment": sentiment,
		"theme": theme,
		"issue_detected": issue_detected,
		"issue_tag": issue_tag,
		"human_evaluation": human_evaluation,
		"sort": sort,

		"displayed_entry_analyses_count": total_filtered_entry_analyses_count,
		"total_filtered_entry_analyses_count": total_filtered_entry_analyses_count,
		"total_run_entry_analyses_count": total_run_entry_analyses_count,
	}



# -------- FILTER FOR CANONICAL THEME CATALOGUE PAGE -------- #

ALLOWED_CANONICAL_THEME_SORTS = {
	"name",
	"-name",
	"source",
	"-source",
	"status",
	"-status",
	"is_active",
	"-is_active",
	"created_at",
	"-created_at",
	"updated_at",
	"-updated_at",
}

def filter_canonical_themes(request):
	q = request.GET.get("q", "").strip()
	source = request.GET.get("source", "").strip()
	status = request.GET.get("status", "").strip()
	is_active = request.GET.get("is_active", "").strip()
	sort = request.GET.get("sort", "name")

	themes = CanonicalTheme.objects.all()

	if q:
		themes = themes.filter(
			Q(name__icontains=q)
			| Q(description__icontains=q)
			| Q(examples__icontains=q)
			| Q(aliases__icontains=q)
		)

	if source:
		themes = themes.filter(source=source)

	if status:
		themes = themes.filter(status=status)

	if is_active == "yes":
		themes = themes.filter(is_active=True)
	elif is_active == "no":
		themes = themes.filter(is_active=False)

	if sort not in ALLOWED_CANONICAL_THEME_SORTS:
		sort = "name"

	themes = themes.order_by(sort)

	paginator = Paginator(themes, 10)
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	page_params = request.GET.copy()
	page_params.pop("page", None)

	sort_params = request.GET.copy()
	sort_params.pop("page", None)
	sort_params.pop("sort", None)

	displayed_entries_count = len(page_obj.object_list)
	total_filtered_entries_count = paginator.count
	total_study_entries_count = CanonicalTheme.objects.count()

	return {
		"themes": page_obj.object_list,
		"page_obj": page_obj,

		"page_params": page_params.urlencode(),
		"sort_params": sort_params.urlencode(),

		"q": q,
		"source": source,
		"status": status,
		"is_active": is_active,
		"sort": sort,

		"source_choices": ThemeAndIssueSource.choices,
		"status_choices": ThemeAndIssueStatus.choices,

		"displayed_entries_count": displayed_entries_count,
		"total_filtered_entries_count": total_filtered_entries_count,
		"total_study_entries_count": total_study_entries_count,
	}


# -------- FILTER FOR CANONICAL ISSUE CATALOGUE PAGE -------- #

ALLOWED_CANONICAL_ISSUE_SORTS = {
	"name",
	"-name",
	"source",
	"-source",
	"status",
	"-status",
	"is_active",
	"-is_active",
	"created_at",
	"-created_at",
	"updated_at",
	"-updated_at",
}

def filter_canonical_issues(request):
	q = request.GET.get("q", "").strip()
	source = request.GET.get("source", "").strip()
	status = request.GET.get("status", "").strip()
	is_active = request.GET.get("is_active", "").strip()
	sort = request.GET.get("sort", "name")

	issues = CanonicalIssue.objects.all()

	if q:
		issues = issues.filter(
			Q(name__icontains=q)
			| Q(description__icontains=q)
			| Q(examples__icontains=q)
			| Q(aliases__icontains=q)
		)

	if source:
		issues = issues.filter(source=source)

	if status:
		issues = issues.filter(status=status)

	if is_active == "yes":
		issues = issues.filter(is_active=True)
	elif is_active == "no":
		issues = issues.filter(is_active=False)

	if sort not in ALLOWED_CANONICAL_ISSUE_SORTS:
		sort = "name"

	issues = issues.order_by(sort)

	paginator = Paginator(issues, 10)
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	page_params = request.GET.copy()
	page_params.pop("page", None)

	sort_params = request.GET.copy()
	sort_params.pop("page", None)
	sort_params.pop("sort", None)

	displayed_entries_count = len(page_obj.object_list)
	total_filtered_entries_count = paginator.count
	total_study_entries_count = CanonicalIssue.objects.count()

	return {
		"issues": page_obj.object_list,
		"page_obj": page_obj,

		"page_params": page_params.urlencode(),
		"sort_params": sort_params.urlencode(),

		"q": q,
		"source": source,
		"status": status,
		"is_active": is_active,
		"sort": sort,

		"source_choices": ThemeAndIssueSource.choices,
		"status_choices": ThemeAndIssueStatus.choices,

		"displayed_entries_count": displayed_entries_count,
		"total_filtered_entries_count": total_filtered_entries_count,
		"total_study_entries_count": total_study_entries_count,
	}