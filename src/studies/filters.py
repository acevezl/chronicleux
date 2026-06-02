from django.db.models import Q
from django.core.paginator import Paginator


ALLOWED_ENTRY_SORTS = {
    "created_at",
    "-created_at",
    "participant_display_name",
    "-participant_display_name",
    "sentiment_self_report",
    "-sentiment_self_report",
    "machine_sentiment_label",
    "-machine_sentiment_label",
    "machine_theme_label",
    "-machine_theme_label",
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

    if q:
        entries = entries.filter(
            Q(content__icontains=q)
            | Q(participant_display_name__icontains=q)
            | Q(source__icontains=q)
            | Q(sentiment_self_report__icontains=q)
            | Q(issue_self_report__icontains=q)
        )

    if participant:
        entries = entries.filter(
            Q(participant_display_name__icontains=participant)
            | Q(participant__username__icontains=participant)
            | Q(participant__first_name__icontains=participant)
            | Q(participant__last_name__icontains=participant)
        )

    if date_from:
        entries = entries.filter(created_at__date__gte=date_from)

    if date_to:
        entries = entries.filter(created_at__date__lte=date_to)

    if reported_sentiment:
        entries = entries.filter(sentiment_self_report=reported_sentiment)

    if run:
        # This depends on your model structure.
        # If EntryAnalysisResult is linked to StudyAnalysisRun:
        entries = entries.filter(
            analysis_results__study_analysis_run=run
        )

        if detected_sentiment:
            entries = entries.filter(
                analysis_results__study_analysis_run=run,
                analysis_results__sentiment__label=detected_sentiment,
            )

        if theme:
            entries = entries.filter(
                analysis_results__study_analysis_run=run,
                analysis_results__theme__label=theme,
            )

        if issue == "true":
            entries = entries.filter(
                analysis_results__study_analysis_run=run,
                analysis_results__issues__isnull=False,
            )

        elif issue == "false":
            entries = entries.filter(
                analysis_results__study_analysis_run=run,
                analysis_results__issues__isnull=True,
            )

    else:
        if detected_sentiment:
            entries = entries.filter(machine_sentiment_label=detected_sentiment)

        if theme:
            entries = entries.filter(machine_theme_label=theme)

        if issue == "true":
            entries = entries.filter(machine_issue_count__gt=0)

        elif issue == "false":
            entries = entries.filter(
                Q(machine_issue_count=0) | Q(machine_issue_count__isnull=True)
            )

    if sort not in ALLOWED_ENTRY_SORTS:
        sort = "-created_at"

    entries = entries.order_by(sort)

    paginator = Paginator(entries, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    page_params = request.GET.copy()
    page_params.pop("page", None)

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
        "issue": issue,
        "sort": sort,
    }