from django.shortcuts import render, redirect, get_object_or_404

from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.contrib import messages

from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db import transaction
from django.http import HttpResponseForbidden
from django.utils.dateparse import parse_datetime

from .forms import StudyForm, DiaryEntryForm
from .models import  DiaryEntry, SentimentCategory, DiaryEntrySource, MembershipRole, Study, StudyMembership

# CREATE STUDY
@login_required
def create_study(request):
    if request.method == "POST":
        form = StudyForm(request.POST)
        if form.is_valid():
            study = form.save(commit=False)

            # Makes study creator the owner by default
            study.owner = request.user
            study.save()

            # The owner is an evaluator by default
            StudyMembership.objects.get_or_create(
                study = study,
                user = request.user,
                defaults={"role": MembershipRole.EVALUATOR}
            )

            messages.success(request, f"Study '{study.title}' created successfully by {study.owner}.")
            return redirect("study_detail", pk=study.pk)
    else:
        form = StudyForm()
    return render(request, "studies/create_study.html", {"form": form})

# VIEW STUDY
@login_required
def study_detail(request, pk):
    study = get_object_or_404(Study, pk=pk)
    tags_list = [tag.strip() for tag in study.tags.split(",") if tag.strip()] if study.tags else []

    is_evaluator = StudyMembership.objects.filter(
        study=study,
        user=request.user,
        role=MembershipRole.EVALUATOR
    ).exists()

    diary_entries = DiaryEntry.objects.filter(study=study).select_related("participant").order_by("-created_at")

    return render(request, "studies/study_detail.html", {
        "study": study,
        "tags_list": tags_list,
        "is_evaluator": is_evaluator,
        "diary_entries": diary_entries,
    })

# EDIT STUDY
@login_required
def edit_study(request, pk):
    study = get_object_or_404(Study, pk=pk)

    if request.method == "POST":
        form = StudyForm(request.POST, instance=study)
        if form.is_valid():
            form.save()
            messages.success(request, f"Study '{study.title}' was updated successfully.")
            return redirect("study_detail", pk=study.pk)
    else:
        form = StudyForm(instance=study)

    return render(request, "studies/edit_study.html", {
        "study": study,
        "form": form,
    })

# CREATE ENTRY
@login_required
def create_diary_entry(request, study_id):
    study = get_object_or_404(Study, pk=study_id)

    if request.method == "POST":
        form = DiaryEntryForm(request.POST)
        if form.is_valid():
            entry = form.save(commit=False)
            entry.study = study
            entry.participant = request.user
            entry.source = DiaryEntrySource.INTERNAL

            try:
                entry.full_clean()
                entry.save()
            except ValidationError as e:
                if hasattr(e, "message_dict"):
                    for field, errors in e.message_dict.items():
                        for error in errors:
                            if field in form.fields:
                                form.add_error(field, error)
                            else:
                                form.add_error(None, error)
                else:
                    form.add_error(None, e)
            else:
                messages.success(request, "Your diary entry was submitted successfully.")
                return redirect("study_detail", pk=study.pk)
    else:
        form = DiaryEntryForm()

    context = {
        "study": study,
        "form": form,
    }
    return render(request, "studies/create_diary_entry.html", context)

# VIEW ENTRIES
@login_required
def study_entries(request, pk):

    study = get_object_or_404(Study, pk=pk)

    diary_entries = (
        DiaryEntry.objects
        .filter(study=study)
        .select_related("participant")
    )

    # Filtering
    q = request.GET.get("q")
    participant = request.GET.get("participant")
    sentiment = request.GET.get("sentiment")
    issue = request.GET.get("issue")

    if q:
        diary_entries = diary_entries.filter(content__icontains=q)

    if participant:
        diary_entries = diary_entries.filter(participant_display_name__icontains=participant)

    if sentiment:
        diary_entries = diary_entries.filter(sentiment_self_report=sentiment)

    if issue in ["true", "false"]:
        diary_entries = diary_entries.filter(issue_encountered=(issue == "true"))

    # Sorting
    sort = request.GET.get("sort", "-created_at")
    
    allowed_sort_fields = {
        "created_at",
        "participant_display_name",
        "sentiment_self_report",
    }

    if sort.lstrip("-") in allowed_sort_fields:
        diary_entries = diary_entries.order_by(sort)

    # Pagination
    paginator = Paginator(diary_entries, 10)  # 10 per page
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    
    params = request.GET.copy()
    params.pop("page", None)

    context = {
        "study": study,
        "diary_entries": page_obj,
        "page_obj": page_obj,
        "sentiment": sentiment,
        "issue": issue,
        "participant": participant,
        "q": q,
        "sort": sort,
        "page_params": params,
    }

    return render(request, "studies/diary_entries.html", context)

# STUDY ANALYSIS
@login_required
def study_analysis(request, pk):
    study = get_object_or_404(Study, pk=pk)

    entries_total = study.entries.count()
    entries_analyzed = study.entries.exclude(sentiment__isnull=True).count()

    context = {
        "study": study,
        "entries_total": entries_total,
        "entries_analyzed": entries_analyzed,
    }
    return render(request, "studies/study_analysis.html", context)

# IMPORT ENTRIES
User = get_user_model()

VALID_SENTIMENTS = {choice[0] for choice in SentimentCategory.choices}

@login_required
def import_entries(request, pk):
    study = get_object_or_404(Study, pk=pk)

    # Access is restricted to evaluators only
    is_evaluator = StudyMembership.objects.filter(
        study=study,
        user=request.user,
        role=MembershipRole.EVALUATOR,
    ).exists()
    if not is_evaluator:
        return HttpResponseForbidden()

    # If form was post-submitted
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            messages.error(request, "Please choose a file to import.")
        else:
            try:
                rows = parse_uploaded_file(uploaded_file)
                result = import_rows_into_study(study, rows)

                messages.success(
                    request,
                    f"Imported {result['created']} entries. "
                    f"Ignored {result['skipped_owner']} owner rows and "
                    f"{result['skipped_evaluator']} evaluator rows."
                )

                return redirect("study_detail", pk=study.pk)
            except ValueError as e:
                messages.error(request, str(e))
            except ValidationError as e:
                messages.error(request, str(e))
            except Exception as e:
                messages.error(request, f"Import failed: {e}")

    return render(request, "studies/import_entries.html", {
        "study": study,
    })

@transaction.atomic
def import_rows_into_study (study, rows):
    created_count = 0
    skipped_owner = 0
    skipped_evaluator = 0

    for index, row in enumerate(rows, start=1):
        try:
            # Ignore the rows if they were authored by the study owner or an evaluator
            # B/c study owners and evaluators shall never write diary entries
            if is_owner_row(study, row):
                skipped_owner+=1
                continue

            if is_evaluator_row(study, row):
                skipped_evaluator+=1
                continue

            create_diary_entry_from_row(study, row)
            created_count+=1

        except Exception as e:
            raise ValueError(f"Row {index}: {e}")
    
    return {
        "created": created_count,
        "skipped_owner": skipped_owner,
        "skipped_evaluator": skipped_evaluator,
    }

def create_diary_entry_from_row(study, row):
    participant = resolve_participant_for_study(study, row)

    content = row.get("content")
    if not content:
        raise ValueError ("Field `content` is required in a diary entry")
    
    sentiment_self_report = row.get("sentiment_self_report")
    if not sentiment_self_report:
        raise ValueError ("Field `sentiment_self_report` is required in a diary entry")
    
    participant_display_name = row.get("participant_display_name")
    if participant is None and not participant_display_name:
        raise ValueError("Each entry must have either a resolvable ChronicleUX participant or a `participant_display_name`")
    
    created_at = parse_imported_datetime (row.get("created_at"))

    entry = DiaryEntry(
        study=study,
        participant=participant,
        participant_display_name=participant_display_name or "",
        participant_external_id=row.get("participant_external_id") or "",
        participant_email=row.get("participant_email") or "",
        content=content,
        sentiment_self_report=sentiment_self_report,
        issue_encountered=row.get("issue_encountered"),
        created_at=created_at,
        source=DiaryEntrySource.EXTERNAL,
    )

    entry.full_clean()
    entry.save()

    return entry

def is_owner_row(study, row):
    participant_email = row.get("participant_email")
    participant_external_id = row.get("participant_external_id")

    if participant_email and study.owner.email and participant_email.lower() == study.owner.email.lower():
        return True
    
    if participant_external_id and participant_external_id == study.owner.username:
        return True
    
    return False

def is_evaluator_row(study, row):
    participant_email = row.get("participant_email")
    participant_external_id = row.get("participant_external_id")

    evaluator_user = None

    if participant_email:
        evaluator_user = User.objects.filter(email__iexact=participant_email).first()

    if evaluator_user is None and participant_external_id:
        evaluator_user = User.objects.filter(username=participant_external_id).first()

    if evaluator_user is None:
        return False

    return StudyMembership.objects.filter(
        study=study,
        user=evaluator_user,
        role=MembershipRole.EVALUATOR,
    ).exists()


def resolve_participant_for_study(study, row):
    participant_email = row.get("participant_email")
    participant_external_id = row.get("participant_external_id")

    participant_user = None

    # If the entry has a participant e-mail or participant_external_id, see if they resolve to a user
    if participant_email:
        participant_user = User.objects.filter(email__iexact=participant_email).first()
    
    if participant_user is None and participant_external_id:
        participant_user = User.objects.filter(username=participant_external_id).first()
        
    # If a user was found, validate if the user is actually a study participant
    if participant_user:
        is_participant_in_study = StudyMembership.objects.filter(
            study=study,
            user=participant_user,
            role=MembershipRole.PARTICIPANT,
        ).exists()

        # If the user is not a participant in the study, make them a participant
        if not is_participant_in_study:
            StudyMembership.objects.create(
                study=study,
                user=participant_user,
                role=MembershipRole.PARTICIPANT
            )

        return participant_user
    
    return None

def parse_imported_datetime(value):
    value = normalize_str(value)
    if value is None:
        return None

    dt = parse_datetime(value)
    if dt is None:
        raise ValueError(
            "Invalid created_at value. Use ISO 8601 format, for example "
            "'2026-03-29T14:30:00Z'."
        )

    return dt

def parse_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return parse_csv(uploaded_file.file)

    if filename.endswith(".json"):
        return parse_json(uploaded_file.file)

    raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")

def normalize_sentiment(value):
    value = normalize_str(value)
    if value is None:
        raise ValueError("sentiment_self_report is required.")
    if value not in VALID_SENTIMENTS:
        raise ValueError(
            f"Invalid sentiment_self_report: {value}. "
            f"Allowed values: {', '.join(VALID_SENTIMENTS)}"
        )
    return value

def normalize_str(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def normalize_bool(value):
    if isinstance(value, bool):
        return value

    value = normalize_str(value)
    if value is None:
        raise ValueError("issue_encountered is required.")

    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Invalid boolean value: {value}")


def normalize_row(row):
    return {
        "participant_external_id": normalize_str(row.get("participant_external_id")),
        "participant_display_name": normalize_str(row.get("participant_display_name")),
        "participant_email": normalize_str(row.get("participant_email")),
        "sentiment_self_report": normalize_sentiment(row.get("sentiment_self_report")),
        "issue_encountered": normalize_bool(row.get("issue_encountered")),
        "content": normalize_str(row.get("content")),
        "created_at": normalize_str(row.get("created_at")),
    }

def parse_csv(file):
    import csv
    from io import TextIOWrapper

    text_file = TextIOWrapper(file, encoding="utf-8", newline="")
    reader = csv.DictReader(text_file)

    rows = []
    for row in reader:
        rows.append(normalize_row(row))

    return rows

def parse_json(file):
    import json

    data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of entries.")

    rows = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each JSON entry must be an object.")
        rows.append(normalize_row(item))

    return rows