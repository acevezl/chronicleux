from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages

from .models import Study
from .forms import StudyForm
from .models import StudyMembership, MembershipRole

# CREATE STUDY
@login_required
def create_study(request):
    if request.method == "POST":
        form = StudyForm(request.POST)
        if form.is_valid():
            study = form.save(commit=False)
            study.owner = request.user
            study.save()

            # Makes study creator the owner by default
            StudyMembership.objects.get_or_create(
                study = study,
                user = request.user,
                defaults={"role":"owner"}
            )

            messages.success(request, f"Study '{study.title}' created successfully by {study.owner}.")
            return redirect("study_detail", pk=study.pk)
    else:
        form = StudyForm()
    return render(request, "studies/create_study.html", {"form": form})

from django.shortcuts import get_object_or_404

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

    return render(request, "studies/study_detail.html", {
        "study": study,
        "tags_list": tags_list,
        "is_evaluator": is_evaluator,
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
                return redirect("study_detail", study_id=study.pk)
    else:
        form = DiaryEntryForm()

    context = {
        "study": study,
        "form": form,
    }
    return render(request, "studies/create_diary_entry.html", context)

# IMPORT ENTRIES
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

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            messages.error(request, "Please choose a file to import.")
        else:
            try:
                rows = parse_uploaded_file(uploaded_file)
                messages.success(request, f"Parsed {len(rows)} entries successfully.")
                print(rows[:2])  # temporary debug
            except ValueError as e:
                messages.error(request, str(e))

    return render(request, "studies/import_entries.html", {
        "study": study,
    })

def parse_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return parse_csv(uploaded_file.file)

    if filename.endswith(".json"):
        return parse_json(uploaded_file.file)

    raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")

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
        "sentiment_self_report": normalize_str(row.get("sentiment_self_report")),
        "issue_encountered": normalize_bool(row.get("issue_encountered")),
        "content": normalize_str(row.get("content")),
        "created_at": normalize_str(row.get("created_at")),
    }

def parse_csv(file):
    import csv
    from io import TextIOWrapper

    text_file = TextIOWrapper(file, encoding="utf-8")
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