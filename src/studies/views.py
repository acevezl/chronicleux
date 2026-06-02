from django.shortcuts import render, redirect, get_object_or_404

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required

from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Q
from django.http import HttpResponseForbidden
from django.views.decorators.http import require_POST
from django.urls import reverse
from django.utils.dateparse import parse_datetime

from studies.services.analysis_runner import run_study_analysis

from .filters import filter_diary_entries
from .forms import StudyForm, DiaryEntryForm
from .models import  DiaryEntry, SentimentCategory, DiaryEntrySource, MembershipRole, Study, StudyMembership, StudyAnalysisRun, DiaryEntryAnalysis

# ----------------------- VIEWS ----------------------- #

# ------------------ #
# CREATE DIARY ENTRY #
# ------------------ #
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


# ------------ #
# CREATE STUDY #
# ------------ #
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


# ------------------ #
# DIARY ENTRY DETAIL #
# ------------------ #
@login_required
def diary_entry_detail(request, study_pk, entry_pk):
	study = get_object_or_404(Study, pk=study_pk)

	is_owner = study.owner_id == request.user.id

	is_evaluator = StudyMembership.objects.filter(
		study=study,
		user=request.user,
		role=MembershipRole.EVALUATOR,
	).exists()

	if not is_owner and not is_evaluator:
		return HttpResponseForbidden()

	entry = get_object_or_404(
		DiaryEntry.objects.select_related("study", "participant"),
		pk=entry_pk,
		study=study,
	)

	diary_entry_analysis = (
		DiaryEntryAnalysis.objects
		.filter(entry=entry)
		.first()
	)

	print (diary_entry_analysis)
	return render(
		request,
		"studies/diary_entry_detail.html",
		{
			"study": study,
			"entry": entry,
			"selected_run": diary_entry_analysis,
		},
	)


# ---------- #
# EDIT STUDY #
# ---------- #
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

# -------------- #
# IMPORT ENTRIES #
# -------------- #
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


# ------------------------ #
# MACHINE ANALYSIS DETAILS #
# ------------------------ #
@login_required
def machine_analysis_details (request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)
	run = get_object_or_404(StudyAnalysisRun, pk=run_pk, study=study)

	diary_entries = (
		DiaryEntry.objects
		.filter(study=study)
		.select_related("participant")
	)

	is_evaluator = StudyMembership.objects.filter(
		study=study,
		user=request.user,
		role=MembershipRole.EVALUATOR
	).exists()

	context = {
		"study": study,
		"run": run,
		"diary_entries": diary_entries,
		"diary_entries_filter_url": reverse("machine_analysis_entries_partial", args=[study.pk, run.pk]),
	}

	return render (request, "studies/machine_analysis_details.html", context)


# ------------------------------- #
# MACHINE ANALYSIS ETRIES PARTIAL #
# ------------------------------- #

@login_required
def machine_analysis_entries_partial(request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)
	run = get_object_or_404(StudyAnalysisRun, pk=run_pk, study=study)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	context = filter_diary_entries(request, study, run=run)
	context["study"] = study
	context["run"] = run
	context["diary_entries_filter_url"] = reverse(
		"machine_analysis_entries_partial",
		args=[study.pk, run.pk],
	)

	return render(request, "studies/partials/_diary_entries.html", context)

# ---------------------------- #
# MANAGE EVALUATORS (OF STUDY) #
# ---------------------------- #
@login_required
def manage_evaluators(request, pk):
	study = get_object_or_404(Study, pk=pk)

	if study.owner != request.user:
		return HttpResponseForbidden()

	if request.method == "POST":
		action = request.POST.get("action")
		user_id = request.POST.get("user_id")

		user = get_object_or_404(User, pk=user_id)

		if user == study.owner:
			messages.error(request, "The study owner cannot be managed as an evaluator.")
			return redirect("manage_evaluators", pk=study.pk)

		if action == "add":
			membership, created = StudyMembership.objects.get_or_create(
				study=study,
				user=user,
				defaults={"role": MembershipRole.EVALUATOR},
			)

			if created:
				messages.success(request, "Evaluator added.")
			elif membership.role == MembershipRole.PARTICIPANT:
				messages.error(request, "This user is already a participant and cannot also be an evaluator.")
			else:
				messages.warning(request, "This user is already an evaluator in this study.")

		elif action == "remove":
			deleted_count, _ = StudyMembership.objects.filter(
				study=study,
				user=user,
				role=MembershipRole.EVALUATOR,
			).exclude(
				user=study.owner
			).delete()

			if deleted_count:
				messages.success(request, "Evaluator removed.")
			else:
				messages.warning(request, "This user is not a removable evaluator in this study.")

		return redirect("manage_evaluators", pk=study.pk)

	q = request.GET.get("q", "").strip()

	evaluator_memberships = (
		StudyMembership.objects
		.filter(study=study, role=MembershipRole.EVALUATOR)
		.select_related("user")
		.order_by("user__username")
	)

	excluded_user_ids = StudyMembership.objects.filter(
		study=study
	).values_list("user_id", flat=True)

	available_users = User.objects.exclude(
		id__in=excluded_user_ids
	).exclude(
		id=study.owner_id
	)

	if q:
		available_users = available_users.filter(
			Q(username__icontains=q)
			| Q(email__icontains=q)
			| Q(first_name__icontains=q)
			| Q(last_name__icontains=q)
		)

	available_users = available_users.order_by("username")[:25]

	context = {
		"study": study,
		"evaluator_memberships": evaluator_memberships,
		"available_users": available_users,
		"q": q,
	}

	return render(request, "studies/manage_evaluators.html", context)


# ------------------------------ #
# MANAGE PARTICIPANTS (OF STUDY) #
# ------------------------------ #
@login_required
def manage_participants(request, pk):
	study = get_object_or_404(Study, pk=pk)

	if study.owner != request.user:
		return HttpResponseForbidden()

	if request.method == "POST":
		action = request.POST.get("action")
		user_id = request.POST.get("user_id")

		user = get_object_or_404(User, pk=user_id)

		if user == study.owner:
			messages.error(request, "The study owner cannot be managed as a participant.")
			return redirect("manage_participants", pk=study.pk)

		if action == "add":
			membership, created = StudyMembership.objects.get_or_create(
				study=study,
				user=user,
				defaults={"role": MembershipRole.PARTICIPANT},
			)

			if created:
				messages.success(request, "Participant added.")
			elif membership.role == MembershipRole.EVALUATOR:
				messages.error(request, "This user is already an evaluator and cannot also be a participant.")
			else:
				messages.warning(request, "This user is already a participant in this study.")

		elif action == "remove":
			deleted_count, _ = StudyMembership.objects.filter(
				study=study,
				user=user,
				role=MembershipRole.PARTICIPANT,
			).delete()

			if deleted_count:
				messages.success(request, "Participant removed.")
			else:
				messages.warning(request, "This user is not a participant in this study.")

		return redirect("manage_participants", pk=study.pk)

	q = request.GET.get("q", "").strip()

	participant_memberships = (
		StudyMembership.objects
		.filter(study=study, role=MembershipRole.PARTICIPANT)
		.select_related("user")
		.order_by("user__username")
	)

	excluded_user_ids = StudyMembership.objects.filter(
		study=study
	).values_list("user_id", flat=True)

	available_users = User.objects.exclude(
		id__in=excluded_user_ids
	).exclude(
		id=study.owner_id
	)

	if q:
		available_users = available_users.filter(
			Q(username__icontains=q)
			| Q(email__icontains=q)
			| Q(first_name__icontains=q)
			| Q(last_name__icontains=q)
		)

	available_users = available_users.order_by("username")[:25]

	context = {
		"study": study,
		"participant_memberships": participant_memberships,
		"available_users": available_users,
		"q": q,
	}

	return render(request, "studies/manage_participants.html", context)


# -------------------- #
# RUN MACHINE ANALYSIS #
# -------------------- #
@login_required
@require_POST
def run_machine_analysis(request, pk):
	study = get_object_or_404(Study, pk=pk)

	if study.owner != request.user:
		return HttpResponseForbidden()

	if not study.entries.exists():
		messages.warning(
			request,
			"Cannot run machine analysis: No diary entries found in the study."
		)
		return redirect("study_detail", pk=study.pk)

	try:
		study_analysis_run = run_study_analysis(study_id=study.pk)

	except Exception as e:
		messages.error(request, f"Machine analysis failed: {e}")
		return redirect("study_detail", pk=study.pk)

	messages.success(
		request,
		f"Machine analysis completed successfully. Run ID: {study_analysis_run.pk}"
	)

	return redirect(
		"machine_analysis_details",
		study_pk=study.pk,
		run_pk=study_analysis_run.pk,
	)


# ----------------- #
# STUDIES (LIST OF) #
# ----------------- #
@login_required
def studies(request):

	studies = Study.objects.all()

	# Filtering
	q = request.GET.get("q")
	owner = request.GET.get("owner")
	status = request.GET.get("status")

	if q:
		studies = studies.filter(title__icontains=q)

	if owner:
		studies = studies.filter(owner__username__icontains=owner)

	if status:
		studies = studies.filter(status=status)

	# Sorting
	sort = request.GET.get("sort", "-created_at")

	allowed_sort_fields = [
		"title",
		"created_at",
		"status",
		"owner__username",
	]

	if sort.lstrip("-") not in allowed_sort_fields:
		studies = studies.order_by(sort)

	# Pagination
	paginator = Paginator(studies, 10)  # 10 per page
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	sort_params = request.GET.copy()
	sort_params.pop("sort", None)
	sort_params.pop("page", None)

	page_params = request.GET.copy()
	page_params.pop("page", None)

	context = {
		"studies": page_obj,
		"page_obj": page_obj,
		"sort": sort,
		"sort_params": sort_params,
		"page_params": page_params,
	}

	return render(request, "studies/studies.html", context)


# ----------------------------- #
# STUDY DETAIL (AKA VIEW STUDY) #
# ----------------------------- #
@login_required
def study_detail(request, pk):
	study = get_object_or_404(Study, pk=pk)
	tags_list = [tag.strip() for tag in study.tags.split(",") if tag.strip()] if study.tags else []

	is_evaluator = StudyMembership.objects.filter(
		study=study,
		user=request.user,
		role=MembershipRole.EVALUATOR
	).exists()

	diary_entries = DiaryEntry.objects.filter(
		study=study
	).select_related("participant").order_by("-created_at")

	analysis_runs = StudyAnalysisRun.objects.filter(
		study=study
	).order_by("-started_at")

	return render(request, "studies/study_detail.html", {
		"study": study,
		"tags_list": tags_list,
		"is_evaluator": is_evaluator,
		"diary_entries": diary_entries,
		"analysis_runs": analysis_runs,
	})


# ----------------------- #
# STUDY ENTRIES (LIST OF) #
# ----------------------- #
@login_required
def study_entries(request, pk):
	study = get_object_or_404(Study, pk=pk)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	context = filter_diary_entries(request, study)
	context["study"] = study
	context["diary_entries_filter_url"] = reverse(
		"diary_entries_partial",
		args=[study.pk],
	)

	return render(request, "studies/diary_entries.html", context)

# @login_required
# def study_entries(request, pk):

# 	study = get_object_or_404(Study, pk=pk)

# 	diary_entries = (
# 		DiaryEntry.objects
# 		.filter(study=study)
# 		.select_related("participant")
# 	)

# 	# # Filtering
# 	# Suppressing this while I try HTMX for a/s get requests instead of posts b/c I HATE WITH ODIO JAROCHO reloading the page every time I update the filter or sort.
# 	# q = request.GET.get("q")
# 	# participant = request.GET.get("participant")
# 	# sentiment = request.GET.get("sentiment")
# 	# issue = request.GET.get("issue")

# 	# if q:
# 	# 	diary_entries = diary_entries.filter(content__icontains=q)

# 	# if participant:
# 	# 	diary_entries = diary_entries.filter(participant_display_name__icontains=participant)

# 	# if sentiment:
# 	# 	diary_entries = diary_entries.filter(sentiment_self_report=sentiment)

# 	# if issue in ["true", "false"]:
# 	# 	diary_entries = diary_entries.filter(issue_encountered=(issue == "true"))

# 	# # Sorting
# 	# sort = request.GET.get("sort", "-created_at")
	
# 	# allowed_sort_fields = {
# 	# 	"created_at",
# 	# 	"participant_display_name",
# 	# 	"sentiment_self_report",
# 	# }

# 	# if sort.lstrip("-") in allowed_sort_fields:
# 	# 	diary_entries = diary_entries.order_by(sort)

# 	# # Pagination
# 	# paginator = Paginator(diary_entries, 10)  # 10 per page
# 	# page_number = request.GET.get("page")
# 	# page_obj = paginator.get_page(page_number)
	
# 	# params = request.GET.copy()
# 	# params.pop("page", None)

# 	is_evaluator = StudyMembership.objects.filter(
# 		study=study,
# 		user=request.user,
# 		role=MembershipRole.EVALUATOR
# 	).exists()

# 	# context = {
# 	# 	"study": study,
# 	# 	"diary_entries": page_obj,
# 	# 	"page_obj": page_obj,
# 	# 	"sentiment": sentiment,
# 	# 	"issue": issue,
# 	# 	"participant": participant,
# 	# 	"q": q,
# 	# 	"sort": sort,
# 	# 	"page_params": params,
# 	# 	"is_evaluator": is_evaluator,
# 	# }

# 	# Using this context instead of the long one with a query while I try HTMX for async querying, sorting, and pagination.
# 	context = {
# 		"study": study,
# 		"diary_entries": diary_entries,
# 		"diary_entries_filter_url": reverse("diary_entries_partial", args=[study.pk]),
# 	}

# 	return render(request, "studies/diary_entries.html", context)


# --------------------- #
# STUDY ENTRIES PARTIAL #
# --------------------- #
@login_required
def study_entries_partial(request, study_pk):
	study = get_object_or_404(Study, pk=study_pk)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()
	
	# If the user opens or refeshes the partial URL directly,
	# send them to the full page instead. No looky looky for you Mr or Mrs...
	if request.headers.get("HX-Request") != "true":
		url = reverse("study_entries", args=[study.pk])
		querystring = request.GET.urlencode()

		if querystring:
			url = f"{url}?{querystring}"

		return redirect(url)

	context = filter_diary_entries(request, study)
	context["study"] = study
	context["diary_entries_filter_url"] = reverse(
		"diary_entries_partial",
		args=[study.pk],
	)

	response = render(
		request,
		"studies/partials/_diary_entries.html",
		context,
	)

	# Push the clean full-page URL into the browser, not the partial URL.
	full_page_url = reverse("study_entries", args=[study.pk])
	querystring = request.GET.urlencode()

	if querystring:
		full_page_url = f"{full_page_url}?{querystring}"

	response["HX-Push-Url"] = full_page_url

	return response


# ----------------------- FILTERS ----------------------- #
# Filters moved to filter.py



# ----------------------- HELPERS ----------------------- #
# Thinking about moving these to helpers.py (-n-)... maybe in the future

# USER CAN EVALUATE STUDY?
def user_can_evaluate_study(user, study):
	"""
	Returns True if the user can view/evaluate study entries and analysis.

	Allowed:
	- Study owner
	- Study members with evaluator role

	Not allowed:
	- Participants
	- Non authenticated users (obvs)
	"""

	if not user or not user.is_authenticated:
		return False

	if study.owner_id == user.id:
		return True

	return StudyMembership.objects.filter(
		study=study,
		user=user,
		role=MembershipRole.EVALUATOR,
	).exists()

# IMPORT ENTRIES: IMPORT ROWS INTO STUDY
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

# IMPORT ENTRIES: CREATE DIARY ENTRY FROM DATA ROW
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

# ENTRY IMPORT HELPER: IS OWNER ROW
def is_owner_row(study, row):
	participant_email = row.get("participant_email")
	participant_external_id = row.get("participant_external_id")

	if participant_email and study.owner.email and participant_email.lower() == study.owner.email.lower():
		return True
	
	if participant_external_id and participant_external_id == study.owner.username:
		return True
	
	return False

# ENTRY IMPORT HELPER: IS EVALUATOR ROW
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


# ENTRY IMPORT HELPER: RESOLVE PARTICIPANT FOR STUDY
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

# ENTRY IMPORT HELPER: PARSE IMPORTED DATETIME
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

# ENTRY IMPORT HELPER: PARSE UPLOADED FILE
def parse_uploaded_file(uploaded_file):
	filename = uploaded_file.name.lower()

	if filename.endswith(".csv"):
		return parse_csv(uploaded_file.file)

	if filename.endswith(".json"):
		return parse_json(uploaded_file.file)

	raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")

# ENTRY IMPORT HELPER: NORMALISE SENTIMENT
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

# ENTRY IMPORT HELPER: NORMALIZE STRING
def normalize_str(value):
	if value is None:
		return None
	value = str(value).strip()
	return value if value else None

# ENTRY IMPORT HELPER: NORMALIZE BOOLEAN
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

# ENTRY IMPORT HELPER: NORMALIZE ROW
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

# ENTRY IMPORT HELPER: PARSE CSV
def parse_csv(file):
	import csv
	from io import TextIOWrapper

	text_file = TextIOWrapper(file, encoding="utf-8", newline="")
	reader = csv.DictReader(text_file)

	rows = []
	for row in reader:
		rows.append(normalize_row(row))

	return rows

# ENTRY IMPORT HELPER: PARSE JSON
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