from django.shortcuts import render, redirect, get_object_or_404

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required

from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpResponseForbidden
from django.views.decorators.http import require_POST
from django.urls import reverse
from django.utils import timezone
from django.utils.formats import date_format

from studies.services.nlp.registry import (
	get_available_sentiment_methods,
	get_available_theme_methods,
	get_available_sentiment_method_values,
	get_available_theme_method_values,
)

from studies.services.llm.client import get_available_llm_providers, get_llm_model

from studies.services.analysis_runner import create_study_analysis_run
from studies.services.analysis_tasks import queue_study_analysis_run

from .filters import (
	filter_diary_entries, 
	filter_analysis_entries, 
	filter_canonical_themes, 
	filter_canonical_issues,
	)

from .forms import (
	CanonicalIssueForm, 
	CanonicalThemeForm,
	DiaryEntryForm, 
	DiaryEntryManualEvaluationForm,
	StudyForm, 
)

from .models import (
	AnalysisRunStatus, 
	DiaryEntry, 
	DiaryEntryAnalysis, 
	DiaryEntrySource, 
	MembershipRole, 
	SentimentCategory, 
	Study, 
	StudyMembership, 
	StudyAnalysisRun, 
	StudyStatus, 
	CanonicalIssue, 
	CanonicalTheme, 
	ThemeAndIssueSource, 
	ThemeAndIssueStatus,
)

from .helpers import (
	import_rows_into_study,
	parse_uploaded_file,
	user_can_evaluate_study,
	require_catalogue_manager,
	import_canonical_themes,
	import_canonical_issues,
	parse_uploaded_canonical_theme_file,
	parse_uploaded_canonical_issue_file,
)

# ----------------------- STUDIES ----------------------- #

# ------------------ #
# CREATE DIARY STUDY #
# ------------------ #
@login_required
def create_diary_study(request):
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
			return redirect("diary_study_detail", pk=study.pk)
	else:
		form = StudyForm()

	context = {
		"page_title_heroicon":"book-open",
		"page_title":"Creating New Diary Study",
		"page_subtitle":"Use the form below to define the protocol of your diary study.",
		"form": form
	}

	return render(request, "studies/diary_study_create.html", context)

# ------------------ #
# DIARY STUDY DETAIL #
# ------------------ #
@login_required
def diary_study_detail(request, pk):
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

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context = {
		"page_title_heroicon":"book-open",
		"page_title":study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"study": study,
		"tags_list": tags_list,
		"is_evaluator": is_evaluator,
		"diary_entries": diary_entries,
		"analysis_runs": analysis_runs,
	}

	return render(request, "studies/diary_study_detail.html", context)

# ---------------- #
# EDIT DIARY STUDY #
# ---------------- #
@login_required
def edit_diary_study(request, pk):
	study = get_object_or_404(Study, pk=pk)

	if request.method == "POST":
		form = StudyForm(request.POST, instance=study)
		if form.is_valid():
			form.save()
			messages.success(request, f"Study '{study.title}' was updated successfully.")
			return redirect("diary_study_detail", pk=study.pk)
	else:
		form = StudyForm(instance=study)

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context = {
		"page_title_heroicon":"book-open",
		"page_title":study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"study": study,
		"form": form,
	}

	return render(request, "studies/diary_study_edit.html", context)

# ---------------- #
# DIARY STUDY LIST #
# ---------------- #
@login_required
def studies(request):
	studies = Study.objects.filter(
		Q(owner=request.user) |
		Q(memberships__user=request.user)
	).distinct()

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

	if sort.lstrip("-") in allowed_sort_fields:
		studies = studies.order_by(sort)
	else:
		studies = studies.order_by("-created_at")

	# Pagination
	paginator = Paginator(studies, 10)
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	sort_params = request.GET.copy()
	sort_params.pop("sort", None)
	sort_params.pop("page", None)

	page_params = request.GET.copy()
	page_params.pop("page", None)

	context = {
		"page_title_heroicon":"queue-list",
		"page_title":"My Studies",
		"page_subtitle":"Studies I own and studies I evaluate.",
		"studies": page_obj,
		"page_obj": page_obj,
		"sort": sort,
		"sort_params": sort_params,
		"page_params": page_params,
		"status_choices": StudyStatus.choices,
	}

	return render(request, "studies/diary_study_list.html", context)

# ----------------------- ENTRIES ----------------------- #

# ------------------ #
# CREATE DIARY ENTRY #
# ------------------ #
@login_required
def create_diary_entry(request, pk):
	study = get_object_or_404(Study, pk=pk)

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
				return redirect("diary_study_detail", pk=study.pk)
	else:
		form = DiaryEntryForm()

	context = {
		"page_title_heroicon":"book-open",
		"page_title": study.title,
		"page_subtitle":"Write a new diary entry.",
		"study": study,
		"form": form
	}

	return render(request, "studies/diary_entry_create.html", context)

# ---------------- #
# DIARY ENTRY LIST #
# ---------------- #
@login_required
def entries(request, pk):
	study = get_object_or_404(Study, pk=pk)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	context = filter_diary_entries(request, study)
	context["study"] = study
	context["diary_entries_filter_url"] = reverse(
		"diary_entries_partial",
		args=[study.pk],
	)

	context["is_evaluator"] = StudyMembership.objects.filter(
		study=study,
		user=request.user,
		role=MembershipRole.EVALUATOR
	).exists()

	context["page_title_heroicon"]="book-open"
	context["page_title"]= study.title

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context["page_subtitle"] = f"Owner: {owner_name}, Created on: {created_at}"

	return render(request, "studies/diary_entry_list.html", context)


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
		url = reverse("entries", args=[study.pk])
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
		"studies/partials/_entries.html",
		context,
	)

	# Push the clean full-page URL into the browser, not the partial URL.
	full_page_url = reverse("entries", args=[study.pk])
	querystring = request.GET.urlencode()

	if querystring:
		full_page_url = f"{full_page_url}?{querystring}"

	response["HX-Push-Url"] = full_page_url

	return response

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

	# Pick up all the different analyses that exist for this entry
	entry_runs = DiaryEntryAnalysis.objects.filter(entry=entry)

	# Pick up the default / selected run for this entry
	selected_run = entry.selected_entry_run

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context = {
		"page_title_heroicon":"book-open",
		"page_title":study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"study": study,
		"entry": entry,
		"selected_run": selected_run,
		"entry_runs": entry_runs,
	}

	return render(
		request,
		"studies/diary_entry_detail.html",
		context
	)

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
	
	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context = {
		"page_title_heroicon":"book-open",
		"page_title":study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"study": study,
	}

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

				return redirect("entries", pk=study.pk)
			except ValueError as e:
				messages.error(request, str(e))
			except ValidationError as e:
				messages.error(request, str(e))
			except Exception as e:
				messages.error(request, f"Import failed: {e}")

	return render(request, "studies/import_entries.html", context)


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

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context = {
		"page_title_heroicon":"book-open",
		"page_title":study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
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

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context = {
		"page_title_heroicon":"book-open",
		"page_title":study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"study": study,
		"participant_memberships": participant_memberships,
		"available_users": available_users,
		"q": q,
	}

	return render(request, "studies/manage_participants.html", context)

# ----------------------- #
# SELECT ANALYSIS METHODS #
# ----------------------- #
@login_required
def select_analysis_methods(request, pk):
	study = get_object_or_404(Study, pk=pk)

	if study.owner != request.user:
		return HttpResponseForbidden()

	context = {
		"study": study,
		"sentiment_methods": get_available_sentiment_methods(),
		"theme_methods": get_available_theme_methods(),
		"llm_providers": get_available_llm_providers(),
	}

	return render(request, "studies/select_analysis_methods.html", context)


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
		return redirect("diary_study_detail", pk=study.pk)

	analysis_mode = request.POST.get("analysis_mode")

	sentiment_method = None
	theme_method = None
	llm_provider = None
	llm_model = None

	if analysis_mode == "nlp":
		sentiment_method = request.POST.get("sentiment_method")
		theme_method = request.POST.get("theme_method")

		if not sentiment_method or not theme_method:
			messages.error(request, "Select both a sentiment method and a thematic analysis method.")
			return redirect("select_analysis_methods", pk=study.pk)

		if sentiment_method == "llm" or theme_method == "llm":
			messages.error(request, "LLM methods cannot be selected in NLP mode.")
			return redirect("select_analysis_methods", pk=study.pk)

		if sentiment_method not in get_available_sentiment_method_values():
			messages.error(request, "Invalid sentiment analysis method.")
			return redirect("select_analysis_methods", pk=study.pk)

		if theme_method not in get_available_theme_method_values():
			messages.error(request, "Invalid thematic analysis method.")
			return redirect("select_analysis_methods", pk=study.pk)

	elif analysis_mode == "llm":
		sentiment_method = "llm"
		theme_method = "llm"

		llm_provider = request.POST.get("llm_provider")
		llm_model = (request.POST.get("llm_model") or "").strip() or None

		if not llm_provider:
			messages.error(request, "Select an LLM provider.")
			return redirect("select_analysis_methods", pk=study.pk)
		
		if not llm_model:
			llm_model = get_llm_model(llm_provider)

	else:
		messages.error(request, "Select a valid analysis mode.")
		return redirect("select_analysis_methods", pk=study.pk)
	
	try:
		study_analysis_run = create_study_analysis_run(
			study_id=study.pk,
			user_id=request.user.id,
			sentiment_method=sentiment_method,
			theme_method=theme_method,
			llm_provider=llm_provider,
			llm_model=llm_model,
		)

		queue_study_analysis_run(study_analysis_run.pk)

	except Exception as e:
		messages.error(request, f"Machine analysis could not be queued: {e}")
		return redirect("diary_study_detail", pk=study.pk)

	messages.success(
		request,
		f"Machine analysis queued successfully. Run ID: {study_analysis_run.pk}"
	)

	return redirect("diary_study_detail", pk=study.pk)


# ------------------------ #
# MACHINE ANALYSIS DETAILS #
# ------------------------ #
@login_required
def machine_analysis_details(request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)
	run = get_object_or_404(StudyAnalysisRun, pk=run_pk, study=study)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	context = filter_analysis_entries(request, study, run)

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	pending_human_evaluation_count = DiaryEntryAnalysis.objects.filter(
		run=run,
	).filter(
		Q(evaluator_sentiment_label__isnull=True)
		| Q(evaluator_sentiment_label="")
		| Q(evaluator_dominant_theme__isnull=True)
	).count()

	completed_human_evaluation_count = run.total_entries - pending_human_evaluation_count

	context.update({
		"page_title_heroicon":"book-open",
		"page_title":study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"study": study,
		"run": run,
		"pending_human_evaluation_count":pending_human_evaluation_count,
		"completed_human_evaluation_count":completed_human_evaluation_count,
		"machine_analysis_entries_filter_url": reverse(
			"machine_analysis_entries_partial",
			args=[study.pk, run.pk],
		),
		
	})

	return render(
		request,
		"studies/analysis_details.html",
		context
	)

# -------------------------------- #
# MACHINE ANALYSIS ENTRIES PARTIAL #
# -------------------------------- #
@login_required
def machine_analysis_entries_partial(request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)
	run = get_object_or_404(StudyAnalysisRun, pk=run_pk, study=study)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	context = filter_analysis_entries(request, study, run)

	context.update({
		"study": study,
		"run": run,
		"machine_analysis_entries_filter_url": reverse(
			"machine_analysis_entries_partial",
			args=[study.pk, run.pk],
		),
	})

	return render(
		request,
		"studies/partials/_entries_with_analysis.html",
		context
	)

# ----------------------------- #
# ANALYSIS RUNS (LIST OF)       #
# ----------------------------- #
@login_required
def analysis_runs(request):

	runs = StudyAnalysisRun.objects.select_related(
		"study",
		"created_by",
	).filter(
		created_by=request.user
	)

	# Filtering
	q = request.GET.get("q")
	status = request.GET.get("status")
	study = request.GET.get("study")
	model = request.GET.get("model")
	sentiment_method = request.GET.get("sentiment_method")
	theme_method = request.GET.get("theme_method")

	if q:
		runs = runs.filter(
			Q(study__title__icontains=q) |
			Q(analysis_model__icontains=q) |
			Q(analysis_version__icontains=q) |
			Q(error_message__icontains=q)
		)

	if status:
		runs = runs.filter(status=status)

	if study:
		runs = runs.filter(study__title__icontains=study)

	if model:
		runs = runs.filter(analysis_model__icontains=model)

	if sentiment_method:
		runs = runs.filter(methods__sentiment=sentiment_method)

	if theme_method:
		runs = runs.filter(methods__theme=theme_method)

	# Sorting
	sort = request.GET.get("sort", "-started_at")

	allowed_sort_fields = [
		"started_at",
		"completed_at",
		"status",
		"study__title",
		"analysis_model",
		"analysis_version",
		"total_entries",
		"average_sentiment_score",
	]

	if sort.lstrip("-") in allowed_sort_fields:
		runs = runs.order_by(sort)
	else:
		sort = "-started_at"
		runs = runs.order_by(sort)

	# Pagination
	paginator = Paginator(runs, 10)
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	sort_params = request.GET.copy()
	sort_params.pop("sort", None)
	sort_params.pop("page", None)

	page_params = request.GET.copy()
	page_params.pop("page", None)

	context = {
		"analysis_runs": page_obj,
		"page_obj": page_obj,
		"sort": sort,
		"sort_params": sort_params,
		"page_params": page_params,
		"status_choices": AnalysisRunStatus.choices,
		"analysis_runs_filter_url": reverse("analysis_runs_partial"),
	}

	return render(request, "studies/analysis_run_list.html", context)


# ----------------------------- #
# ANALYSIS RUNS PARTIAL         #
# ----------------------------- #
@login_required
def analysis_runs_partial(request):

	if request.headers.get("HX-Request") != "true":
		url = reverse("analysis_runs")
		querystring = request.GET.urlencode()

		if querystring:
			url = f"{url}?{querystring}"

		return redirect(url)

	runs = StudyAnalysisRun.objects.select_related(
		"study",
		"created_by",
	).filter(
		created_by=request.user
	)

	# Filtering
	q = request.GET.get("q")
	status = request.GET.get("status")
	study = request.GET.get("study")
	model = request.GET.get("model")
	sentiment_method = request.GET.get("sentiment_method")
	theme_method = request.GET.get("theme_method")

	if q:
		runs = runs.filter(
			Q(study__title__icontains=q) |
			Q(analysis_model__icontains=q) |
			Q(analysis_version__icontains=q) |
			Q(error_message__icontains=q)
		)

	if status:
		runs = runs.filter(status=status)

	if study:
		runs = runs.filter(study__title__icontains=study)

	if model:
		runs = runs.filter(analysis_model__icontains=model)

	if sentiment_method:
		runs = runs.filter(methods__sentiment=sentiment_method)

	if theme_method:
		runs = runs.filter(methods__theme=theme_method)

	# Sorting
	sort = request.GET.get("sort", "-started_at")

	allowed_sort_fields = [
		"started_at",
		"completed_at",
		"status",
		"study__title",
		"analysis_model",
		"analysis_version",
		"total_entries",
		"average_sentiment_score",
	]

	if sort.lstrip("-") in allowed_sort_fields:
		runs = runs.order_by(sort)
	else:
		sort = "-started_at"
		runs = runs.order_by(sort)

	# Pagination
	paginator = Paginator(runs, 10)
	page_number = request.GET.get("page")
	page_obj = paginator.get_page(page_number)

	sort_params = request.GET.copy()
	sort_params.pop("sort", None)
	sort_params.pop("page", None)

	page_params = request.GET.copy()
	page_params.pop("page", None)

	context = {
		"analysis_runs": page_obj,
		"page_obj": page_obj,
		"sort": sort,
		"sort_params": sort_params,
		"page_params": page_params,
		"status_choices": AnalysisRunStatus.choices,
		"analysis_runs_filter_url": reverse("analysis_runs_partial"),
	}

	response = render(
		request,
		"studies/partials/_analysis_runs.html",
		context,
	)

	full_page_url = reverse("analysis_runs")
	querystring = request.GET.urlencode()

	if querystring:
		full_page_url = f"{full_page_url}?{querystring}"

	response["HX-Push-Url"] = full_page_url

	return response


# ------------------------------ #	
# CANONICAL THEME CATALOGUE LIST #
# ------------------------------ #
@login_required
def canonical_theme_catalogue_list(request):
	require_catalogue_manager(request.user)

	context = filter_canonical_themes(request)

	context.update({
		"page_title_heroicon":"tag",
		"page_title": "Canonical Themes Catalogue",
		"page_subtitle": "Manage the global catalogue of canonical themes used by analysis methods.",
		"canonical_themes_filter_url": reverse("canonical_theme_catalogue_partial"),
		"diary_entries_filter_url": reverse("canonical_theme_catalogue_partial"),
	})

	return render(
		request,
		"studies/catalogues/canonical_theme_list.html",
		context,
	)


# -------------------------------- #
# CANONICAL THEME CATALOGUE PARTIAL #
# -------------------------------- #
@login_required
def canonical_theme_catalogue_partial(request):
	require_catalogue_manager(request.user)

	if request.headers.get("HX-Request") != "true":
		url = reverse("canonical_theme_catalogue_list")
		querystring = request.GET.urlencode()

		if querystring:
			url = f"{url}?{querystring}"

		return redirect(url)

	context = filter_canonical_themes(request)

	context.update({
		"canonical_themes_filter_url": reverse("canonical_theme_catalogue_partial"),
		"diary_entries_filter_url": reverse("canonical_theme_catalogue_partial"),
	})

	response = render(
		request,
		"studies/catalogues/partials/_canonical_themes.html",
		context,
	)

	full_page_url = reverse("canonical_theme_catalogue_list")
	querystring = request.GET.urlencode()

	if querystring:
		full_page_url = f"{full_page_url}?{querystring}"

	response["HX-Push-Url"] = full_page_url

	return response


# -----------------------#
# CANONICAL THEME CREATE #
# -----------------------#
@login_required
def canonical_theme_create(request):
	require_catalogue_manager(request.user)

	if request.method == "POST":
		form = CanonicalThemeForm(request.POST)

		if form.is_valid():
			theme = form.save(commit=False)
			theme.created_by = request.user

			if not theme.source:
				theme.source = ThemeAndIssueSource.EVALUATOR

			if not theme.status:
				theme.status = ThemeAndIssueStatus.APPROVED

			theme.save()

			messages.success(request, "Canonical theme created.")
			return redirect("canonical_theme_catalogue_list")
	else:
		form = CanonicalThemeForm()

	return render(
		request,
		"studies/catalogues/canonical_theme_form.html",
		{
			"form": form,
			"page_title": "Create Canonical Theme",
			"page_subtitle": "Create and maintain reusable canonical themes for machine and evaluator analysis.",
			"submit_label": "Create theme",
		},
	)

# -----------------------#
# CANONICAL THEME UPDATE #
# -----------------------#
@login_required
def canonical_theme_update(request, theme_pk):
	require_catalogue_manager(request.user)

	theme = get_object_or_404(CanonicalTheme, pk=theme_pk)

	if request.method == "POST":
		form = CanonicalThemeForm(request.POST, instance=theme)

		if form.is_valid():
			form.save()

			messages.success(request, "Canonical theme updated.")
			return redirect("canonical_theme_catalogue_list")
	else:
		form = CanonicalThemeForm(instance=theme)

	return render(
		request,
		"studies/catalogues/canonical_theme_form.html",
		{
			"theme": theme,
			"form": form,
			"page_title": "Edit Canonical Theme",
			"page_subtitle": "Update and maintain reusable canonical themes for machine and evaluator analysis.",
			"submit_label": "Save theme",
		},
	)

# -----------------------#
# CANONICAL THEME DELETE #
# -----------------------#
@login_required
def canonical_theme_delete(request, theme_pk):
	require_catalogue_manager(request.user)

	theme = get_object_or_404(CanonicalTheme, pk=theme_pk)

	if request.method == "POST":
		theme.delete()

		messages.success(request, "Canonical theme deleted.")
		return redirect("canonical_theme_catalogue_list")

	return render(
		request,
		"studies/catalogues/canonical_theme_confirm_delete.html",
		{
			"theme": theme,
			"page_title": "Delete Canonical Theme",
			"page_subtitle": "Confirm whether this theme should be removed from the global catalogue.",
		},
	)


# -----------------------#
# CANONICAL THEME IMPORT #
# -----------------------#
@login_required
def canonical_theme_import(request):
	require_catalogue_manager(request.user)

	if request.method == "POST":
		uploaded_file = request.FILES.get("file")

		try:
			rows = parse_uploaded_canonical_theme_file(uploaded_file)
			result = import_canonical_themes(rows, request.user)

			messages.success(
				request,
				f"Imported canonical themes. "
				f"Created: {result['created']}. "
				f"Updated: {result['updated']}. "
				f"Skipped: {result['skipped']}."
			)

			return redirect("canonical_theme_catalogue_list")

		except ValueError as e:
			messages.error(request, str(e))

		except Exception as e:
			messages.error(request, f"Theme import failed: {e}")

	return render(
		request,
		"studies/catalogues/canonical_theme_import.html",
		{
			"page_title": "Import Canonical Themes",
			"page_subtitle": "Import canonical themes into the global catalogue from a CSV file.",
		},
	)


# ------------------------------ #
# CANONICAL ISSUE CATALOGUE LIST #
# ------------------------------ #
@login_required
def canonical_issue_catalogue_list(request):
	require_catalogue_manager(request.user)

	context = filter_canonical_issues(request)

	context.update({
		"page_title": "Canonical Issues Catalogue",
		"page_subtitle": "Manage the global catalogue of canonical issues used by analysis methods.",
		"canonical_issues_filter_url": reverse("canonical_issue_catalogue_partial"),
		"diary_entries_filter_url": reverse("canonical_issue_catalogue_partial"),
	})

	return render(
		request,
		"studies/catalogues/canonical_issue_list.html",
		context,
	)


# -------------------------------- #
# CANONICAL ISSUE CATALOGUE PARTIAL #
# -------------------------------- #
@login_required
def canonical_issue_catalogue_partial(request):
	require_catalogue_manager(request.user)

	if request.headers.get("HX-Request") != "true":
		url = reverse("canonical_issue_catalogue_list")
		querystring = request.GET.urlencode()

		if querystring:
			url = f"{url}?{querystring}"

		return redirect(url)

	context = filter_canonical_issues(request)

	context.update({
		"canonical_issues_filter_url": reverse("canonical_issue_catalogue_partial"),
		"diary_entries_filter_url": reverse("canonical_issue_catalogue_partial"),
	})

	response = render(
		request,
		"studies/catalogues/partials/_canonical_issues.html",
		context,
	)

	full_page_url = reverse("canonical_issue_catalogue_list")
	querystring = request.GET.urlencode()

	if querystring:
		full_page_url = f"{full_page_url}?{querystring}"

	response["HX-Push-Url"] = full_page_url

	return response


# -----------------------#
# CANONICAL ISSUE CREATE #
# -----------------------#
@login_required
def canonical_issue_create(request):
	require_catalogue_manager(request.user)

	if request.method == "POST":
		form = CanonicalIssueForm(request.POST)

		if form.is_valid():
			issue = form.save(commit=False)
			issue.created_by = request.user

			if not issue.source:
				issue.source = ThemeAndIssueSource.EVALUATOR

			if not issue.status:
				issue.status = ThemeAndIssueStatus.APPROVED

			issue.save()

			messages.success(request, "Canonical issue created.")
			return redirect("canonical_issue_catalogue_list")
		
	else:
		form = CanonicalIssueForm()

	return render(
		request,
		"studies/catalogues/canonical_issue_form.html",
		{
			"form": form,
			"page_title": "Create Canonical Issue",
			"page_subtitle": "Create and maintain reusable canonical issues for machine and evaluator analysis.",
			"submit_label": "Create issue",
		},
	)

# -----------------------#
# CANONICAL ISSUE UPDATE #
# -----------------------#
@login_required
def canonical_issue_update(request, issue_pk):
	require_catalogue_manager(request.user)

	issue = get_object_or_404(CanonicalIssue, pk=issue_pk)

	if request.method == "POST":
		form = CanonicalIssueForm(request.POST, instance=issue)

		if form.is_valid():
			form.save()

			messages.success(request, "Canonical issue updated.")
			return redirect("canonical_issue_catalogue_list")
	else:
		form = CanonicalIssueForm(instance=issue)

	return render(
		request,
		"studies/catalogues/canonical_issue_form.html",
		{
			"issue": issue,
			"form": form,
			"page_title": "Edit Canonical Issue",
			"page_subtitle": "Update and maintain reusable canonical issues for machine and evaluator analysis.",
			"submit_label": "Save issue",
		},
	)

# -----------------------#
# CANONICAL ISSUE DELETE #
# -----------------------#
@login_required
def canonical_issue_delete(request, issue_pk):
	require_catalogue_manager(request.user)

	issue = get_object_or_404(CanonicalIssue, pk=issue_pk)

	if request.method == "POST":
		issue.delete()

		messages.success(request, "Canonical issue deleted.")
		return redirect("canonical_issue_catalogue_list")

	return render(
		request,
		"studies/catalogues/canonical_issue_confirm_delete.html",
		{
			"issue": issue,
			"page_title": "Delete Canonical Issue",
			"page_subtitle": "Confirm whether this issue should be removed from the global catalogue.",
		},
	)

# -----------------------#
# CANONICAL ISSUE IMPORT #
# -----------------------#
@login_required
def canonical_issue_import(request):
	require_catalogue_manager(request.user)

	if request.method == "POST":
		uploaded_file = request.FILES.get("file")

		try:
			rows = parse_uploaded_canonical_issue_file(uploaded_file)
			result = import_canonical_issues(rows, request.user)

			messages.success(
				request,
				f"Imported canonical issues. "
				f"Created: {result['created']}. "
				f"Updated: {result['updated']}. "
				f"Skipped: {result['skipped']}."
			)

			return redirect("canonical_issue_catalogue_list")

		except ValueError as e:
			messages.error(request, str(e))

		except Exception as e:
			messages.error(request, f"Issue import failed: {e}")

	return render(
		request,
		"studies/catalogues/canonical_issue_import.html",
		{
			"page_title": "Import Canonical Issues",
			"page_subtitle": "Import canonical issues into the global catalogue from a CSV file.",
		},
	)


# -------------------------- #
# HUMAN EVALUATION QUEUE    #
# -------------------------- #
@login_required
def human_evaluation_queue(request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)
	run = get_object_or_404(StudyAnalysisRun, pk=run_pk, study=study)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	context = filter_analysis_entries(request, study, run)

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context.update({
		"page_title_heroicon": "book-open",
		"page_title": study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"study": study,
		"run": run,
		"is_human_evaluation_queue": True,
		"machine_analysis_entries_filter_url": reverse(
			"human_evaluation_queue_partial",
			args=[study.pk, run.pk],
		),
	})

	return render(
		request,
		"studies/human_evaluation_queue.html",
		context
	)


# ------------------------------- #
# HUMAN EVALUATION QUEUE PARTIAL #
# ------------------------------- #
@login_required
def human_evaluation_queue_partial(request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)
	run = get_object_or_404(StudyAnalysisRun, pk=run_pk, study=study)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	if request.headers.get("HX-Request") != "true":
		url = reverse("human_evaluation_queue", args=[study.pk, run.pk])
		querystring = request.GET.urlencode()

		if querystring:
			url = f"{url}?{querystring}"

		return redirect(url)

	context = filter_analysis_entries(request, study, run)

	context.update({
		"study": study,
		"run": run,
		"is_human_evaluation_queue": True,
		"machine_analysis_entries_filter_url": reverse(
			"human_evaluation_queue_partial",
			args=[study.pk, run.pk],
		),
	})

	response = render(
		request,
		"studies/partials/_entries_with_analysis.html",
		context,
	)

	full_page_url = reverse("human_evaluation_queue", args=[study.pk, run.pk])
	querystring = request.GET.urlencode()

	if querystring:
		full_page_url = f"{full_page_url}?{querystring}"

	response["HX-Push-Url"] = full_page_url

	return response


# ------------------------- #
# EVALUATE ENTRY ANALYSIS   #
# ------------------------- #
@login_required
def evaluate_entry_analysis(request, study_pk, run_pk, analysis_pk):
	study = get_object_or_404(Study, pk=study_pk)
	run = get_object_or_404(StudyAnalysisRun, pk=run_pk, study=study)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	entry_analysis = get_object_or_404(
		DiaryEntryAnalysis.objects.select_related(
			"entry",
			"entry__study",
			"entry__participant",
			"run",
			"run__study",
		).prefetch_related(
			"evaluator_issues",
		),
		pk=analysis_pk,
		run=run,
	)

	if request.method == "POST":
		form = DiaryEntryManualEvaluationForm(
			request.POST,
			instance=entry_analysis,
		)

		if form.is_valid():
			evaluated_analysis = form.save(commit=False)
			evaluated_analysis.evaluated_by = request.user
			evaluated_analysis.evaluated_at = timezone.now()
			evaluated_analysis.save()
			form.save_m2m()

			messages.success(request, "Human evaluation saved.")

			return redirect(
				"human_evaluation_queue",
				study_pk=study.pk,
				run_pk=run.pk,
			)

	else:
		form = DiaryEntryManualEvaluationForm(instance=entry_analysis)

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context = {
		"page_title_heroicon": "book-open",
		"page_title": study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"study": study,
		"run": run,
		"entry": entry_analysis.entry,
		"entry_analysis": entry_analysis,
		"form": form,
	}

	return render(
		request,
		"studies/diary_entry_evaluation.html",
		context,
	)