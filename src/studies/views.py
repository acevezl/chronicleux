from django.shortcuts import render, redirect, get_object_or_404

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required

from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.http import HttpResponseForbidden, HttpResponseNotAllowed
from django.views.decorators.http import require_POST
from django.urls import reverse
from django.utils import timezone
from django.utils.formats import date_format

from studies.services.nlp.registry import (
	get_available_issue_methods,
	get_available_issue_method_values,
	get_available_sentiment_methods,
	get_available_sentiment_method_values,
	get_available_theme_methods,
	get_available_theme_method_values,
)

from studies.services.llm.client import get_available_llm_providers, get_llm_model

from studies.services.analysis_runner import create_study_analysis_run
from studies.services.analysis_tasks import queue_study_analysis_run
from studies.services._confusion_matrix_calculator import refresh_sentiment_confusion_matrix_for_run
from studies.services._ordinal_distance_calculator import refresh_sentiment_ordinal_distance_for_run
from studies.services._sentiment_summarizer import summarize_study_sentiment
from studies.services._theme_summarizer import consolidate_themes_on_study
from studies.services._issue_summarizer import consolidate_issues_on_study
from studies.services.ux_recommendation_builder import build_ux_recommendation_report
from studies.services.issue_framework_mapper import map_all_canonical_issues_to_framework_criteria

from .filters import (
	filter_diary_entries, 
	filter_analysis_entries, 
	filter_canonical_themes, 
	filter_canonical_issues,
	filter_ux_frameworks,
	filter_ux_framework_criteria,
	)

from .forms import (
	CanonicalIssueForm, 
	CanonicalThemeForm,
	EntryForm, 
	EntryManualEvaluationForm,
	StudyForm, 
	UXFrameworkForm,
	UXFrameworkCriterionForm,
)

from .models import (
	AnalysisStatus, 
	CanonicalIssue, 
	CanonicalTheme, 
	CanonicalIssueToFrameworkMapping,
	CanonicalThemeToFrameworkMapping,
	Entry, 
	EntryAnalysis, 
	EntryEvaluation,
	EntrySource, 
	UXFrameworkMappingStatus,
	MembershipRole, 
	SentimentCategory, 
	Study, 
	StudyMembership, 
	StudyAnalysis,
	StudyEvaluationTheme,
	StudyEvaluationIssue, 
	StudyStatus, 
	ThemeAndIssueSource, 
	ThemeAndIssueStatus,
	UXFramework,
	UXFrameworkCriterion,
	UXFrameworkType,
	UXRecommendationReport,
)

from .helpers import (
	build_participant_analysis_data,
	build_study_dominant_sentiment_evolution,
	build_study_average_sentiment_evolution,
	import_canonical_themes,
	import_canonical_issues,
	import_rows_into_study,
	import_ux_frameworks,
	parse_csv_int_ids,
	parse_uploaded_canonical_theme_file,
	parse_uploaded_canonical_issue_file,
	parse_uploaded_file,
	parse_uploaded_ux_framework_file,
	require_catalogue_manager,
	sync_run_evaluator_sentiment_from_study,
	sync_canonical_issue_framework_mappings,
	sync_canonical_theme_framework_mappings,
	user_can_evaluate_study,
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

	diary_entries = Entry.objects.filter(
		study=study
	).select_related("participant").order_by("-created_at")

	analysis_runs = (
		StudyAnalysis.objects
		.filter(study=study)
		.select_related("dominant_theme")
		.prefetch_related("themes", "issues")
		.order_by("-started_at")
	)

	evaluator_themes = (
		StudyEvaluationTheme.objects
		.filter(study=study)
		.order_by("-entry_count", "theme")
	)

	evaluator_issues = (
		StudyEvaluationIssue.objects
		.filter(study=study)
		.order_by("-entry_count", "issue")
	)

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
		"evaluator_themes": evaluator_themes,
		"evaluator_issues": evaluator_issues,
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
		form = EntryForm(request.POST)
		if form.is_valid():
			entry = form.save(commit=False)
			entry.study = study
			entry.participant = request.user
			entry.source = EntrySource.INTERNAL

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
		form = EntryForm()

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

	context = filter_diary_entries(request, study, evaluator=request.user)
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

	context = filter_diary_entries(request, study, evaluator=request.user)
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
		Entry.objects.select_related("study", "participant"),
		pk=entry_pk,
		study=study,
	)

	entry_evaluation = (
		EntryEvaluation.objects
		.filter(
			entry=entry,
			evaluated_by=request.user,
		)
		.prefetch_related(
			"evaluator_themes",
			"evaluator_issues",
		)
		.first()
	)

	# Pick up all the different analyses that exist for this entry
	entry_runs = EntryAnalysis.objects.filter(entry=entry)

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
		"entry_evaluation": entry_evaluation,
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

	return render(request, "studies/diary_entry_import.html", context)


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
		"sentiment_methods": get_available_sentiment_methods(),
		"theme_methods": get_available_theme_methods(),
		"issue_methods": get_available_issue_methods(),
		"llm_providers": get_available_llm_providers(),
	}

	return render(request, "studies/diary_analysis_select_methods.html", context)


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
	issue_method = None
	llm_provider = None
	llm_model = None

	if analysis_mode == "nlp":
		sentiment_method = request.POST.get("sentiment_method")
		theme_method = request.POST.get("theme_method")
		issue_method = request.POST.get("issue_method")

		if not sentiment_method or not theme_method or not issue_method:
			messages.error(request, "You must select one sentiment-analysis method, one thematic-extraction method, and one issue-detection method.")
			return redirect("select_analysis_methods", pk=study.pk)

		if sentiment_method == "llm" or theme_method == "llm" or issue_method == "llm":
			messages.error(request, "LLM methods cannot be selected in NLP mode.")
			return redirect("select_analysis_methods", pk=study.pk)

		if sentiment_method not in get_available_sentiment_method_values():
			messages.error(request, "Invalid sentiment-analysis method.")
			return redirect("select_analysis_methods", pk=study.pk)

		if theme_method not in get_available_theme_method_values():
			messages.error(request, "Invalid thematic-extraction method.")
			return redirect("select_analysis_methods", pk=study.pk)
		
		if issue_method not in get_available_issue_method_values():
			messages.error(request, "Invalid issue-detection method.")
			return redirect("select_analysis_methods", pk=study.pk)

	elif analysis_mode == "llm":
		sentiment_method = theme_method = issue_method = "llm"

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
			issue_method=issue_method,
			llm_provider=llm_provider,
			llm_model=llm_model,
		)

		queue_study_analysis_run(study_analysis_run.pk)

	except Exception as e:
		messages.error(request, f"Machine analysis could not be queued: {e}")
		return redirect("analysis_runs", study_pk=study.pk)

	messages.success(
		request,
		f"Machine analysis queued successfully. Run ID: {study_analysis_run.pk}"
	)

	return redirect("analysis_runs", study_pk=study.pk)


# ------------------------ #
# MACHINE ANALYSIS DETAILS #
# ------------------------ #
@login_required
def machine_analysis_details(request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)
	
	run = get_object_or_404(
		StudyAnalysis.objects
		.filter(study=study)
		.select_related("dominant_theme")
		.prefetch_related("themes", "issues"),
		pk=run_pk,
	)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()
	
	context = filter_analysis_entries(request, study, run)

	participant_analysis_data = build_participant_analysis_data(
		study,
		run,
	)

	# builds the average sentiment evolution for both the self-reported by participant, and the assessed by evaluator.
	study_average_sentiment_evolution = build_study_average_sentiment_evolution(
		study
	)

	# builds the sentiment evolution for both the self-reported by participant, and the assessed by evaluator.
	study_dominant_sentiment_evolution = build_study_dominant_sentiment_evolution (
		study, run
	)

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	completed_human_evaluation_count = (
		EntryEvaluation.objects
		.filter(
			entry__study=study,
			evaluator_sentiment_label__isnull=False,
			evaluator_themes__isnull=False,
		)
		.exclude(evaluator_sentiment_label="")
		.distinct()
		.count()
	)

	pending_human_evaluation_count = run.total_entries - completed_human_evaluation_count

	context.update({
		"page_title_heroicon":"book-open",
		"page_title":study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"study": study,
		"run": run,
		"participant_analysis_data": participant_analysis_data,
		"participant_average_sentiment_evolution":study_average_sentiment_evolution["participant"],
		"evaluator_average_sentiment_evolution":study_average_sentiment_evolution["evaluator"],
		"participant_dominant_sentiment_evolution":study_dominant_sentiment_evolution["participant"],
		"evaluator_dominant_sentiment_evolution":study_dominant_sentiment_evolution["evaluator"],
		"machine_dominant_sentiment_evolution":study_dominant_sentiment_evolution["machine"],
		"pending_human_evaluation_count":pending_human_evaluation_count,
		"completed_human_evaluation_count":completed_human_evaluation_count,
		"machine_analysis_entries_filter_url": reverse(
			"machine_analysis_entries_partial",
			args=[study.pk, run.pk],
		),
		
	})

	return render(
		request,
		"studies/diary_analysis_details.html",
		context
	)

# -------------------------------- #
# MACHINE ANALYSIS ENTRIES PARTIAL #
# -------------------------------- #
@login_required
def machine_analysis_entries_partial(request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)
	
	run = get_object_or_404(
		StudyAnalysis.objects
		.select_related("dominant_theme")
		.prefetch_related("themes", "issues"),
		pk=run_pk,
		study=study,
	)

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
def analysis_runs(request, study_pk):

	study = get_object_or_404(Study, pk=study_pk)

	runs = StudyAnalysis.objects.select_related(
		"study",
		"created_by",
		"dominant_theme",
	).prefetch_related(
		"themes",
		"issues",
	).filter(
		study=study,
		created_by=request.user,
	)

	# Filtering
	q = request.GET.get("q")
	status = request.GET.get("status")
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

	if model:
		runs = runs.filter(analysis_model__icontains=model)

	if sentiment_method:
		runs = runs.filter(methods__sentiment=sentiment_method)

	if theme_method:
		runs = runs.filter(methods__theme=theme_method)

	evaluator_themes = (
		StudyEvaluationTheme.objects
		.filter(study=study)
		.order_by("-entry_count", "theme")
	)

	evaluator_issues = (
		StudyEvaluationIssue.objects
		.filter(study=study)
		.order_by("-entry_count", "issue")
	)

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

	owner_name = (study.owner.get_full_name() or study.owner.get_username()).title()

	created_at = date_format(
		timezone.localtime(study.created_at),
		"j M Y, H:i"
	)

	context = {
		"study": study,
		"page_title_heroicon":"book-open",
		"page_title":study.title,
		"page_subtitle": f"Owner: {owner_name}, Created on: {created_at}",
		"analysis_runs": page_obj,
		"page_obj": page_obj,
		"sort": sort,
		"sort_params": sort_params,
		"page_params": page_params,
		"status_choices": AnalysisStatus.choices,
		"analysis_runs_filter_url": reverse(
			"analysis_runs_partial",
			kwargs={"study_pk": study.pk},
		),
		"evaluator_themes": evaluator_themes,
		"evaluator_issues": evaluator_issues,
	}

	return render(request, "studies/diary_analysis_list.html", context)


# ----------------------------- #
# ANALYSIS RUNS PARTIAL         #
# ----------------------------- #
@login_required
def analysis_runs_partial(request, study_pk):

	study = get_object_or_404(Study, pk=study_pk)

	if request.headers.get("HX-Request") != "true":
		url = reverse(
			"analysis_runs",
			kwargs={"study_pk": study.pk},
		)
		querystring = request.GET.urlencode()

		if querystring:
			url = f"{url}?{querystring}"

		return redirect(url)

	runs = StudyAnalysis.objects.select_related(
		"study",
		"created_by",
		"dominant_theme",
	).prefetch_related(
		"themes",
		"issues",
	).filter(
		study=study,
		created_by=request.user,
	)

	# Filtering
	q = request.GET.get("q")
	status = request.GET.get("status")
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
		"study": study,
		"analysis_runs": page_obj,
		"page_obj": page_obj,
		"sort": sort,
		"sort_params": sort_params,
		"page_params": page_params,
		"status_choices": AnalysisStatus.choices,
		"analysis_runs_filter_url": reverse(
			"analysis_runs_partial",
			kwargs={"study_pk": study.pk},
		),
	}

	response = render(
		request,
		"studies/partials/_analysis_runs.html",
		context,
	)

	full_page_url = reverse(
		"analysis_runs",
		kwargs={"study_pk": study.pk},
	)
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

			sync_canonical_theme_framework_mappings(
				theme=theme,
				framework_criteria=form.cleaned_data["framework_criteria"],
				user=request.user,
			)

			messages.success(request, "Canonical theme created.")
			return redirect("canonical_theme_catalogue_list")
	else:
		form = CanonicalThemeForm()

	return render(
		request,
		"studies/catalogues/canonical_theme_form.html",
		{
			"form": form,
			"page_title_heroicon":"tag",
			"page_title": "Create Canonical Theme",
			"page_subtitle": "Create and maintain reusable canonical themes for machine and evaluator analysis.",
			"submit_label": "Create theme",
		},
	)


# -----------------------#
# CANONICAL THEME UPDATE #
# -----------------------#
@login_required
@login_required
def canonical_theme_update(request, theme_pk):
	require_catalogue_manager(request.user)

	theme = get_object_or_404(CanonicalTheme, pk=theme_pk)

	if request.method == "POST":
		form = CanonicalThemeForm(request.POST, instance=theme)

		if form.is_valid():
			theme = form.save()

			sync_canonical_theme_framework_mappings(
				theme=theme,
				framework_criteria=form.cleaned_data["framework_criteria"],
				user=request.user,
			)

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
			"page_title_heroicon":"tag",
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
			"page_title_heroicon":"tag",
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

	context = {
		"page_title_heroicon":"tag",
		"page_title": "Import Canonical Themes",
		"page_subtitle": "Import canonical themes into the global catalogue from a CSV file.",
	}
	
	return render(
		request,
		"studies/catalogues/canonical_theme_import.html",
		context,
	)


# ------------------------------ #
# CANONICAL ISSUE CATALOGUE LIST #
# ------------------------------ #
@login_required
def canonical_issue_catalogue_list(request):
	require_catalogue_manager(request.user)

	context = filter_canonical_issues(request)

	context.update({
		"page_title_heroicon":"exclamation-triangle",
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

            sync_canonical_issue_framework_mappings(
                issue=issue,
                framework_criteria=form.cleaned_data["framework_criteria"],
                user=request.user,
            )

            messages.success(request, "Canonical issue created.")
            return redirect("canonical_issue_catalogue_list")
        
    else:
        form = CanonicalIssueForm()

    return render(
        request,
        "studies/catalogues/canonical_issue_form.html",
        {
            "form": form,
            "page_title_heroicon":"exclamation-triangle",
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
            issue = form.save()

            sync_canonical_issue_framework_mappings(
                issue=issue,
                framework_criteria=form.cleaned_data["framework_criteria"],
                user=request.user,
            )

            approved_pending_mapping_ids = parse_csv_int_ids(
                request.POST.get("approved_pending_mapping_ids")
            )
            rejected_pending_mapping_ids = parse_csv_int_ids(
                request.POST.get("rejected_pending_mapping_ids")
            )

            if approved_pending_mapping_ids:
                CanonicalIssueToFrameworkMapping.objects.filter(
                    issue=issue,
                    pk__in=approved_pending_mapping_ids,
                    status=UXFrameworkMappingStatus.SUGGESTED,
                ).update(
                    status=UXFrameworkMappingStatus.APPROVED,
                )

            if rejected_pending_mapping_ids:
                CanonicalIssueToFrameworkMapping.objects.filter(
                    issue=issue,
                    pk__in=rejected_pending_mapping_ids,
                    status=UXFrameworkMappingStatus.SUGGESTED,
                ).update(
                    status=UXFrameworkMappingStatus.REJECTED,
                )

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
            "page_title_heroicon": "exclamation-triangle",
            "page_title": "Edit Canonical Issue",
            "page_subtitle": "Update and maintain reusable canonical issues for machine and evaluator analysis.",
            "submit_label": "Save issue",
            "pending_framework_mappings": form.pending_framework_mappings,
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
			"page_title_heroicon":"exclamation-triangle",
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
			"page_title_heroicon":"exclamation-triangle",
			"page_title": "Import Canonical Issues",
			"page_subtitle": "Import canonical issues into the global catalogue from a CSV file.",
		},
	)

# -------------- #
# EVALUATE ENTRY #
# -------------- #

@login_required
def evaluate_entry(request, study_pk, entry_pk):
	study = get_object_or_404(Study, pk=study_pk)

	entry = get_object_or_404(
		Entry.objects.select_related(
			"study",
			"participant",
		),
		pk=entry_pk,
		study=study,
	)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	entry_evaluation, _created = EntryEvaluation.objects.get_or_create(
		entry=entry,
		evaluated_by=request.user,
	)

	entry_analyses = (
		EntryAnalysis.objects
		.filter(entry=entry)
		.select_related(
			"run",
			"run__study",
		)
		.prefetch_related(
			"themes",
			"issues",
		)
		.order_by("-analyzed_at")
	)

	latest_entry_analysis = entry_analyses.first()

	if request.method == "POST":
		form = EntryManualEvaluationForm(
			request.POST,
			instance=entry_evaluation,
			evaluator=request.user,
		)

		if form.is_valid():
			entry_evaluation = form.save(commit=False)
			entry_evaluation.entry = entry
			entry_evaluation.evaluated_by = request.user
			entry_evaluation.save()
			form.save_m2m()

			summarize_study_sentiment(study)
			consolidate_themes_on_study(study)
			consolidate_issues_on_study(study)

			affected_runs = StudyAnalysis.objects.filter(
				entry_analyses__entry=entry,
			).distinct()

			for run in affected_runs:
				sync_run_evaluator_sentiment_from_study(run=run, study=study)
				refresh_sentiment_ordinal_distance_for_run(study_analysis=run, study=study)
				refresh_sentiment_confusion_matrix_for_run(study_analysis=run)

			messages.success(request, "Human evaluation saved.")

			return redirect(
				"evaluate_entry",
				study_pk=study.pk,
				entry_pk=entry.pk,
			)

	else:
		form = EntryManualEvaluationForm(
			instance=entry_evaluation,
			evaluator=request.user,
		)

	previous_entry = (
		Entry.objects
		.filter(study=study, pk__lt=entry.pk)
		.order_by("-pk")
		.first()
	)

	next_entry = (
		Entry.objects
		.filter(study=study, pk__gt=entry.pk)
		.order_by("pk")
		.first()
	)

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
		"entry": entry,
		"entry_evaluation": entry_evaluation,
		"entry_analyses": entry_analyses,
		"latest_entry_analysis": latest_entry_analysis,
		"form": form,
		"previous_entry": previous_entry,
		"next_entry": next_entry,
	}

	return render(
		request,
		"studies/diary_entry_evaluate.html",
		context,
	)

# ------------------------- #
# EVALUATE ENTRY ANALYSIS   #
# ------------------------- #
@login_required
def evaluate_entry_analysis(request, study_pk, run_pk, analysis_pk):
	study = get_object_or_404(Study, pk=study_pk)
	
	run = get_object_or_404(
		StudyAnalysis.objects
		.select_related("dominant_theme")
		.prefetch_related("themes", "issues"),
		pk=run_pk,
		study=study,
	)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	entry_analysis = get_object_or_404(
		EntryAnalysis.objects.select_related(
			"entry",
			"entry__study",
			"entry__participant",
			"run",
			"run__study",
		).prefetch_related(
			"themes",
			"issues",
			"entry__entry_evaluations",
			"entry__entry_evaluations__evaluator_themes",
			"entry__entry_evaluations__evaluator_issues",
		),
		pk=analysis_pk,
		run=run,
	)

	entry = entry_analysis.entry

	entry_evaluation, _ = EntryEvaluation.objects.get_or_create(
		entry=entry,
		evaluated_by=request.user,
	)

	entry_analyses_qs = (
		EntryAnalysis.objects
		.filter(run=run)
		.select_related("entry")
		.order_by("entry__created_at", "entry__pk", "pk")
	)

	previous_entry_analysis = (
		entry_analyses_qs
		.filter(
			Q(entry__created_at__lt=entry.created_at)
			| Q(
				entry__created_at=entry.created_at,
				entry__pk__lt=entry.pk,
			)
		)
		.order_by("-entry__created_at", "-entry__pk", "-pk")
		.first()
	)

	next_entry_analysis = (
		entry_analyses_qs
		.filter(
			Q(entry__created_at__gt=entry.created_at)
			| Q(
				entry__created_at=entry.created_at,
				entry__pk__gt=entry.pk,
			)
		)
		.first()
	)

	if request.method == "POST":
		form = EntryManualEvaluationForm(
			request.POST or None,
			instance=entry_evaluation,
			evaluator=request.user,
		)

		if form.is_valid():
			form.save()

			summarize_study_sentiment(study)
			consolidate_themes_on_study(study)
			consolidate_issues_on_study(study)
			sync_run_evaluator_sentiment_from_study(run=run, study=study)
			refresh_sentiment_ordinal_distance_for_run(study_analysis=run, study=study)
			refresh_sentiment_confusion_matrix_for_run(study_analysis=run)

			messages.success(request, "Human evaluation saved.")

			return redirect(
				"evaluate_entry_analysis",
				study_pk=study.pk,
				run_pk=run.pk,
				analysis_pk=entry_analysis.pk,
			)

	else:
		form = EntryManualEvaluationForm(
			instance=entry_evaluation,
			evaluator=request.user,
		)

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
		"entry_evaluation": entry_evaluation,
		"previous_entry_analysis": previous_entry_analysis,
		"next_entry_analysis": next_entry_analysis,
		"form": form,
	}

	return render(
		request,
		"studies/diary_analysis_evaluation.html",
		context,
	)


# --------------- #
# REFRESH METRICS #
# --------------- #
@login_required
@require_POST
def refresh_analysis_run_metrics(request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)

	run = get_object_or_404(
		StudyAnalysis,
		pk=run_pk,
		study=study,
	)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	summarize_study_sentiment(study)
	consolidate_themes_on_study(study)
	consolidate_issues_on_study(study)
	refresh_sentiment_ordinal_distance_for_run(run, study)
	refresh_sentiment_confusion_matrix_for_run(run)
	
	messages.success(request, "Metrics refreshed successfully.")

	return redirect(
		"machine_analysis_details",
		study_pk=study.pk,
		run_pk=run.pk,
	)


# ---------------------------- #
# UX FRAMEWORK CATALOGUE LIST  #
# ---------------------------- #
@login_required
def ux_framework_catalogue_list(request):
	require_catalogue_manager(request.user)

	context = filter_ux_frameworks(request)
	context.update({
		"page_title_heroicon": "bookmark-square",
		"page_title": "UX Frameworks Catalogue",
		"page_subtitle": "Manage usability frameworks used to ground UX issue and theme recommendations.",
		"ux_frameworks_filter_url": reverse("ux_framework_catalogue_partial"),
	})

	return render(
		request,
		"studies/catalogues/ux_framework_list.html",
		context,
	)


# ------------------------------ #
# UX FRAMEWORK CATALOGUE PARTIAL #
# ------------------------------ #
@login_required
def ux_framework_catalogue_partial(request):
	require_catalogue_manager(request.user)

	if request.headers.get("HX-Request") != "true":
		url = reverse("ux_framework_catalogue_list")
		querystring = request.GET.urlencode()

		if querystring:
			url = f"{url}?{querystring}"

		return redirect(url)

	context = filter_ux_frameworks(request)
	context.update({
		"ux_frameworks_filter_url": reverse("ux_framework_catalogue_partial"),
	})

	response = render(
		request,
		"studies/catalogues/partials/_ux_frameworks.html",
		context,
	)

	full_page_url = reverse("ux_framework_catalogue_list")
	querystring = request.GET.urlencode()

	if querystring:
		full_page_url = f"{full_page_url}?{querystring}"

	response["HX-Push-Url"] = full_page_url

	return response


# --------------------- #
# UX FRAMEWORK DETAIL   #
# --------------------- #
@login_required
def ux_framework_detail(request, framework_pk):
	require_catalogue_manager(request.user)

	framework = get_object_or_404(
		UXFramework.objects.prefetch_related("criteria"),
		pk=framework_pk,
	)

	criteria = framework.criteria.all().order_by("code", "name")

	return render(
		request,
		"studies/catalogues/ux_framework_detail.html",
		{
			"framework": framework,
			"criteria": criteria,
			"page_title_heroicon": "bookmark-square",
			"page_title": framework.name,
			"page_subtitle": "Review this framework and its criteria.",
		},
	)


# ------------------- #
# UX FRAMEWORK CREATE #
# ------------------- #
@login_required
def ux_framework_create(request):
	require_catalogue_manager(request.user)

	if request.method == "POST":
		form = UXFrameworkForm(request.POST)

		if form.is_valid():
			framework = form.save(commit=False)
			framework.created_by = request.user
			framework.save()

			messages.success(request, "UX framework created.")
			return redirect("ux_framework_catalogue_list")
	else:
		form = UXFrameworkForm()

	return render(
		request,
		"studies/catalogues/ux_framework_form.html",
		{
			"form": form,
			"page_title_heroicon": "bookmark-square",
			"page_title": "Create UX Framework",
			"page_subtitle": "Create a reusable usability framework such as Nielsen, ISO, WCAG, or a custom taxonomy.",
			"submit_label": "Create",
		},
	)


# ------------------- #
# UX FRAMEWORK UPDATE #
# ------------------- #
@login_required
def ux_framework_update(request, framework_pk):
	require_catalogue_manager(request.user)

	framework = get_object_or_404(UXFramework, pk=framework_pk)

	if request.method == "POST":
		form = UXFrameworkForm(request.POST, instance=framework)

		if form.is_valid():
			form.save()

			messages.success(request, "UX framework updated.")
			return redirect("ux_framework_detail", framework_pk=framework.pk)
	else:
		form = UXFrameworkForm(instance=framework)

	return render(
		request,
		"studies/catalogues/ux_framework_form.html",
		{
			"framework": framework,
			"form": form,
			"page_title_heroicon": "bookmark-square",
			"page_title": "Edit UX Framework",
			"page_subtitle": "Update this reusable usability framework.",
			"submit_label": "Save",
		},
	)


# ------------------- #
# UX FRAMEWORK DELETE #
# ------------------- #
@login_required
def ux_framework_delete(request, framework_pk):
	require_catalogue_manager(request.user)

	framework = get_object_or_404(
		UXFramework.objects.annotate(criteria_count=Count("criteria", distinct=True)),
		pk=framework_pk,
	)

	if request.method == "POST":
		framework.delete()

		messages.success(request, "UX framework deleted.")
		return redirect("ux_framework_catalogue_list")

	return render(
		request,
		"studies/catalogues/ux_framework_confirm_delete.html",
		{
			"framework": framework,
			"page_title_heroicon": "bookmark-square",
			"page_title": "Delete UX Framework",
			"page_subtitle": "Confirm whether this framework should be removed from the global catalogue.",
		},
	)


# ------------------------------------- #
# UX FRAMEWORK CRITERION CATALOGUE LIST #
# ------------------------------------- #
@login_required
def ux_framework_criterion_catalogue_list(request, framework_pk):
	require_catalogue_manager(request.user)

	framework = get_object_or_404(UXFramework, pk=framework_pk)

	context = filter_ux_framework_criteria(request, framework=framework)
	context.update({
		"framework": framework,
		"page_title_heroicon": "rectangle-stack",
		"page_title": f"{framework.name} Criteria",
		"page_subtitle": "Manage framework criteria, heuristics, dimensions, principles, or guideline chunks used for recommendations.",
		"ux_framework_criteria_filter_url": reverse(
			"ux_framework_criterion_catalogue_partial",
			kwargs={"framework_pk": framework.pk},
		),
	})

	return render(
		request,
		"studies/catalogues/ux_framework_criterion_list.html",
		context,
	)

# --------------------------------------- #
# UX FRAMEWORK CRITERION CATALOGUE PARTIAL #
# --------------------------------------- #
@login_required
def ux_framework_criterion_catalogue_partial(request):
	require_catalogue_manager(request.user)

	if request.headers.get("HX-Request") != "true":
		url = reverse("ux_framework_criterion_catalogue_list")
		querystring = request.GET.urlencode()

		if querystring:
			url = f"{url}?{querystring}"

		return redirect(url)

	context = filter_ux_framework_criteria(request)
	context.update({
		"ux_framework_criteria_filter_url": reverse("ux_framework_criterion_catalogue_partial"),
	})

	response = render(
		request,
		"studies/catalogues/partials/_ux_framework_criteria.html",
		context,
	)

	full_page_url = reverse("ux_framework_criterion_catalogue_list")
	querystring = request.GET.urlencode()

	if querystring:
		full_page_url = f"{full_page_url}?{querystring}"

	response["HX-Push-Url"] = full_page_url

	return response


# ----------------------------- #
# UX FRAMEWORK CRITERION DETAIL #
# ----------------------------- #
@login_required
def ux_framework_criterion_detail(request, framework_pk, criterion_pk):
	require_catalogue_manager(request.user)

	framework = get_object_or_404(UXFramework, pk=framework_pk)

	criterion = get_object_or_404(
		UXFrameworkCriterion.objects.select_related("framework"),
		pk=criterion_pk,
		framework=framework,
	)

	context = {
		"framework": framework,
		"criterion": criterion,
		"page_title_heroicon": "clipboard-document-list",
		"page_title": criterion.name,
		"page_subtitle": f"{framework.name} criterion details.",
	}

	return render(
		request,
		"studies/catalogues/ux_framework_criterion_detail.html",
		context,
	)


# ----------------------------- #
# UX FRAMEWORK CRITERION CREATE #
# ----------------------------- #
@login_required
def ux_framework_criterion_create(request, framework_pk=None):
	require_catalogue_manager(request.user)

	framework = None
	if framework_pk is not None:
		framework = get_object_or_404(UXFramework, pk=framework_pk)

	if request.method == "POST":
		form = UXFrameworkCriterionForm(request.POST)

		if form.is_valid():
			criterion = form.save(commit=False)

			if framework is not None:
				criterion.framework = framework

			criterion.save()

			messages.success(request, "UX framework criterion created.")
			return redirect("ux_framework_detail", framework_pk=criterion.framework.pk)
	else:
		initial = {}

		if framework is not None:
			initial["framework"] = framework

		form = UXFrameworkCriterionForm(initial=initial)

	return render(
		request,
		"studies/catalogues/ux_framework_criterion_form.html",
		{
			"framework": framework,
			"form": form,
			"page_title_heroicon": "rectangle-stack",
			"page_title": "Create UX Framework Criterion",
			"page_subtitle": "Create a reusable framework criterion with recommendation guidance.",
			"submit_label": "Save",
		},
	)


# ----------------------------- #
# UX FRAMEWORK CRITERION UPDATE #
# ----------------------------- #
@login_required
def ux_framework_criterion_update(request, framework_pk, criterion_pk):
	require_catalogue_manager(request.user)

	framework = get_object_or_404(UXFramework, pk=framework_pk)

	criterion = get_object_or_404(
		UXFrameworkCriterion.objects.select_related("framework"),
		pk=criterion_pk,
		framework=framework,
	)

	if request.method == "POST":
		form = UXFrameworkCriterionForm(request.POST, instance=criterion)

		if form.is_valid():
			criterion = form.save()

			messages.success(request, "UX framework criterion updated.")
			return redirect("ux_framework_detail", framework_pk=framework.pk)
	else:
		form = UXFrameworkCriterionForm(instance=criterion)

	return render(
		request,
		"studies/catalogues/ux_framework_criterion_form.html",
		{
			"criterion": criterion,
			"framework": criterion.framework,
			"form": form,
			"page_title_heroicon": "rectangle-stack",
			"page_title": "Edit UX Framework Criterion",
			"page_subtitle": "Update this reusable framework criterion and its recommendation guidance.",
			"submit_label": "Save",
		},
	)


# ----------------------------- #
# UX FRAMEWORK CRITERION DELETE #
# ----------------------------- #
@login_required
def ux_framework_criterion_delete(request, framework_pk, criterion_pk):
	require_catalogue_manager(request.user)

	framework = get_object_or_404(UXFramework, pk=framework_pk)

	criterion = get_object_or_404(
		UXFrameworkCriterion.objects.select_related("framework"),
		pk=criterion_pk,
		framework=framework,
	)

	if request.method == "POST":
		criterion.delete()

		messages.success(request, "UX framework criterion deleted.")
		return redirect("ux_framework_detail", framework_pk=framework.pk)

	return render(
		request,
		"studies/catalogues/ux_framework_criterion_confirm_delete.html",
		{
			"criterion": criterion,
			"framework": criterion.framework,
			"page_title_heroicon": "rectangle-stack",
			"page_title": "Delete UX Framework Criterion",
			"page_subtitle": "Confirm whether this criterion should be removed from the framework catalogue.",
		},
	)


# -------------------- #
# UX IMPORT FRAMEWORKS #
# -------------------- #
@login_required
def ux_framework_import(request):
	require_catalogue_manager(request.user)

	if request.method == "POST":
		try:
			rows = parse_uploaded_ux_framework_file(request.FILES.get("file"))
			result = import_ux_frameworks(rows, request.user)

			messages.success(
				request,
				(
					"UX frameworks imported successfully. "
					f"Frameworks created: {result['frameworks_created']}. "
					f"Frameworks updated: {result['frameworks_updated']}. "
					f"Criteria created: {result['criteria_created']}. "
					f"Criteria updated: {result['criteria_updated']}. "
					f"Rows skipped: {result['skipped']}."
				),
			)

			return redirect("ux_framework_catalogue_list")

		except ValueError as error:
			messages.error(request, str(error))

	context = {
		"page_title_heroicon": "arrow-up-tray",
		"page_title": "Import UX Frameworks",
		"page_subtitle": "Upload a CSV file containing UX frameworks and their criteria.",
	}

	return render(
		request,
		"studies/catalogues/ux_framework_import.html",
		context,
	)



# ------------------------ #
# UX RECOMMENDATION REPORT #
# ------------------------ #
@login_required
def ux_recommendation_report(request, study_pk, run_pk):
	study = get_object_or_404(Study, pk=study_pk)

	run = get_object_or_404(
		StudyAnalysis.objects
		.select_related("study", "created_by")
		.prefetch_related(
			"theme_summaries__theme",
			"issue_summaries__issue",
		),
		pk=run_pk,
		study=study,
	)

	if not user_can_evaluate_study(request.user, study):
		return HttpResponseForbidden()

	report_data = build_ux_recommendation_report(run)

	report, created = UXRecommendationReport.objects.update_or_create(
		run=run,
		defaults={
			"generated_by": request.user,
			"report_data": report_data,
		},
	)

	return render(
		request,
		"studies/ux_recommendation_report.html",
		{
			"study": study,
			"run": run,
			"report": report,
			"report_data": report_data,
			"created": created,
			"page_title_heroicon": "light-bulb",
			"page_title": "UX Recommendations Report",
			"page_subtitle": "Framework-grounded recommendations generated from detected themes and issues.",
		},
	)

# --------------------------- #
# ISSUE TO FRAMEWORK AUTO-MAP #
# --------------------------- #
@login_required
def canonical_issue_framework_auto_map(request):

	if request.method != "POST":
		return HttpResponseNotAllowed(["POST"])

	if not request.user.is_staff:
		return HttpResponseForbidden()

	result = map_all_canonical_issues_to_framework_criteria(
		min_score=0.25,
		max_mappings_per_issue=8,
		created_by=request.user,
		update_existing_system_suggestions=True,
	)

	if result.errors:
		messages.warning(
			request,
			(
				"Issue-to-framework mapping completed with "
				f"{len(result.errors)} error(s). "
				"Some mappings may not have been generated."
			),
		)
	else:
		messages.success(
			request,
			"Issue-to-framework mapping completed successfully.",
		)

	messages.info(
		request,
		(
			f"Issues scanned: {result.issues_scanned}. "
			f"Criteria scanned: {result.criteria_scanned}. "
			f"Candidates scored: {result.candidates_scored}. "
			f"Mappings created: {result.mappings_created}. "
			f"Mappings updated: {result.mappings_updated}. "
			f"Existing mappings skipped: {result.mappings_skipped_existing}. "
			f"Below threshold: {result.mappings_below_threshold}."
		),
	)

	return redirect("canonical_issue_catalogue_list")