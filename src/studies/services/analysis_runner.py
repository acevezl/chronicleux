from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from django.db import transaction
from django.utils import timezone

from studies.models import (
	Study,
	StudyStatus,
	StudyAnalysisRun,
	DiaryEntry,
	DiaryEntryAnalysis,
	AnalysisRunStatus,

	CanonicalTheme,
	CanonicalIssue,
	ThemeAndIssueSource,
	ThemeAndIssueStatus,

	DiaryEntryAnalysisCanonicalTheme,
	DiaryEntryAnalysisCanonicalIssue,
	StudyAnalysisRunCanonicalTheme,
	StudyAnalysisRunCanonicalIssue,
)

from studies.services.nlp.pipeline import analyze_study_entries
from studies.services.llm.pipeline import analyze_study_entries_with_llm

from studies.services.nlp.sentiment._confusion_matrix import refresh_sentiment_confusion_matrix_for_run


MAX_SUMMARY_LENGTH = 180

# ---------------------
# Helper functions
# ---------------------

def normalize_entry_text(text: str) -> str:
	"""
	Light normalization of entries for better analysis results.
	I'm preserving as much of the original text as possible, since VADER relies on punctuation and casing cues for sentiment analysis.
	This function primarily collapses excessive whitespace and ensures we have a string to work with.
	"""
	return " ".join((text or "").split())


def make_entry_summary(text: str, max_length: int = MAX_SUMMARY_LENGTH) -> str:
	"""
	Create a summary of the entry text, truncated to a maximum length.
	"""
	normalized = normalize_entry_text(text)

	if len(normalized) <= max_length:
		return normalized

	return normalized[:max_length].rstrip() + "..."


def build_sentiment_distribution(entry_results: list[dict]) -> list[dict]:
	"""
	Build a distribution of sentiment categories from the entry results, including both counts and percentages.
	"""
	counts = Counter(
		result["sentiment_label"]
		for result in entry_results
		if result.get("sentiment_label")
	)

	total = sum(counts.values())

	return [
		{
			"label": label,
			"count": count,
			"percentage": round(count / total, 4) if total else 0,
		}
		for label, count in counts.most_common()
	]


def build_recurring_issues(entry_results: list[dict]) -> list[dict]:
	"""
	Build a list of recurring issues based only on issue labels returned by the analyzer.

	No keyword detection.
	No catalogue matching.
	No similarity matching.
	"""
	counter = Counter()
	sentiment_totals = defaultdict(float)
	negative_counts = Counter()
	total_issue_entries = 0

	negative_labels = {"VERY_NEGATIVE", "NEGATIVE"}

	for result in entry_results:
		issue_labels = result.get("analysis_issue_tags") or []

		if not issue_labels:
			continue

		total_issue_entries += 1

		sentiment_score = result.get("sentiment_score") or 0
		sentiment_label = result.get("sentiment_label")

		for issue_label in issue_labels:
			issue_label = normalize_catalog_label(issue_label)

			if not issue_label:
				continue

			counter[issue_label] += 1
			sentiment_totals[issue_label] += sentiment_score

			if sentiment_label in negative_labels:
				negative_counts[issue_label] += 1

	return [
		{
			"label": issue_label,
			"count": count,
			"percentage": round(count / total_issue_entries, 4) if total_issue_entries else 0,
			"negative_ratio": round(negative_counts[issue_label] / count, 4) if count else 0,
			"avg_sentiment": round(sentiment_totals[issue_label] / count, 4) if count else 0,
		}
		for issue_label, count in counter.most_common()
	]


def build_recurring_themes(entry_results: list[dict]) -> list[dict]:
	counts = Counter()
	theme_metadata_by_label = {}

	for result in entry_results:
		theme_label = result.get("theme_label")

		if not theme_label:
			continue

		counts[theme_label] += 1

		if theme_label not in theme_metadata_by_label:
			theme_metadata_by_label[theme_label] = {
				"id": result.get("canonical_theme_id"),
				"label": theme_label,
				"status": result.get("canonical_theme_status"),
			}

	total = sum(counts.values())

	return [
		{
			**theme_metadata_by_label.get(label, {}),
			"label": label,
			"count": count,
			"percentage": round(count / total, 4) if total else 0,
		}
		for label, count in counts.most_common()
	]


def build_evolution_over_time(entries_with_results: list[tuple]) -> list[dict]:
	"""
	Build a time series of average sentiment and sentiment distribution over time by day.
	"""
	grouped = {}

	for entry, result in entries_with_results:
		day = entry.created_at.date().isoformat()
		grouped.setdefault(day, []).append(result)

	output = []

	for day in sorted(grouped.keys()):
		day_results = grouped[day]
		sentiments = [
			result["sentiment_score"]
			for result in day_results
			if result.get("sentiment_score") is not None
		]

		avg_sentiment = round(sum(sentiments) / len(sentiments), 4) if sentiments else None

		output.append({
			"date": day,
			"entry_count": len(day_results),
			"avg_sentiment": avg_sentiment,
			"sentiment_distribution": build_sentiment_distribution(day_results),
			"top_themes": build_recurring_themes(day_results),
			"top_issues": build_recurring_issues(day_results),
		})

	return output


def get_entry_participant_display_name(entry: DiaryEntry) -> str:
	"""
	Return the human-readable participant name for an entry.
	"""
	display_name = (getattr(entry, "participant_display_name", "") or "").strip()

	if display_name:
		return display_name

	participant = getattr(entry, "participant", None)

	if participant:
		full_name = (participant.get_full_name() or "").strip()

		if full_name:
			return full_name

		username = (participant.get_username() or "").strip()

		if username:
			return username

	return "Anonymous"


def serialize_dashboard_entry(entry: DiaryEntry, result: dict) -> dict:
	"""
	Serialize one entry for dashboard sentiment highlights.
	"""
	quote = (entry.content or "").strip()

	if len(quote) > 240:
		quote = quote[:240].rstrip() + "..."

	participant_name = get_entry_participant_display_name(entry)

	return {
		"entry_id": entry.id,
		"summary": result.get("entry_summary") or quote,
		"quote": quote,
		"sentiment_score": result.get("sentiment_score"),
		"sentiment_label": result.get("sentiment_label"),
		"theme_label": result.get("theme_label"),
		"theme_weight": result.get("theme_weight"),
		"issue_tags": result.get("analysis_issue_tags", []),
		"created_at": entry.created_at.isoformat() if entry.created_at else None,
		"created_at_display": entry.created_at.strftime("%b %d, %Y") if entry.created_at else "—",
		"participant_name": participant_name,
	}


def build_top_sentiment_entries(entries_with_results: list[tuple]) -> dict:
	"""
	Return only the top positive and top negative entries for dashboard display.
	"""
	scored_entries = [
		(entry, result)
		for entry, result in entries_with_results
		if result.get("sentiment_score") is not None
	]

	top_positive = None
	top_negative = None

	positive_entries = [
		(entry, result)
		for entry, result in scored_entries
		if result["sentiment_score"] > 0
	]

	negative_entries = [
		(entry, result)
		for entry, result in scored_entries
		if result["sentiment_score"] < 0
	]

	if positive_entries:
		entry, result = max(
			positive_entries,
			key=lambda pair: pair[1]["sentiment_score"],
		)
		top_positive = serialize_dashboard_entry(entry, result)

	if negative_entries:
		entry, result = min(
			negative_entries,
			key=lambda pair: pair[1]["sentiment_score"],
		)
		top_negative = serialize_dashboard_entry(entry, result)

	return {
		"positive": top_positive,
		"negative": top_negative,
	}


def normalize_catalog_label(label: str | None) -> str:
	return " ".join((label or "").strip().split())


def get_enum_value(enum_class: Any, *candidate_names: str):
	"""
	Return the first enum value that exists.

	This keeps this service tolerant of small enum naming changes without hard-coding
	a brittle source/status value in the analysis runner.
	"""
	for candidate_name in candidate_names:
		if hasattr(enum_class, candidate_name):
			return getattr(enum_class, candidate_name)

	return None


def get_catalog_create_defaults() -> dict:
	defaults = {
		"is_active": True,
	}

	source = get_enum_value(
		ThemeAndIssueSource,
		"MACHINE",
		"MACHINE_ANALYSIS",
		"AUTOMATIC",
		"GENERATED",
		"SYSTEM",
		"LLM",
		"NLP",
	)

	status = get_enum_value(
		ThemeAndIssueStatus,
		"SUGGESTED",
		"APPROVED",
		"DRAFT",
		"ACTIVE",
	)

	if source is not None:
		defaults["source"] = source

	if status is not None:
		defaults["status"] = status

	return defaults


def get_or_create_canonical_theme_from_analyzer(theme_label: str | None) -> CanonicalTheme | None:
	"""
	Get or create a canonical theme using the exact analyzer label.

	This intentionally does NOT:
	- compare against aliases
	- run TF-IDF
	- run cosine similarity
	- map to a broader catalogue item

	The analyzer label is the source of truth.
	"""
	theme_label = normalize_catalog_label(theme_label)

	if not theme_label:
		return None

	existing_theme = (
		CanonicalTheme.objects
		.filter(name__iexact=theme_label)
		.first()
	)

	if existing_theme:
		return existing_theme

	return CanonicalTheme.objects.create(
		name=theme_label,
		description="Created automatically from analyzer output.",
		aliases=[],
		examples="",
		**get_catalog_create_defaults(),
	)


def attach_analyzer_theme_to_entry_analysis(
	entry_analysis: DiaryEntryAnalysis,
	theme_label: str | None,
	confidence_score: float | None,
) -> CanonicalTheme | None:
	canonical_theme = get_or_create_canonical_theme_from_analyzer(theme_label)

	if not canonical_theme:
		return None

	DiaryEntryAnalysisCanonicalTheme.objects.update_or_create(
		diary_entry_analysis=entry_analysis,
		canonical_theme=canonical_theme,
		defaults={
			"confidence_score": confidence_score,
			"rationale": "Created and assigned directly from analyzer theme output.",
		},
	)

	return canonical_theme


def get_or_create_canonical_issue_from_analyzer(issue_label: str | None) -> CanonicalIssue | None:
	"""
	Get or create a canonical issue using the exact analyzer label.

	This intentionally does NOT:
	- keyword-detect issues from entry text
	- compare against aliases
	- run TF-IDF
	- run cosine similarity
	- map to a broader catalogue item
	"""
	issue_label = normalize_catalog_label(issue_label)

	if not issue_label:
		return None

	existing_issue = (
		CanonicalIssue.objects
		.filter(name__iexact=issue_label)
		.first()
	)

	if existing_issue:
		return existing_issue

	return CanonicalIssue.objects.create(
		name=issue_label,
		description="Created automatically from analyzer output.",
		aliases=[],
		examples="",
		**get_catalog_create_defaults(),
	)


def attach_analyzer_issues_to_entry_analysis(
	entry_analysis: DiaryEntryAnalysis,
	issue_labels: list[str],
) -> list[CanonicalIssue]:
	canonical_issues = []

	for issue_label in issue_labels or []:
		canonical_issue = get_or_create_canonical_issue_from_analyzer(issue_label)

		if not canonical_issue:
			continue

		DiaryEntryAnalysisCanonicalIssue.objects.update_or_create(
			diary_entry_analysis=entry_analysis,
			canonical_issue=canonical_issue,
			defaults={
				"confidence_score": None,
				"rationale": "Created and assigned directly from analyzer issue output.",
			},
		)

		canonical_issues.append(canonical_issue)

	return canonical_issues


def extract_label_from_issue_item(issue_item: Any) -> str | None:
	if issue_item is None:
		return None

	if isinstance(issue_item, str):
		return issue_item

	if isinstance(issue_item, dict):
		for key in ("label", "name", "issue", "title", "category"):
			value = issue_item.get(key)

			if isinstance(value, str) and value.strip():
				return value

	return None


def extract_issue_labels_from_analyzer_result(entry_analysis_result_data: dict) -> list[str]:
	"""
	Extract issue labels only from the analyzer result payload.

	This function does not inspect entry.content. If the analyzer does not return issues,
	the entry has no machine-detected issues.
	"""
	candidate_keys = (
		"issues",
		"issue_labels",
		"issue_tags",
		"usability_issues",
		"detected_issues",
		"canonical_issues",
	)

	issue_labels = []

	for key in candidate_keys:
		value = entry_analysis_result_data.get(key)

		if not value:
			continue

		if isinstance(value, str):
			label = normalize_catalog_label(value)

			if label:
				issue_labels.append(label)

			continue

		if isinstance(value, list):
			for item in value:
				label = normalize_catalog_label(extract_label_from_issue_item(item))

				if label:
					issue_labels.append(label)

	if not issue_labels:
		issue_value = entry_analysis_result_data.get("issue")
		label = normalize_catalog_label(extract_label_from_issue_item(issue_value))

		if label:
			issue_labels.append(label)

	seen = set()
	unique_issue_labels = []

	for label in issue_labels:
		normalized_key = label.lower()

		if normalized_key in seen:
			continue

		seen.add(normalized_key)
		unique_issue_labels.append(label)

	return unique_issue_labels


def refresh_run_canonical_themes(run: StudyAnalysisRun) -> None:
	StudyAnalysisRunCanonicalTheme.objects.filter(run=run).delete()

	entry_theme_rows = (
		DiaryEntryAnalysisCanonicalTheme.objects
		.filter(diary_entry_analysis__run=run)
		.select_related("canonical_theme")
	)

	grouped = defaultdict(lambda: {"count": 0, "scores": []})

	for row in entry_theme_rows:
		theme_id = row.canonical_theme_id
		grouped[theme_id]["count"] += 1

		if row.confidence_score is not None:
			grouped[theme_id]["scores"].append(row.confidence_score)

	for theme_id, data in grouped.items():
		scores = data["scores"]
		average_confidence_score = (
			round(sum(scores) / len(scores), 4)
			if scores
			else None
		)

		StudyAnalysisRunCanonicalTheme.objects.create(
			run=run,
			canonical_theme_id=theme_id,
			entry_count=data["count"],
			average_confidence_score=average_confidence_score,
		)


def refresh_run_canonical_issues(run: StudyAnalysisRun) -> None:
	StudyAnalysisRunCanonicalIssue.objects.filter(run=run).delete()

	entry_issue_rows = (
		DiaryEntryAnalysisCanonicalIssue.objects
		.filter(diary_entry_analysis__run=run)
		.select_related("canonical_issue")
	)

	grouped = defaultdict(lambda: {"count": 0, "scores": []})

	for row in entry_issue_rows:
		issue_id = row.canonical_issue_id
		grouped[issue_id]["count"] += 1

		if row.confidence_score is not None:
			grouped[issue_id]["scores"].append(row.confidence_score)

	for issue_id, data in grouped.items():
		scores = data["scores"]
		average_confidence_score = (
			round(sum(scores) / len(scores), 4)
			if scores
			else None
		)

		StudyAnalysisRunCanonicalIssue.objects.create(
			run=run,
			canonical_issue_id=issue_id,
			entry_count=data["count"],
			average_confidence_score=average_confidence_score,
		)


def set_run_dominant_theme(run: StudyAnalysisRun) -> None:
	top_theme = (
		StudyAnalysisRunCanonicalTheme.objects
		.filter(run=run)
		.select_related("canonical_theme")
		.order_by("-entry_count", "-average_confidence_score", "canonical_theme__name")
		.first()
	)

	run.dominant_theme = top_theme.canonical_theme if top_theme else None


# ------------------------------ #
# Create queued analysis run      #
# ------------------------------ #

def create_study_analysis_run(
	study_id: int,
	user_id: int | None = None,
	sentiment_method: str = "vader",
	theme_method: str = "tfidf_nmf",
	llm_provider: str | None = None,
	llm_model: str | None = None,
) -> StudyAnalysisRun:
	study = Study.objects.get(pk=study_id)

	is_llm_run = sentiment_method == "llm" or theme_method == "llm"

	methods = {
		"sentiment": sentiment_method,
		"theme": theme_method,
	}

	if is_llm_run:
		methods["provider"] = llm_provider
		methods["model"] = llm_model

	run = StudyAnalysisRun.objects.create(
		study=study,
		status=AnalysisRunStatus.QUEUED,
		analysis_model=f"{sentiment_method}_{theme_method}",
		analysis_version="v3",
		methods=methods,
		created_by_id=user_id,
	)

	study.status = StudyStatus.MACHINE_ANALYSIS
	study.save(update_fields=["status"])

	return run


# -------------------------------- #
# Process existing analysis run     #
# -------------------------------- #

def process_study_analysis_run(run_id: int) -> StudyAnalysisRun:
	run = StudyAnalysisRun.objects.select_related("study").get(pk=run_id)
	study = run.study

	if run.status == AnalysisRunStatus.COMPLETED:
		return run

	sentiment_method = run.methods.get("sentiment", "vader")
	theme_method = run.methods.get("theme", "tfidf_nmf")

	run.status = AnalysisRunStatus.RUNNING
	run.error_message = ""
	run.save(update_fields=["status", "error_message"])

	entry_results = []
	entries_with_results = []

	try:
		with transaction.atomic():
			entries = list(study.entries.all().order_by("created_at"))

			entries_by_id = {
				entry.id: entry
				for entry in entries
			}

			now = timezone.now()
			is_llm_run = sentiment_method == "llm" or theme_method == "llm"

			if is_llm_run:
				llm_provider = run.methods.get("provider")
				llm_model = run.methods.get("model")

				if not llm_provider:
					raise RuntimeError("An LLM provider is required for LLM analysis")

				study_analysis_result = analyze_study_entries_with_llm(
					study_id=study.id,
					entries=entries,
					provider=llm_provider,
					model=llm_model,
				)

			else:
				study_analysis_result = analyze_study_entries(
					study_id=study.id,
					entries=entries,
					sentiment_method=sentiment_method,
					theme_method=theme_method,
				)

			study_analysis_result_data = asdict(study_analysis_result)

			for entry_analysis_result in study_analysis_result.entry_analysis_results:
				entry = entries_by_id.get(entry_analysis_result.entry_id)

				if entry is None:
					continue

				entry_analysis_result_data = asdict(entry_analysis_result)

				sentiment_score = None
				sentiment_label = None

				if entry_analysis_result.sentiment:
					if entry_analysis_result.sentiment.score is not None:
						sentiment_score = round(entry_analysis_result.sentiment.score, 4)

					sentiment_label = entry_analysis_result.sentiment.label

				theme_weight = None
				theme_label = None

				if entry_analysis_result.theme:
					if entry_analysis_result.theme.weight is not None:
						theme_weight = round(entry_analysis_result.theme.weight, 4)

					theme_label = normalize_catalog_label(entry_analysis_result.theme.label)

				normalized_content = normalize_entry_text(entry.content)
				entry_summary = make_entry_summary(normalized_content)

				issue_labels = extract_issue_labels_from_analyzer_result(entry_analysis_result_data)

				selected_entry_run = DiaryEntryAnalysis.objects.create(
					run=run,
					entry=entry,

					sentiment_score=sentiment_score,
					sentiment_label=sentiment_label,
					raw_sentiment_result=asdict(entry_analysis_result.sentiment) if entry_analysis_result.sentiment else {},

					theme_weight=theme_weight,
					theme_label=theme_label,
					raw_theme_result=asdict(entry_analysis_result.theme) if entry_analysis_result.theme else {},

					issue_detected=bool(issue_labels),
					issues=issue_labels,

					methods=study_analysis_result.methods,
					metadata=study_analysis_result.metadata,

					entry_summary=entry_summary,
					analyzed_at=now,
					raw_response=entry_analysis_result_data,
				)

				canonical_theme = attach_analyzer_theme_to_entry_analysis(
					entry_analysis=selected_entry_run,
					theme_label=theme_label,
					confidence_score=theme_weight,
				)

				canonical_issues = attach_analyzer_issues_to_entry_analysis(
					entry_analysis=selected_entry_run,
					issue_labels=issue_labels,
				)

				entry.selected_entry_run = selected_entry_run
				entry.save(update_fields=["selected_entry_run"])

				result = {
					"sentiment_score": sentiment_score,
					"sentiment_label": sentiment_label,

					"theme_weight": theme_weight,
					"theme_label": theme_label,

					"canonical_theme_id": canonical_theme.id if canonical_theme else None,
					"canonical_theme_status": canonical_theme.status if canonical_theme else None,

					"analysis_issue_detected": bool(issue_labels),
					"analysis_issue_tags": issue_labels,
					"canonical_issue_ids": [
						canonical_issue.id
						for canonical_issue in canonical_issues
					],

					"entry_summary": entry_summary,
				}

				entry_results.append(result)
				entries_with_results.append((entry, result))

			now = timezone.now()

			run.average_sentiment_label = study_analysis_result.average_sentiment_label
			run.average_sentiment_score = (
				round(study_analysis_result.average_sentiment_score, 4)
				if study_analysis_result.average_sentiment_score is not None
				else None
			)

			run.dominant_sentiment_label = study_analysis_result.dominant_sentiment_label
			run.dominant_sentiment_score = (
				round(study_analysis_result.dominant_sentiment_score, 4)
				if study_analysis_result.dominant_sentiment_score is not None
				else None
			)

			run.sentiment_distribution = build_sentiment_distribution(entry_results)

			refresh_run_canonical_themes(run)
			refresh_run_canonical_issues(run)
			set_run_dominant_theme(run)

			run.evolution_over_time = build_evolution_over_time(entries_with_results)
			run.top_representative_quotes = build_top_sentiment_entries(entries_with_results)

			run.total_entries = study_analysis_result.total_entries
			run.total_themes = study_analysis_result.total_themes

			run.entry_analysis_results = study_analysis_result_data["entry_analysis_results"]
			run.methods = study_analysis_result.methods
			run.metadata = study_analysis_result.metadata

			run.status = AnalysisRunStatus.COMPLETED
			run.completed_at = now

			run.save(
				update_fields=[
					"average_sentiment_label",
					"average_sentiment_score",
					"dominant_sentiment_label",
					"dominant_sentiment_score",
					"sentiment_distribution",

					"dominant_theme",

					"evolution_over_time",
					"top_representative_quotes",

					"entry_analysis_results",
					"total_entries",
					"total_themes",
					"methods",
					"metadata",
					"status",
					"completed_at",
				]
			)

			refresh_sentiment_confusion_matrix_for_run(run)

			study.selected_study_run = run
			study.status = StudyStatus.HUMAN_ANALYSIS
			study.save(
				update_fields=[
					"selected_study_run",
					"status",
				]
			)

		return run

	except Exception as e:
		run.status = AnalysisRunStatus.FAILED
		run.error_message = str(e)
		run.completed_at = timezone.now()
		run.save(update_fields=["status", "error_message", "completed_at"])

		raise


# ------------------------------ #
# Backward-compatible sync call   #
# ------------------------------ #

def run_study_analysis(
	study_id: int,
	user_id: int | None = None,
	sentiment_method: str = "vader",
	theme_method: str = "tfidf_nmf",
	llm_provider: str | None = None,
	llm_model: str | None = None,
) -> StudyAnalysisRun:
	run = create_study_analysis_run(
		study_id=study_id,
		user_id=user_id,
		sentiment_method=sentiment_method,
		theme_method=theme_method,
		llm_provider=llm_provider,
		llm_model=llm_model,
	)

	return process_study_analysis_run(run.pk)
