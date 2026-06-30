from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from django.db import transaction
from django.utils import timezone

from studies.models import (
	Study,
	StudyStatus,
	StudyAnalysis,
	Entry,
	EntryAnalysis,
	AnalysisStatus,
	EntryAnalysisTheme,
	StudyAnalysisTheme,
	EntryAnalysisIssue,
	StudyAnalysisIssue,
	CanonicalTheme,
	CanonicalIssue,
)

from studies.services._confusion_matrix_calculator import refresh_sentiment_confusion_matrix_for_run
from studies.services._ordinal_distance_calculator import refresh_sentiment_ordinal_distance_for_run
from studies.services._sentiment_summarizer import summarize_study_sentiment
from studies.services.nlp.pipeline import analyze_study_entries
from studies.services.llm.pipeline import analyze_study_entries_with_llm

MAX_SUMMARY_LENGTH = 180


# ---------------------
# Helper functions
# ---------------------

def normalize_entry_text(text: str) -> str:
	"""
	Light normalization of entries for better analysis results.

	This preserves the original wording as much as possible because sentiment
	analysis may rely on punctuation, casing, and phrasing.
	"""
	return " ".join((text or "").split())


def normalize_for_term_matching(text: str) -> str:
	return normalize_entry_text(text).casefold()


def make_entry_summary(text: str, max_length: int = MAX_SUMMARY_LENGTH) -> str:
	"""
	Create a short summary from the entry text.

	For now this is just a safe truncation. We can replace this later with
	a proper summary generator if needed.
	"""
	normalized = normalize_entry_text(text)

	if len(normalized) <= max_length:
		return normalized

	return normalized[:max_length].rstrip() + "..."


def build_sentiment_distribution(entry_results: list[dict]) -> list[dict]:
	"""
	Build sentiment label counts and percentages for the run.
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

def serialize_analysis_tag(result) -> dict:
	"""
	Serialize a theme/issue analyzer result into a stable JSON-friendly shape.

	This keeps both the human-readable label/name and useful metadata so the
	run-level and day-level summaries can be rebuilt without touching the M2M rows.
	"""
	if not result:
		return {}

	metadata = result.metadata or {}

	return {
		"name": result.label,
		"label": result.label,
		"weight": round(result.weight, 4) if result.weight is not None else None,
		"theme_id": metadata.get("canonical_theme_id"),
		"issue_id": metadata.get("canonical_issue_id"),
		"metadata": metadata,
	}

def build_recurring_themes(entry_results: list[dict]) -> list[dict]:
	counts = Counter()

	for result in entry_results:
		for theme in result.get("analysis_theme_tags") or []:
			theme_name = (
				theme.get("name")
				or theme.get("label")
			)

			if not theme_name:
				continue

			counts[theme_name] += 1

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
	counts = Counter()

	for result in entry_results:
		for issue in result.get("analysis_issue_tags") or []:
			issue_name = (
				issue.get("name")
				or issue.get("label")
			)

			if not issue_name:
				continue

			counts[issue_name] += 1

	total = sum(counts.values())

	return [
		{
			"label": label,
			"count": count,
			"percentage": round(count / total, 4) if total else 0,
		}
		for label, count in counts.most_common()
	]


def build_evolution_over_time(entries_with_results: list[tuple]) -> list[dict]:
	"""
	Build a time series of average sentiment and sentiment distribution by day.
	It also includes detected themes and issues by day.
	"""
	grouped = {}

	for entry, result in entries_with_results:
		if not entry.created_at:
			continue

		day = entry.created_at.date().isoformat()
		grouped.setdefault(day, []).append(result)

	output = []

	for day in sorted(grouped.keys()):
		day_results = grouped[day]

		sentiment_scores = [
			result["sentiment_score"]
			for result in day_results
			if result.get("sentiment_score") is not None
		]

		avg_sentiment = (
			round(sum(sentiment_scores) / len(sentiment_scores), 4)
			if sentiment_scores
			else None
		)

		output.append({
			"date": day,
			"entry_count": len(day_results),
			"avg_sentiment": avg_sentiment,
			"sentiment_distribution": build_sentiment_distribution(day_results),

			"themes": build_recurring_themes(day_results),
			"issues": build_recurring_issues(day_results),
		})

	return output


def get_entry_participant_display_name(entry: Entry) -> str:
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


def serialize_dashboard_entry(entry: Entry, result: dict) -> dict:
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

		"themes": result.get("analysis_theme_tags") or [],
		"issues": result.get("analysis_issue_tags") or [],

		"created_at": entry.created_at.isoformat() if entry.created_at else None,
		"created_at_display": entry.created_at.strftime("%b %d, %Y") if entry.created_at else "—",
		"participant_name": participant_name,
	}


def build_top_sentiment_entries(entries_with_results: list[tuple]) -> dict:
	"""
	Return the top positive and top negative entries for dashboard display.
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

def attach_canonical_theme_from_analyzer_result(
	entry_analysis: EntryAnalysis,
	theme_result,
) -> None:
	if not theme_result:
		return

	metadata = theme_result.metadata or {}
	theme_id = metadata.get("canonical_theme_id")

	if not theme_id:
		return

	theme = CanonicalTheme.objects.filter(pk=theme_id).first()

	EntryAnalysisTheme.objects.update_or_create(
		entry_analysis=entry_analysis,
		theme_id=theme_id,
		defaults={
			"confidence_score": theme_result.weight,
			"rationale": build_theme_rationale(theme_result, theme),
			"assigned_by_method": theme_result.method,
		},
	)

def attach_canonical_issue_from_analyzer_result(
	entry_analysis: EntryAnalysis,
	issue_result,
) -> None:
	if not issue_result:
		return

	metadata = issue_result.metadata or {}
	issue_id = metadata.get("canonical_issue_id")

	if not issue_id:
		return

	EntryAnalysisIssue.objects.update_or_create(
		entry_analysis=entry_analysis,
		issue_id=issue_id,
		defaults={
			"confidence_score": issue_result.weight,
			"rationale": build_issue_rationale(entry_analysis.entry.content, issue_result),
			"assigned_by_method": issue_result.method,
		},
	)


def find_present_terms(entry_text: str, terms: list[str]) -> list[str]:
	normalized_entry = normalize_for_term_matching(entry_text)

	present_terms = []

	for term in terms or []:
		clean_term = normalize_entry_text(term)

		if not clean_term:
			continue

		if normalize_for_term_matching(clean_term) in normalized_entry:
			present_terms.append(clean_term)

	return list(dict.fromkeys(present_terms))


def build_theme_rationale(theme_result, canonical_theme=None) -> str:
	if not theme_result:
		return "Assigned from NLP analyzer output."

	metadata = theme_result.metadata or {}

	generated_topic = (
		metadata.get("original_nmf_theme", {}).get("label")
		or theme_result.label
		or "Unknown topic"
	)

	canonical_theme_name = (
		getattr(canonical_theme, "name", None)
		or metadata.get("suggested_theme", {}).get("name")
		or theme_result.label
		or "Unknown canonical theme"
	)

	match_score = metadata.get("catalog_match_weight")

	if match_score is not None:
		return (
			f'Selected because the generated topic "{generated_topic}" '
			f'matched the canonical theme "{canonical_theme_name}" '
			f"with a similarity score of {match_score:.2f}."
		)

	return (
		f'Selected because the generated topic "{generated_topic}" '
		f'matched the canonical theme "{canonical_theme_name}".'
	)


def build_issue_rationale(entry_text: str, issue_result) -> str:
	if not issue_result:
		return "Assigned from NLP analyzer output."

	metadata = issue_result.metadata or {}

	candidate_terms = []

	candidate_terms.append(issue_result.label)
	candidate_terms.extend(getattr(issue_result, "keywords", []) or [])
	candidate_terms.extend(metadata.get("matched_keywords") or [])

	original_candidate = metadata.get("original_candidate") or {}
	candidate_terms.append(original_candidate.get("label"))
	candidate_terms.extend(original_candidate.get("keywords") or [])

	assignment = metadata.get("assignment") or {}
	candidate_terms.append(assignment.get("source_candidate_label"))
	candidate_terms.extend(assignment.get("source_candidate_keywords") or [])

	present_terms = find_present_terms(entry_text, candidate_terms)

	if present_terms:
		quoted_terms = ", ".join(f'"{term}"' for term in present_terms[:5])
		return f"Selected because this entry contains issue-related terms: {quoted_terms}."

	match_type = metadata.get("match_type")
	match_score = metadata.get("catalog_match_weight")

	if match_type and match_score is not None:
		return (
			f"Selected because the generated issue candidate matched this canonical issue "
			f"using {match_type} matching with a score of {match_score:.2f}."
		)

	return "Selected because the NLP analyzer associated this entry with the issue."


# -------------------------------
# Create queued analysis run
# -------------------------------

def create_study_analysis_run(
	study_id: int,
	user_id: int | None = None,
	sentiment_method: str = "vader",
	theme_method: str = "tfidf_nmf",
	issue_method: str = "tfidf",
	llm_provider: str | None = None,
	llm_model: str | None = None,
) -> StudyAnalysis:
	study = Study.objects.get(pk=study_id)

	is_llm_run = (
		sentiment_method == "llm"
		or theme_method == "llm"
		or issue_method == "llm"
	)

	methods = {
		"sentiment": sentiment_method,
		"theme": theme_method,
		"issue": issue_method,
	}

	if is_llm_run:
		methods["provider"] = llm_provider
		methods["model"] = llm_model

	run = StudyAnalysis.objects.create(
		study=study,
		status=AnalysisStatus.QUEUED,
		analysis_model=f"{sentiment_method}_{theme_method}_{issue_method}",
		analysis_version="v3",
		methods=methods,
		created_by_id=user_id,
	)

	study.status = StudyStatus.ANALYZING
	study.save(update_fields=["status"])

	return run


def refresh_run_canonical_themes(run: StudyAnalysis) -> None:
	StudyAnalysisTheme.objects.filter(run=run).delete()

	entry_theme_rows = (
		EntryAnalysisTheme.objects
		.filter(entry_analysis__run=run)
		.select_related("theme", "entry_analysis")
	)

	grouped = {}

	for row in entry_theme_rows:
		theme_id = row.theme_id

		if theme_id not in grouped:
			grouped[theme_id] = {
				"count": 0,
				"confidence_scores": [],
				"sentiment_scores": [],
			}

		grouped[theme_id]["count"] += 1

		if row.confidence_score is not None:
			grouped[theme_id]["confidence_scores"].append(row.confidence_score)

		if row.entry_analysis.sentiment_score is not None:
			grouped[theme_id]["sentiment_scores"].append(row.entry_analysis.sentiment_score)

	for theme_id, data in grouped.items():
		confidence_scores = data["confidence_scores"]
		sentiment_scores = data["sentiment_scores"]

		average_confidence_score = (
			round(sum(confidence_scores) / len(confidence_scores), 4)
			if confidence_scores
			else None
		)

		average_sentiment_score = (
			round(sum(sentiment_scores) / len(sentiment_scores), 4)
			if sentiment_scores
			else None
		)

		StudyAnalysisTheme.objects.create(
			run=run,
			theme_id=theme_id,
			entry_count=data["count"],
			average_confidence_score=average_confidence_score,
			average_sentiment_score=average_sentiment_score,
		)

def refresh_run_canonical_issues(run: StudyAnalysis) -> None:
	StudyAnalysisIssue.objects.filter(run=run).delete()

	entry_issue_rows = (
		EntryAnalysisIssue.objects
		.filter(entry_analysis__run=run)
		.select_related("issue", "entry_analysis")
	)

	grouped = {}

	for row in entry_issue_rows:
		issue_id = row.issue_id

		if issue_id not in grouped:
			grouped[issue_id] = {
				"count": 0,
				"confidence_scores": [],
				"sentiment_scores": [],
			}

		grouped[issue_id]["count"] += 1

		if row.confidence_score is not None:
			grouped[issue_id]["confidence_scores"].append(row.confidence_score)

		if row.entry_analysis.sentiment_score is not None:
			grouped[issue_id]["sentiment_scores"].append(row.entry_analysis.sentiment_score)

	for issue_id, data in grouped.items():
		confidence_scores = data["confidence_scores"]
		sentiment_scores = data["sentiment_scores"]

		average_confidence_score = (
			round(sum(confidence_scores) / len(confidence_scores), 4)
			if confidence_scores
			else None
		)

		average_sentiment_score = (
			round(sum(sentiment_scores) / len(sentiment_scores), 4)
			if sentiment_scores
			else None
		)

		StudyAnalysisIssue.objects.create(
			run=run,
			issue_id=issue_id,
			entry_count=data["count"],
			average_confidence_score=average_confidence_score,
			average_sentiment_score=average_sentiment_score,
		)

def set_run_dominant_theme(run: StudyAnalysis) -> None:
	top_theme = (
		StudyAnalysisTheme.objects
		.filter(run=run)
		.select_related("theme")
		.order_by("-entry_count", "-average_confidence_score", "theme__name")
		.first()
	)

	run.dominant_theme = top_theme.theme if top_theme else None

# -------------------------------
# Process existing analysis run
# -------------------------------

def process_study_analysis_run(run_id: int) -> StudyAnalysis:
	run = StudyAnalysis.objects.select_related("study").get(pk=run_id)
	study = run.study

	if run.status == AnalysisStatus.COMPLETED:
		return run

	sentiment_method = run.methods.get("sentiment", "vader")
	theme_method = run.methods.get("theme", "tfidf_nmf")
	issue_method = run.methods.get("issue", "tfidf")

	run.status = AnalysisStatus.RUNNING
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

			is_llm_run = (
				sentiment_method == "llm"
				or theme_method == "llm"
				or issue_method == "llm"
			)

			if is_llm_run:
				llm_provider = run.methods.get("provider")
				llm_model = run.methods.get("model")

				if not llm_provider:
					raise RuntimeError("An LLM provider is required for LLM analysis.")

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
					issue_method=issue_method,
				)

			for entry_analysis_result in study_analysis_result.entry_analysis_results:
				entry = entries_by_id.get(entry_analysis_result.entry_id)

				if entry is None:
					continue

				entry_analysis_result_data = asdict(entry_analysis_result)

				# debug
				print (entry_analysis_result)
				print ("\n\n----\n\n")
				print (entry_analysis_result_data)

				sentiment_score = None
				sentiment_label = None

				if entry_analysis_result.sentiment:
					if entry_analysis_result.sentiment.score is not None:
						sentiment_score = round(entry_analysis_result.sentiment.score, 4)

					sentiment_label = entry_analysis_result.sentiment.label

				normalized_content = normalize_entry_text(entry.content)
				entry_summary = make_entry_summary(normalized_content)

				selected_entry_run = EntryAnalysis.objects.create(
					run=run,
					entry=entry,

					sentiment_score=sentiment_score,
					sentiment_label=sentiment_label,
					raw_sentiment_result=(
						asdict(entry_analysis_result.sentiment)
						if entry_analysis_result.sentiment
						else {}
					),

					methods=study_analysis_result.methods,
					metadata=study_analysis_result.metadata,

					entry_summary=entry_summary,
					analyzed_at=now,
					raw_response=entry_analysis_result_data,
				)

				for theme_result in entry_analysis_result.themes or []:
					attach_canonical_theme_from_analyzer_result(
						entry_analysis=selected_entry_run,
						theme_result=theme_result,
					)

				for issue_result in entry_analysis_result.issues or []:
					attach_canonical_issue_from_analyzer_result(
						entry_analysis=selected_entry_run,
						issue_result=issue_result,
					)

				analysis_theme_tags = [
					serialize_analysis_tag(theme_result)
					for theme_result in entry_analysis_result.themes or []
				]

				analysis_issue_tags = [
					serialize_analysis_tag(issue_result)
					for issue_result in entry_analysis_result.issues or []
				]

				entry.selected_entry_run = selected_entry_run
				entry.save(update_fields=["selected_entry_run"])

				result = {
					"sentiment_score": sentiment_score,
					"sentiment_label": sentiment_label,

					"analysis_theme_tags": analysis_theme_tags,
					"analysis_issue_tags": analysis_issue_tags,

					"entry_summary": entry_summary,
				}

				entry_results.append(result)
				entries_with_results.append((entry, result))

			now = timezone.now()

			summarize_study_sentiment(study)
			
			run.participant_average_sentiment_score = study.participant_reported_average_sentiment_score
			run.participant_average_sentiment_label = study.participant_reported_average_sentiment_label
			run.participant_dominant_sentiment_score = study.participant_reported_dominant_sentiment_score
			run.participant_dominant_sentiment_label = study.participant_reported_dominant_sentiment_label
			run.participant_sentiment_distribution = study.participant_sentiment_distribution

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

			refresh_sentiment_ordinal_distance_for_run(run, study)
			refresh_sentiment_confusion_matrix_for_run(run)
			
			run.sentiment_distribution = build_sentiment_distribution(entry_results)

			run.total_themes = study_analysis_result.total_themes
			refresh_run_canonical_themes(run)
			set_run_dominant_theme(run)

			run.total_issues = study_analysis_result.total_issues
			refresh_run_canonical_issues(run)

			run.evolution_over_time = build_evolution_over_time(entries_with_results)
			run.top_representative_quotes = build_top_sentiment_entries(entries_with_results)

			run.total_entries = study_analysis_result.total_entries

			run.methods = study_analysis_result.methods
			run.metadata = study_analysis_result.metadata

			run.status = AnalysisStatus.COMPLETED
			run.completed_at = now

			run.save(
				update_fields=[
					"participant_average_sentiment_score",
					"participant_average_sentiment_label",
					"participant_dominant_sentiment_score",
					"participant_dominant_sentiment_label",
					"participant_sentiment_distribution",
					"average_sentiment_label",
					"average_sentiment_score",
					"dominant_sentiment_label",
					"dominant_sentiment_score",
					"sentiment_distribution",

					"dominant_theme",

					"evolution_over_time",
					"top_representative_quotes",

					"total_entries",
					"total_themes",
					"total_issues",
					"methods",
					"metadata",
					"status",
					"completed_at",
				]
			)

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
		run.status = AnalysisStatus.FAILED
		run.error_message = str(e)
		run.completed_at = timezone.now()
		run.save(update_fields=["status", "error_message", "completed_at"])

		raise


# -------------------------------
# Backward-compatible sync call
# -------------------------------

def run_study_analysis(
	study_id: int,
	user_id: int | None = None,
	sentiment_method: str = "vader",
	theme_method: str = "tfidf_nmf",
	issue_method: str = "tfidf",
	llm_provider: str | None = None,
	llm_model: str | None = None,
) -> StudyAnalysis:
	run = create_study_analysis_run(
		study_id=study_id,
		user_id=user_id,
		sentiment_method=sentiment_method,
		theme_method=theme_method,
		issue_method=issue_method,
		llm_provider=llm_provider,
		llm_model=llm_model,
	)

	return process_study_analysis_run(run.pk)