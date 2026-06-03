from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import re
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
)

from studies.services.nlp.pipeline import analyze_study_entries

DEFAULT_THEME_COUNT = 3
DEFAULT_THEME_TERMS = 6
MAX_SUMMARY_LENGTH = 180

ISSUE_CUES = [
	"problem", "issue", "bug", "broken", "confusing", "confused",
	"frustrating", "frustrated", "hard to", "difficult", "annoying",
	"slow", "lag", "crash", "stuck", "couldn't", "cannot", "can't",
	"unclear", "missing", "failed", "error", "didn't work", "trouble"
]

ISSUE_KEYWORDS = {
	"bug": [
		"bug", "bugs", "error", "errors", "crash", "crashed", "crashes",
		"broken", "breaks", "glitch", "glitched", "failed", "failure",
		"doesn't work", "did not work", "not working", "stopped working",
		"unexpected", "wrong", "issue", "problem",
	],

	"confusion": [
		"confusing", "confused", "unclear", "ambiguous", "lost",
		"didn't understand", "did not understand", "don't understand",
		"not sure", "unsure", "uncertain", "couldn't figure out",
		"could not figure out", "hard to understand", "where to",
		"what to do", "how to", "no idea", "blended together", "ambiguity",
	],

	"performance": [
		"slow", "slower", "lag", "laggy", "lagging", "delay", "delayed",
		"loading", "took too long", "takes too long", "waited",
		"freeze", "freezes", "frozen", "stuck", "hang", "hanging",
		"unresponsive", "timeout", "timed out",
	],

	"friction": [
		"hard", "difficult", "awkward", "annoying", "frustrating",
		"frustration", "tedious", "cumbersome", "clunky", "painful",
		"too many steps", "too much work", "hard to use", "difficult to use",
		"not intuitive", "counterintuitive", "inconvenient",
	],

	"navigation": [
		"couldn't find", "could not find", "can't find", "cannot find",
		"hard to find", "where is", "where are", "menu", "navigation",
		"navigate", "back", "next", "search", "filter", "hidden",
		"buried", "missing", "not visible",
	],

	"accessibility": [
		"hard to read", "small text", "too small", "contrast",
		"can't see", "cannot see", "difficult to see", "color",
		"screen reader", "keyboard", "tab", "focus", "accessible",
		"accessibility", "disabled", "vision", "hearing", "readability",
		"trouble reading"
	],

	"trust": [
		"don't trust", "do not trust", "not trustworthy", "suspicious",
		"unsafe", "privacy", "security", "scam", "fake", "unreliable",
		"concerned", "worried", "hesitant", "not confident",
	],

	"content": [
		"wrong information", "incorrect", "outdated", "missing information",
		"not enough information", "too much information", "misleading",
		"irrelevant", "not relevant", "hard to read", "poor explanation",
	],
}

# ---------------------
# Helper functions 
# ---------------------

def normalize_entry_text(text: str) -> str:
	"""
	Light normalization of entries for better analysis results. 
	I'm preserving as much of the original text as possible, since VADER relies on punctuation and casing cues for sentiment analysis. 
	This function primarily collapses excessive whitespace and ensures we have a string to work with.
	(In other words, clean and trim, but not over-clean, to preserve sentiment cues.)
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

def has_keyword_match(text: str, keywords: list[str]) -> bool:
	"""
	Matches keywords with word boundaries \b
	"""
	lowered = (text or "").lower()

	for keyword in keywords:
		pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

		if re.search(pattern, lowered):
			return True

	return False

def detect_issue_tags(text: str) -> list[str]:
	"""
	Detect issue tags using keyword matching, but only when the entry appears
	to describe an actual problem, barrier, frustration, or negative experience.
	"""
	lowered = (text or "").lower()

	if not has_keyword_match(lowered, ISSUE_CUES):
		return []

	tags: list[str] = []

	for tag, keywords in ISSUE_KEYWORDS.items():
		if has_keyword_match(lowered, keywords):
			tags.append(tag)

	return tags

def build_sentiment_distribution(entry_results: list[dict]) -> dict:
	"""
	Build a distribution of sentiment categories from the entry results, including both COUNTS and PERCENTAGES.
	Useful for understanding the sentiment landscape of the study.

	Returns:
		[
			{
				"label": "POSITIVE",
				"count": 12,
				"percentage": 0.3158,
			},
			...
		]

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
	Build a list of recurring issues based on detected issue tags in the entry results,
	including counts, percentages, negative ratio, and average sentiment.

	This helps identify common problems users are mentioning across entries in the study.

	Returns:
	[
		{
			"label": value,
			"count": value,
			"percentage": value,
			"negative_ratio": value,
			"avg_sentiment": value,
		}
	]
	"""
	counter = Counter()
	sentiment_totals = defaultdict(float)
	negative_counts = Counter()
	total_issue_entries = 0

	negative_labels = {"VERY_NEGATIVE", "NEGATIVE"}
	for result in entry_results:
		tags = result.get("analysis_issue_tags") or []

		if not tags:
			continue

		total_issue_entries += 1

		sentiment_score = result.get("sentiment_score") or 0
		sentiment_label = result.get("sentiment_label")

		for tag in tags:
			tag = str(tag).strip().lower()

			if not tag:
				continue

			counter[tag] += 1
			sentiment_totals[tag] += sentiment_score

			if sentiment_label in negative_labels:
				negative_counts[tag] += 1

	return [
		{
			"label": tag,
			"count": count,
			"percentage": round(count / total_issue_entries, 4) if total_issue_entries else 0,
			"negative_ratio": round(negative_counts[tag] / count, 4) if count else 0,
			"avg_sentiment": round(sentiment_totals[tag] / count, 4) if count else 0,
		}
		for tag, count in counter.most_common()
	]

def build_recurring_themes(entry_results: list[dict]) -> list[dict]:
	"""
	Build a list of recurring themes based on the themes detected in the entry results, including both COUNTS and PERCENTAGES.
	This helps identify common themes that are emerging across entries in the study.
	"""
	counts = Counter(
		result["theme_label"]
		for result in entry_results
		if result.get("theme_label")
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

def build_evolution_over_time(entries_with_results: list[tuple]) -> list[dict]:
	"""
	Build a time series of average sentiment and sentiment distribution over time (by day) based on the entry results.
	This helps visualize how sentiment is evolving throughout the study period.
	"""
	grouped = {}

	for entry, result in entries_with_results:
		day = entry.created_at.date().isoformat()
		grouped.setdefault(day, []).append(result)

	output = []
	for day in sorted(grouped.keys()):
		day_results = grouped[day]
		sentiments = [r["sentiment_score"] for r in day_results if r.get("sentiment_score") is not None]
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

	Imported entries may not have an associated User/participant relation,
	so participant_display_name is the canonical display value.
	"""
	display_name = (getattr(entry, "participant_display_name", "") or "").strip()

	if display_name:
		return display_name

	# Fallback only for older entries or defensive compatibility.
	participant = getattr(entry, "participant", None)

	if participant:
		full_name = (participant.get_full_name() or "").strip()
		if full_name:
			return full_name

		username = (participant.get_username() or "").strip()
		if username:
			return username

	return "Anonymous"


def serialize_dashboard_entry(entry, result: dict) -> dict:
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

#-------------------------------#
# Main analysis runner function #
#-------------------------------#
def run_study_analysis(study_id: int) -> StudyAnalysisRun:
	study = Study.objects.get(pk=study_id)

	# Future: Obtain these from dropdown
	sentiment_method = "vader"
	theme_method = "tfidf_nmf"

	run = StudyAnalysisRun.objects.create(
		study=study,
		status=AnalysisRunStatus.RUNNING,
		analysis_model=f"{sentiment_method}_{theme_method}",
		analysis_version="v3",
	)

	study.status = StudyStatus.MACHINE_ANALYSIS
	study.save(update_fields=["status"])

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
					sentiment_score = round(entry_analysis_result.sentiment.score, 4)
					sentiment_label = entry_analysis_result.sentiment.label

				theme_weight = None
				theme_label = None

				if entry_analysis_result.theme:
					theme_weight = round(entry_analysis_result.theme.weight, 4)    
					theme_label = entry_analysis_result.theme.label
				
				normalized_content = normalize_entry_text(entry.content)
				issue_tags = detect_issue_tags(normalized_content)

				entry_summary = make_entry_summary(normalized_content)
				
				selected_entry_run = DiaryEntryAnalysis.objects.create(
					run=run,
					entry=entry,
						
					sentiment_score=sentiment_score,
					sentiment_label=sentiment_label,
					raw_sentiment_result=asdict(entry_analysis_result.sentiment),
						
					theme_weight= theme_weight,
					theme_label=theme_label,
					raw_theme_result=asdict(entry_analysis_result.theme),
					
					issue_detected=bool(issue_tags),
					issues=issue_tags,

					methods = study_analysis_result.methods,
					metadata = study_analysis_result.metadata,
						
					entry_summary=entry_summary,
					analyzed_at=now,
					raw_response=entry_analysis_result_data
				)
				
				entry.selected_entry_run = selected_entry_run

				entry.save(
					update_fields=[
						"selected_entry_run"
					]
				)

				# entry.machine_sentiment_score = sentiment_score
				# entry.machine_sentiment_label = sentiment_label
				
				# entry.machine_theme_weight = theme_weight
				# entry.machine_theme_label = theme_label
				
				# entry.analysis_issue_detected = bool(issue_tags)
				# entry.analysis_issue_tags = issue_tags
				
				# entry.entry_summary = entry_summary
				
				# entry.analysis_model = run.analysis_model
				# entry.analysis_version = run.analysis_version
				# entry.analyzed_at = now

				# entry.save(
				# 	update_fields=[
				# 		"machine_sentiment_score",
				# 		"machine_sentiment_label",
				# 		"machine_theme_weight",
				# 		"machine_theme_label",
				# 		"analysis_issue_detected",
				# 		"analysis_issue_tags",
				# 		"entry_summary",
				# 		"analysis_model",
				# 		"analysis_version",
				# 		"analyzed_at",
				# 	]
				# )

				result = {
					"sentiment_score": sentiment_score,
					"sentiment_label":sentiment_label,
						
					"theme_weight": theme_weight,
					"theme_label": theme_label,
					
					"analysis_issue_detected": bool(issue_tags),
					"analysis_issue_tags": issue_tags,

					"entry_summary": entry_summary
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

			run.dominant_theme_label = study_analysis_result.dominant_theme_label
			run.dominant_theme_weight = (
				round(study_analysis_result.dominant_theme_weight, 4)
				if study_analysis_result.dominant_theme_weight is not None
				else None
			)
			run.theme_distribution = build_recurring_themes(entry_results)

			run.recurring_issues = build_recurring_issues(entry_results)

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

					"dominant_theme_label",
					"dominant_theme_weight",
					"theme_distribution",

					"recurring_issues",
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

			study.selected_study_run = run
			study.status = StudyStatus.HUMAN_ANALYSIS
			study.save(
				update_fields=[
					"selected_study_run",
					"status",
				]
			)

			# study.analysis_model = run.analysis_model
			# study.analysis_version = run.analysis_version
			# study.analyzed_at = now
			

			# study.save(
			# 	update_fields=[
			# 		"analysis_model",
			# 		"analysis_version",
			# 		"analyzed_at",
			# 		"status",

			# 		"average_sentiment_label",
			# 		"average_sentiment_score",
			# 		"dominant_sentiment_label",
			# 		"dominant_sentiment_score",
			# 		"sentiment_distribution",

			# 		"dominant_theme_label",
			# 		"dominant_theme_weight",
			# 		"theme_distribution",

			# 		"recurring_issues",
			# 		"evolution_over_time", 
			# 		"top_representative_quotes", 

			# 		"total_entries",
			# 		"total_themes",
			# 	]
			# )

		return run

	except Exception as e:
		run.status = AnalysisRunStatus.FAILED
		run.error_message = str(e)
		run.completed_at = timezone.now()
		run.save(update_fields=["status", "error_message", "completed_at"])
		raise