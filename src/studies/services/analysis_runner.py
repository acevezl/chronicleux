from __future__ import annotations
import re

from collections import Counter, defaultdict
from dataclasses import asdict

from django.db import transaction
from django.db.models import Q
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.3
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


def normalize_catalog_text(value: str | None) -> str:
	return " ".join((value or "").lower().strip().split())


def normalize_aliases(aliases) -> list[str]:
	if not aliases:
		return []

	if isinstance(aliases, list):
		return [
			normalize_catalog_label(str(alias))
			for alias in aliases
			if normalize_catalog_label(str(alias))
		]

	if isinstance(aliases, str):
		return [
			normalize_catalog_label(alias)
			for alias in aliases.split(",")
			if normalize_catalog_label(alias)
		]

	return []


def theme_is_usable(theme: CanonicalTheme) -> bool:
	return (
		theme.is_active
		and theme.status != ThemeAndIssueStatus.REJECTED
	)


def build_catalog_theme_text(theme: CanonicalTheme) -> str:
	aliases = normalize_aliases(theme.aliases)

	return normalize_catalog_text(
		" ".join([
			theme.name or "",
			" ".join(aliases),
			theme.description or "",
			theme.examples or "",
		])
	)


def tokenize_catalog_text(value: str | None) -> set[str]:
    text = normalize_catalog_text(value)
    return {
        token
        for token in re.split(r"[^a-z0-9]+", text)
        if len(token) >= 3
    }

def find_existing_theme_by_name_or_alias(label: str | None) -> CanonicalTheme | None:
    label = normalize_catalog_label(label)

    if not label:
        return None

    label_text = normalize_catalog_text(label)
    label_tokens = tokenize_catalog_text(label)

    themes = CanonicalTheme.objects.all()

    for theme in themes:
        name_text = normalize_catalog_text(theme.name)

        if name_text == label_text:
            return theme

        aliases = normalize_aliases(theme.aliases)

        for alias in aliases:
            alias_text = normalize_catalog_text(alias)

            if alias_text == label_text:
                return theme

            if alias_text in label_text or label_text in alias_text:
                return theme

            alias_tokens = tokenize_catalog_text(alias)

            if label_tokens and alias_tokens and label_tokens & alias_tokens:
                return theme

    return None


def find_similar_existing_theme(
	label: str | None,
	similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> CanonicalTheme | None:
	label = normalize_catalog_label(label)

	if not label:
		return None

	catalog_themes = list(
		CanonicalTheme.objects
		.filter(is_active=True)
		.filter(
			Q(status=ThemeAndIssueStatus.APPROVED)
			| Q(status=ThemeAndIssueStatus.SUGGESTED)
			| Q(status__isnull=True)
			| Q(status="")
		)
	)

	if not catalog_themes:
		return None

	candidate_text = normalize_catalog_text(label)
	catalog_texts = [build_catalog_theme_text(theme) for theme in catalog_themes]

	vectorizer = TfidfVectorizer(
		stop_words="english",
		ngram_range=(1, 3),
		min_df=1,
		max_df=1.0,
	)

	matrix = vectorizer.fit_transform([candidate_text, *catalog_texts])
	similarities = cosine_similarity(matrix[0], matrix[1:])[0]

	best_index = int(similarities.argmax())
	best_score = float(similarities[best_index])

	# DEBUG
	print(
		"THEME MATCH DEBUG:",
		{
			"label": label,
			"best_match": catalog_themes[best_index].name,
			"score": round(best_score, 4),
			"threshold": similarity_threshold,
		}
	)

	if best_score < similarity_threshold:
		return None

	return catalog_themes[best_index]


def normalize_catalog_label(label: str | None) -> str:
	return " ".join((label or "").strip().split())


def get_or_create_canonical_theme(label: str | None):
	label = normalize_catalog_label(label)

	if not label:
		return None, "empty_label"

	existing_theme = find_existing_theme_by_name_or_alias(label)

	if existing_theme:
		if not existing_theme.is_active:
			return None, "theme_exists_but_inactive"

		if existing_theme.status == ThemeAndIssueStatus.REJECTED:
			return None, "theme_exists_but_rejected"

		return existing_theme, None

	similar_theme = find_similar_existing_theme(
		label,
	)

	if similar_theme:
		return similar_theme, None

	return None, "no_catalog_match"


def attach_canonical_theme_to_entry_analysis(
	entry_analysis: DiaryEntryAnalysis,
	theme_label: str | None,
	confidence_score: float | None,
) -> CanonicalTheme | None:
	canonical_theme, skip_reason = get_or_create_canonical_theme(theme_label)

	if not canonical_theme:
		entry_analysis.metadata = {
			**(entry_analysis.metadata or {}),
			"canonical_theme_assignment_skipped": {
				"label": theme_label,
				"reason": skip_reason,
			},
		}
		entry_analysis.save(update_fields=["metadata"])
		return None

	DiaryEntryAnalysisCanonicalTheme.objects.update_or_create(
		diary_entry_analysis=entry_analysis,
		canonical_theme=canonical_theme,
		defaults={
			"confidence_score": confidence_score,
			"rationale": "Assigned automatically from machine thematic analysis.",
		},
	)

	return canonical_theme


def issue_is_usable(issue: CanonicalIssue) -> bool:
	return (
		issue.is_active
		and issue.status != ThemeAndIssueStatus.REJECTED
	)


def build_catalog_issue_text(issue: CanonicalIssue) -> str:
	aliases = normalize_aliases(issue.aliases)

	return normalize_catalog_text(
		" ".join([
			issue.name or "",
			" ".join(aliases),
			issue.description or "",
			issue.examples or "",
		])
	)


def find_existing_issue_by_name_or_alias(label: str | None) -> CanonicalIssue | None:
    label = normalize_catalog_label(label)

    if not label:
        return None

    label_text = normalize_catalog_text(label)
    label_tokens = tokenize_catalog_text(label)

    issues = CanonicalIssue.objects.all()

    for issue in issues:
        name_text = normalize_catalog_text(issue.name)

        if name_text == label_text:
            return issue

        aliases = normalize_aliases(issue.aliases)

        for alias in aliases:
            alias_text = normalize_catalog_text(alias)

            if alias_text == label_text:
                return issue

            if alias_text in label_text or label_text in alias_text:
                return issue

            alias_tokens = tokenize_catalog_text(alias)

            if label_tokens and alias_tokens and label_tokens & alias_tokens:
                return issue

    return None


def find_similar_existing_issue(
	label: str | None,
	similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> CanonicalIssue | None:
	label = normalize_catalog_label(label)

	if not label:
		return None

	catalog_issues = list(
		CanonicalIssue.objects
		.filter(is_active=True)
		.filter(
			Q(status=ThemeAndIssueStatus.APPROVED)
			| Q(status=ThemeAndIssueStatus.SUGGESTED)
			| Q(status__isnull=True)
			| Q(status="")
		)
	)

	if not catalog_issues:
		return None

	candidate_text = normalize_catalog_text(label)
	catalog_texts = [build_catalog_issue_text(issue) for issue in catalog_issues]

	vectorizer = TfidfVectorizer(
		stop_words="english",
		ngram_range=(1, 3),
		min_df=1,
		max_df=1.0,
	)

	matrix = vectorizer.fit_transform([candidate_text, *catalog_texts])
	similarities = cosine_similarity(matrix[0], matrix[1:])[0]

	best_index = int(similarities.argmax())
	best_score = float(similarities[best_index])

	if best_score < similarity_threshold:
		return None

	return catalog_issues[best_index]


def get_or_create_canonical_issue(label: str | None):
	label = normalize_catalog_label(label)

	if not label:
		return None, "empty_label"

	existing_issue = find_existing_issue_by_name_or_alias(label)

	if existing_issue:
		if not existing_issue.is_active:
			return None, "issue_exists_but_inactive"

		if existing_issue.status == ThemeAndIssueStatus.REJECTED:
			return None, "issue_exists_but_rejected"

		return existing_issue, None

	similar_issue = find_similar_existing_issue(
		label,
	)

	if similar_issue:
		return similar_issue, None

	return None, "no_catalog_match"


def attach_canonical_issues_to_entry_analysis(
	entry_analysis: DiaryEntryAnalysis,
	issue_labels: list[str],
) -> list[CanonicalIssue]:
	canonical_issues = []

	skipped_issues = []

	for issue_label in issue_labels or []:
		canonical_issue, skip_reason = get_or_create_canonical_issue(issue_label)

		if not canonical_issue:
			skipped_issues.append({
				"label": issue_label,
				"reason": skip_reason,
			})
			continue

		DiaryEntryAnalysisCanonicalIssue.objects.update_or_create(
			diary_entry_analysis=entry_analysis,
			canonical_issue=canonical_issue,
			defaults={
				"confidence_score": None,
				"rationale": "Assigned automatically from keyword issue detection.",
			},
		)

		canonical_issues.append(canonical_issue)

	if skipped_issues:
		entry_analysis.metadata = {
			**(entry_analysis.metadata or {}),
			"canonical_issue_assignment_skipped": skipped_issues,
		}
		entry_analysis.save(update_fields=["metadata"])

	return canonical_issues


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


#-------------------------------#
# Create queued analysis run     #
#-------------------------------#
def create_study_analysis_run(
	study_id: int,
	user_id: int | None = None,
	sentiment_method="vader",
	theme_method="tfidf_nmf",
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
		created_by_id=user_id
	)

	study.status = StudyStatus.MACHINE_ANALYSIS
	study.save(update_fields=["status"])

	return run

#--------------------------------#
# Process existing analysis run  #
#--------------------------------#
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

					theme_label = entry_analysis_result.theme.label

				normalized_content = normalize_entry_text(entry.content)
				issue_tags = detect_issue_tags(normalized_content)
				entry_summary = make_entry_summary(normalized_content)

				selected_entry_run = DiaryEntryAnalysis.objects.create(
					run=run,
					entry=entry,

					sentiment_score=sentiment_score,
					sentiment_label=sentiment_label,
					raw_sentiment_result=asdict(entry_analysis_result.sentiment) if entry_analysis_result.sentiment else {},

					theme_weight=theme_weight,
					theme_label=theme_label,
					raw_theme_result=asdict(entry_analysis_result.theme) if entry_analysis_result.theme else {},

					issue_detected=bool(issue_tags),
					issues=issue_tags,

					methods=study_analysis_result.methods,
					metadata=study_analysis_result.metadata,

					entry_summary=entry_summary,
					analyzed_at=now,
					raw_response=entry_analysis_result_data,
				)

				attach_canonical_theme_to_entry_analysis(
					entry_analysis=selected_entry_run,
					theme_label=theme_label,
					confidence_score=theme_weight,
				)

				attach_canonical_issues_to_entry_analysis(
					entry_analysis=selected_entry_run,
					issue_labels=issue_tags,
				)

				entry.selected_entry_run = selected_entry_run
				entry.save(update_fields=["selected_entry_run"])

				result = {
					"sentiment_score": sentiment_score,
					"sentiment_label": sentiment_label,

					"theme_weight": theme_weight,
					"theme_label": theme_label,

					"analysis_issue_detected": bool(issue_tags),
					"analysis_issue_tags": issue_tags,

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

			# And safely refresh binary metrics
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


#-------------------------------#
# Backward-compatible sync call  #
#-------------------------------#
def run_study_analysis(
	study_id: int,
	user_id: int | None = None,
	sentiment_method="vader",
	theme_method="tfidf_nmf",
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