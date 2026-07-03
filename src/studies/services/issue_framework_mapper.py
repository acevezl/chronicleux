from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable

from django.db import transaction
from django.db.models import Q

from studies.models import (
	CanonicalIssue,
	CanonicalIssueToFrameworkMapping,
	ThemeAndIssueStatus,
	UXFrameworkMappingMethod,
	UXFrameworkMappingStatus,
	UXFrameworkCriterion,
)

MIN_SCORE = 0.25
MAX_MAPPINGS_PER_ISSUE = 8
UPDATE_EXISTING_SYSTEM_SUGGESTIONS = True
TEXT_SIMILARITY_WEIGHT = 0.60
ALIAS_SCORE_WEIGHT = 0.25
KEYWORD_SCORE_WEIGHT = 0.15
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"

# --------------- #
# RESULT CONTRACT #
# --------------- #
@dataclass
class IssueFrameworkMappingResult:
	issues_scanned: int = 0
	criteria_scanned: int = 0
	candidates_scored: int = 0
	mappings_created: int = 0
	mappings_updated: int = 0
	mappings_skipped_existing: int = 0
	mappings_below_threshold: int = 0
	errors: list[str] = field(default_factory=list)


@dataclass
class IssueCriterionScore:
	score: float
	text_similarity: float
	alias_score: float
	keyword_score: float
	rationale: str


# ---------- #
# PUBLIC API #
# ---------- #
@transaction.atomic
def map_all_canonical_issues_to_framework_criteria(
	*,
	min_score = MIN_SCORE,
	max_mappings_per_issue = MAX_MAPPINGS_PER_ISSUE,
	created_by=None,
	update_existing_system_suggestions = UPDATE_EXISTING_SYSTEM_SUGGESTIONS,
) -> IssueFrameworkMappingResult:

	result = IssueFrameworkMappingResult()

	issues = list(
		CanonicalIssue.objects
		.filter(is_active=True)
		.exclude(status=ThemeAndIssueStatus.REJECTED)
		.order_by("name")
	)

	criteria = list(
		UXFrameworkCriterion.objects
		.select_related("framework")
		.filter(
			is_active=True,
			framework__is_active=True,
		)
		.order_by("framework__name", "code", "name")
	)

	result.issues_scanned = len(issues)
	result.criteria_scanned = len(criteria)

	if not issues or not criteria:
		return result

	for issue in issues:
		issue_result = map_canonical_issue_to_framework_criteria(
			issue,
			criteria=criteria,
			min_score=min_score,
			max_mappings=max_mappings_per_issue,
			created_by=created_by,
			update_existing_system_suggestions=update_existing_system_suggestions,
		)

		result.candidates_scored += issue_result.candidates_scored
		result.mappings_created += issue_result.mappings_created
		result.mappings_updated += issue_result.mappings_updated
		result.mappings_skipped_existing += issue_result.mappings_skipped_existing
		result.mappings_below_threshold += issue_result.mappings_below_threshold
		result.errors.extend(issue_result.errors)

	return result


@transaction.atomic
def map_canonical_issue_to_framework_criteria(
	issue: CanonicalIssue,
	*,
	criteria: Iterable[UXFrameworkCriterion] | None = None,
	min_score = MIN_SCORE,
	max_mappings: int = 8,
	created_by=None,
	update_existing_system_suggestions: bool = True,
) -> IssueFrameworkMappingResult:

	result = IssueFrameworkMappingResult(issues_scanned=1)

	if criteria is None:
		criteria = UXFrameworkCriterion.objects.select_related("framework").filter(
			is_active=True,
			framework__is_active=True,
		)

	criteria = list(criteria)
	result.criteria_scanned = len(criteria)

	scored_candidates: list[tuple[UXFrameworkCriterion, IssueCriterionScore]] = []

	for criterion in criteria:
		try:
			scored = score_issue_against_framework_criterion(issue, criterion)
			result.candidates_scored += 1

			if scored.score < min_score:
				result.mappings_below_threshold += 1
				continue

			scored_candidates.append((criterion, scored))

		except Exception as exc:
			result.errors.append(
				f"Could not score issue '{issue}' against criterion '{criterion}': {exc}"
			)

	scored_candidates.sort(key=lambda item: item[1].score, reverse=True)

	for criterion, scored in scored_candidates[:max_mappings]:
		created_or_updated = create_or_update_issue_framework_mapping(
			issue=issue,
			criterion=criterion,
			scored=scored,
			created_by=created_by,
			update_existing_system_suggestions=update_existing_system_suggestions,
		)

		if created_or_updated == "created":
			result.mappings_created += 1
		elif created_or_updated == "updated":
			result.mappings_updated += 1
		elif created_or_updated == "skipped":
			result.mappings_skipped_existing += 1

	return result


# ------------------- #
# MAPPING WRITE LOGIC #
# ------------------- #
def create_or_update_issue_framework_mapping(
	*,
	issue: CanonicalIssue,
	criterion: UXFrameworkCriterion,
	scored: IssueCriterionScore,
	created_by=None,
	update_existing_system_suggestions: bool = True,
) -> str:

	existing = CanonicalIssueToFrameworkMapping.objects.filter(
		issue=issue,
		criterion=criterion,
	).first()

	if existing:
		if existing.status in {
			UXFrameworkMappingStatus.APPROVED,
			UXFrameworkMappingStatus.REJECTED,
		}:
			return "skipped"

		if existing.method == UXFrameworkMappingMethod.MANUAL:
			return "skipped"

		if not update_existing_system_suggestions:
			return "skipped"

		existing.method = UXFrameworkMappingMethod.TFIDF
		existing.status = UXFrameworkMappingStatus.SUGGESTED
		existing.score = scored.score
		existing.rationale = scored.rationale

		if created_by and not existing.created_by:
			existing.created_by = created_by

		existing.save(
			update_fields=[
				"method",
				"status",
				"score",
				"rationale",
				"created_by",
				"updated_at",
			]
		)

		return "updated"

	CanonicalIssueToFrameworkMapping.objects.create(
		issue=issue,
		criterion=criterion,
		method=UXFrameworkMappingMethod.TFIDF,
		status=UXFrameworkMappingStatus.SUGGESTED,
		score=scored.score,
		rationale=scored.rationale,
		created_by=created_by,
	)

	return "created"


# ------- #
# SCORING #
# ------- #
def score_issue_against_framework_criterion(
	issue: CanonicalIssue,
	criterion: UXFrameworkCriterion,
) -> IssueCriterionScore:
	issue_text = build_issue_mapping_text(issue)
	criterion_text = build_framework_criterion_mapping_text(criterion)

	text_similarity = calculate_text_similarity(issue_text, criterion_text)
	alias_score = calculate_alias_score(issue, criterion)
	keyword_score = calculate_keyword_score(issue_text, criterion_text)

	final_score = (
		(TEXT_SIMILARITY_WEIGHT * text_similarity)
		+ (ALIAS_SCORE_WEIGHT * alias_score)
		+ (KEYWORD_SCORE_WEIGHT * keyword_score)
	)

	final_score = round(min(max(final_score, 0.0), 1.0), 4)

	rationale = build_mapping_rationale(
		issue=issue,
		criterion=criterion,
		score=final_score,
		text_similarity=text_similarity,
		alias_score=alias_score,
		keyword_score=keyword_score,
	)

	return IssueCriterionScore(
		score=final_score,
		text_similarity=text_similarity,
		alias_score=alias_score,
		keyword_score=keyword_score,
		rationale=rationale,
	)


def calculate_text_similarity(text_a: str, text_b: str) -> float:

	text_a = normalize_text(text_a)
	text_b = normalize_text(text_b)

	if not text_a or not text_b:
		return 0.0

	try:
		from sklearn.feature_extraction.text import TfidfVectorizer
		from sklearn.metrics.pairwise import cosine_similarity

		vectorizer = TfidfVectorizer(
			stop_words=TFIDF_STOP_WORDS,
			ngram_range=TFIDF_NGRAM_RANGE,
			min_df=1,
		)

		matrix = vectorizer.fit_transform([text_a, text_b])
		score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

		return round(float(score), 4)

	except Exception:
		return round(SequenceMatcher(None, text_a, text_b).ratio(), 4)


def calculate_alias_score(
	issue: CanonicalIssue,
	criterion: UXFrameworkCriterion,
) -> float:
	issue_aliases = normalize_terms(issue.aliases)
	criterion_aliases = normalize_terms(criterion.aliases)

	issue_terms = {
		normalize_text(issue.name),
		*issue_aliases,
	}

	criterion_terms = {
		normalize_text(criterion.name),
		normalize_text(criterion.code),
		*criterion_aliases,
	}

	issue_terms = {term for term in issue_terms if term}
	criterion_terms = {term for term in criterion_terms if term}

	if not issue_terms or not criterion_terms:
		return 0.0

	exact_hits = issue_terms.intersection(criterion_terms)

	if exact_hits:
		return 1.0

	partial_hits = 0

	for issue_term in issue_terms:
		for criterion_term in criterion_terms:
			if not issue_term or not criterion_term:
				continue

			if issue_term in criterion_term or criterion_term in issue_term:
				partial_hits += 1
				continue

			if SequenceMatcher(None, issue_term, criterion_term).ratio() >= 0.82:
				partial_hits += 1

	return round(min(partial_hits / max(len(issue_terms), 1), 1.0), 4)


def calculate_keyword_score(issue_text: str, criterion_text: str) -> float:
	issue_keywords = extract_keywords(issue_text)
	criterion_keywords = extract_keywords(criterion_text)

	if not issue_keywords or not criterion_keywords:
		return 0.0

	overlap = issue_keywords.intersection(criterion_keywords)
	union = issue_keywords.union(criterion_keywords)

	if not union:
		return 0.0

	return round(len(overlap) / len(union), 4)


# ------------- #
# TEXT BUILDERS #
# ------------- #
def build_issue_mapping_text(issue: CanonicalIssue) -> str:
	parts = [
		issue.name,
		issue.description,
		join_json_list(issue.aliases),
		issue.examples,
	]

	return normalize_text(" ".join(part for part in parts if part))


def build_framework_criterion_mapping_text(criterion: UXFrameworkCriterion) -> str:
	framework = criterion.framework

	parts = [
		framework.name,
		framework.description,
		framework.framework_type,
		framework.source,
		framework.version,
		criterion.code,
		criterion.name,
		criterion.description,
		join_json_list(criterion.aliases),
		criterion.examples,
		criterion.recommendation_guidance,
		join_json_list(criterion.evaluation_questions),
	]

	return normalize_text(" ".join(part for part in parts if part))


def join_json_list(value) -> str:
	if not value:
		return ""

	if isinstance(value, list):
		return " ".join(str(item) for item in value if item)

	if isinstance(value, tuple):
		return " ".join(str(item) for item in value if item)

	return str(value)


# ----------------- #
# MAPPING RATIONALE #
# ----------------- #
def build_mapping_rationale(
	*,
	issue: CanonicalIssue,
	criterion: UXFrameworkCriterion,
	score: float,
	text_similarity: float,
	alias_score: float,
	keyword_score: float,
) -> str:
	shared_keywords = sorted(
		extract_keywords(build_issue_mapping_text(issue)).intersection(
			extract_keywords(build_framework_criterion_mapping_text(criterion))
		)
	)

	keyword_preview = ", ".join(shared_keywords[:8])

	reasons = []

	if text_similarity >= MIN_SCORE - 0.02:
		reasons.append("the issue description is textually similar to the criterion metadata")

	if alias_score >= 0.50:
		reasons.append("the issue aliases overlap with the criterion name or aliases")

	if keyword_score >= 0.08 and keyword_preview:
		reasons.append(f"both records share relevant terms: {keyword_preview}")

	if not reasons:
		reasons.append("the combined similarity score passed the automatic mapping threshold")

	return (
		f"Suggested by TF-IDF issue-to-framework mapper with score {score:.2f}. "
		f"The mapping was suggested because {', and '.join(reasons)}. "
		f"Signals: text_similarity={text_similarity:.2f}, "
		f"alias_score={alias_score:.2f}, keyword_score={keyword_score:.2f}."
	)


# --------------------- #
# NORMALIZATION HELPERS #
# --------------------- #
STOPWORDS = {
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"has",
	"have",
	"how",
	"in",
	"into",
	"is",
	"it",
	"its",
	"of",
	"on",
	"or",
	"that",
	"the",
	"their",
	"this",
	"to",
	"use",
	"used",
	"user",
	"users",
	"using",
	"when",
	"where",
	"which",
	"with",
	"without",
}


def normalize_text(value: str | None) -> str:
	if not value:
		return ""

	value = str(value).lower()
	value = value.replace("_", " ")
	value = value.replace("-", " ")
	value = re.sub(r"[^a-z0-9áéíóúüñç\s]", " ", value, flags=re.IGNORECASE)
	value = re.sub(r"\s+", " ", value).strip()

	return value


def normalize_terms(value) -> set[str]:
	if not value:
		return set()

	if isinstance(value, str):
		values = [value]
	elif isinstance(value, Iterable):
		values = list(value)
	else:
		values = [str(value)]

	return {
		normalize_text(item)
		for item in values
		if normalize_text(item)
	}


def extract_keywords(text: str) -> set[str]:
	text = normalize_text(text)

	tokens = {
		token
		for token in text.split()
		if len(token) >= 4 and token not in STOPWORDS
	}

	# Add common UX bigrams so phrases like "error recovery" or
	# "cognitive load" can influence the score.
	words = text.split()
	bigrams = {
		f"{words[index]} {words[index + 1]}"
		for index in range(len(words) - 1)
		if words[index] not in STOPWORDS
		and words[index + 1] not in STOPWORDS
		and len(words[index]) >= 3
		and len(words[index + 1]) >= 3
	}

	return tokens.union(bigrams)