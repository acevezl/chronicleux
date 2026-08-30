import json
import time
from typing import Any

from django.core.serializers.json import DjangoJSONEncoder

from studies.services.contracts import (
	EntryAnalysisResult,
	IssueResult,
	SentimentResult,
	ThemeResult,
)
from studies.services.llm.client import connect_to_llm
from studies.services.nlp.sentiment._thresholds import map_sentiment_score_to_label
from studies.services.llm.llm_rate_limiter import wait_for_token_capacity

MAX_OUTPUT_TOKENS = 1200

MAX_LLM_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2

SENTIMENT_THEME_ANALYZER_SYSTEM_PROMPT = """
You are acting as an entry-level sentiment and theme analyzer for UX diary studies.

Your task is to analyze ONE diary entry at a time.

You must:
1. Analyze the sentiment of the entry and score its polarity between -1 and 1.
2. Assign exactly one best-matching theme from the provided theme catalog.

Rules:
- You must only use a theme_id value from the provided theme catalog.
- The theme catalog may include canonical themes and suggested themes. Both are valid.
- Do not create a new theme at entry-analysis time.
- Do not analyze or return UX issues.
- Do not invent recommendations.
- Return only valid JSON.
""".strip()


ISSUE_ANALYZER_SYSTEM_PROMPT = """
You are acting as an entry-level UX issue analyzer for diary studies.

Your task is to analyze ONE diary entry at a time.

You must:
1. Determine whether the entry describes a UX or usability issue.
2. If an issue is present, assign exactly one best-matching issue from the provided issue catalog.

Rules:
- You must only use an issue_id value from the provided issue catalog.
- The issue catalog may include canonical issues and suggested issues. Both are valid.
- Do not create a new issue at entry-analysis time.
- If no UX or usability issue is clearly present, return an empty issue object.
- Do not analyze or return sentiment.
- Do not analyze or return themes.
- Do not invent recommendations.
- Return only valid JSON.
""".strip()


def format_theme_catalog(theme_catalog: list[ThemeResult]) -> str:
	return json.dumps(
		[
			{
				"theme_id": f"THEME_{theme.theme_id}",
				"label": theme.label,
				"keywords": theme.keywords,
			}
			for theme in theme_catalog
		],
		ensure_ascii=False,
		separators=(",", ":"),
		cls=DjangoJSONEncoder,
	)


def format_issue_catalog(issue_catalog: list[IssueResult]) -> str:
	return json.dumps(
		[
			{
				"issue_id": f"ISSUE_{issue.issue_id}",
				"label": issue.label,
				"keywords": issue.keywords,
			}
			for issue in issue_catalog
		],
		ensure_ascii=False,
		separators=(",", ":"),
		cls=DjangoJSONEncoder,
	)


def build_sentiment_theme_prompt(
	entry,
	theme_catalog: list[ThemeResult],
) -> str:
	theme_catalog_json = format_theme_catalog(theme_catalog)

	return f"""
Analyze the following diary study entry.

Return the response using this exact JSON structure:

{{
  "entry_id": {entry.id},
  "sentiment": {{
	"score": -0.62,
	"method": "llm",
	"metadata": {{
	  "positive": 0.05,
	  "neutral": 0.55,
	  "negative": 0.40,
	  "compound": -0.62
	}}
  }},
  "theme": {{
	"theme_id": "THEME_1",
	"weight": 0.75,
	"metadata": {{
	  "language": "english",
	  "rationale": "short explanation of why this theme matches the entry"
	}}
  }},
  "metadata": {{
	"word_count": 42,
	"language": "english"
  }}
}}

Sentiment rules:
- score must be between -1.0 and 1.0.
- compound must match the score.
- Do not return a sentiment label. The application assigns it from the score.

Theme rules:
- You must choose exactly one theme from the Theme catalog.
- Return theme.theme_id exactly as shown in the Theme catalog.
- Every valid theme identifier begins with "THEME_".
- Copy the complete identifier from the catalog.
- Never construct a theme identifier from a number.
- Never return a theme identifier that is absent from the Theme catalog.
- Do not use canonical_theme_id.
- Do not use canonical_issue_id.
- If you cannot choose confidently, choose the closest valid theme from the Theme catalog.
- Never invent a theme identifier.

Metadata rules:
- Keep metadata concise.
- Do not include recommendations.
- Do not include extra top-level keys.

Return a JSON object only. Do not return anything else.

Theme catalog:
{theme_catalog_json}

Diary entry:
{{
  "entry_id": {entry.id},
  "text": {json.dumps(entry.content, ensure_ascii=False)}
}}
""".strip()


def build_issue_analyzer_prompt(
	entry,
	issue_catalog: list[IssueResult],
) -> str:
	issue_catalog_json = format_issue_catalog(issue_catalog)

	return f"""
Analyze the following diary study entry for UX or usability issues.

Return the response using one of these exact JSON structures.

When an issue is present:

{{
  "entry_id": {entry.id},
  "issue": {{
	"issue_id": "ISSUE_1",
	"weight": 0.80,
	"metadata": {{
	  "language": "english",
	  "rationale": "short explanation of why this issue matches the entry"
	}}
  }}
}}

When no UX or usability issue is present:

{{
  "entry_id": {entry.id},
  "issue": {{}}
}}

Issue rules:
- Return an issue only if the entry clearly describes UX friction, confusion, failure, accessibility problems, errors, inefficiency, dissatisfaction, or another usability problem.
- If no UX or usability issue is clearly present, return "issue": {{}}
- If an issue is present, choose exactly one issue from the Issue catalog.
- Return issue.issue_id exactly as shown in the Issue catalog.
- Every valid issue identifier begins with "ISSUE_".
- Copy the complete identifier from the catalog.
- Never construct an issue identifier from a number.
- Never return an issue identifier that is absent from the Issue catalog.
- Do not use canonical_issue_id.
- Never invent an issue identifier.
- Keep metadata concise.
- Do not include recommendations.
- Do not include extra top-level keys.

Return a JSON object only. Do not return anything else.

Issue catalog:
{issue_catalog_json}

Diary entry:
{{
  "entry_id": {entry.id},
  "text": {json.dumps(entry.content, ensure_ascii=False)}
}}
""".strip()


def get_theme_by_id(
	theme_catalog: list[ThemeResult],
	theme_id: Any,
) -> ThemeResult | None:
	if isinstance(theme_id, str):
		if not theme_id.startswith("THEME_"):
			return None

		theme_id = theme_id.removeprefix("THEME_")

	try:
		theme_id = int(theme_id)
	except (TypeError, ValueError):
		return None

	for theme in theme_catalog:
		if theme.theme_id == theme_id:
			return theme

	return None


def get_issue_by_id(
	issue_catalog: list[IssueResult],
	issue_id: Any,
) -> IssueResult | None:
	if isinstance(issue_id, str):
		if not issue_id.startswith("ISSUE_"):
			return None

		issue_id = issue_id.removeprefix("ISSUE_")

	try:
		issue_id = int(issue_id)
	except (TypeError, ValueError):
		return None

	for issue in issue_catalog:
		if issue.issue_id == issue_id:
			return issue

	return None


def clamp_score(score: Any, default: float = 0.0) -> float:
	try:
		score = float(score)
	except (TypeError, ValueError):
		score = default

	return max(-1.0, min(1.0, score))


def clamp_weight(weight: Any, default: float = 0.0) -> float:
	try:
		weight = float(weight)
	except (TypeError, ValueError):
		weight = default

	return max(0.0, min(1.0, weight))


def parse_sentiment_result(
	sentiment_data: dict,
	provider: str,
	model: str,
) -> SentimentResult:
	sentiment_score = clamp_score(sentiment_data.get("score", 0.0))
	sentiment_label = map_sentiment_score_to_label(sentiment_score)

	sentiment_metadata = dict(sentiment_data.get("metadata") or {})
	sentiment_metadata["compound"] = sentiment_score
	sentiment_metadata["model"] = model
	sentiment_metadata["provider"] = provider

	return SentimentResult(
		score=sentiment_score,
		label=sentiment_label,
		method="llm",
		metadata=sentiment_metadata,
	)


def parse_theme_result(
	theme_data: dict,
	theme_catalog: list[ThemeResult],
	provider: str,
	model: str,
	entry_id: int,
) -> ThemeResult:
	selected_theme = get_theme_by_id(
		theme_catalog=theme_catalog,
		theme_id=theme_data.get("theme_id"),
	)

	if not selected_theme:
		raw_theme_id = theme_data.get("theme_id")
		valid_theme_ids = [theme.theme_id for theme in theme_catalog]

		raise RuntimeError(
			f"LLM returned invalid theme_id '{raw_theme_id}' "
			f"for entry {entry_id}. "
			f"Valid theme_ids: {valid_theme_ids}. "
			f"Raw theme payload: {theme_data}."
		)

	theme_metadata = dict(selected_theme.metadata or {})
	theme_metadata.update(theme_data.get("metadata") or {})
	theme_metadata["model"] = model
	theme_metadata["provider"] = provider
	theme_metadata["num_keywords"] = len(selected_theme.keywords or [])

	return ThemeResult(
		theme_id=selected_theme.theme_id,
		weight=clamp_weight(theme_data.get("weight", selected_theme.weight)),
		label=selected_theme.label,
		keywords=selected_theme.keywords or [],
		method=selected_theme.method,
		metadata=theme_metadata,
	)


def parse_issue_result(
	issue_data: dict,
	issue_catalog: list[IssueResult],
	provider: str,
	model: str,
	entry_id: int,
) -> IssueResult | None:
	if not issue_data:
		return None

	selected_issue = get_issue_by_id(
		issue_catalog=issue_catalog,
		issue_id=issue_data.get("issue_id"),
	)

	if not selected_issue:
		raw_issue_id = issue_data.get("issue_id")
		valid_issue_ids = [issue.issue_id for issue in issue_catalog]

		raise RuntimeError(
			f"LLM returned invalid issue_id '{raw_issue_id}' "
			f"for entry {entry_id}. "
			f"Valid issue_ids: {valid_issue_ids}. "
			f"Raw issue payload: {issue_data}."
		)

	issue_metadata = dict(selected_issue.metadata or {})
	issue_metadata.update(issue_data.get("metadata") or {})
	issue_metadata["model"] = model
	issue_metadata["provider"] = provider
	issue_metadata["num_keywords"] = len(selected_issue.keywords or [])

	return IssueResult(
		issue_id=selected_issue.issue_id,
		weight=clamp_weight(issue_data.get("weight", selected_issue.weight)),
		label=selected_issue.label,
		keywords=selected_issue.keywords or [],
		method=selected_issue.method,
		metadata=issue_metadata,
	)


def parse_sentiment_theme_response(
		*,
		raw_content: str,
		entry,
		theme_catalog: list[ThemeResult],
		provider: str,
		model: str,
	) -> tuple[SentimentResult, ThemeResult, dict]:
	try:
		data = json.loads(raw_content)

	except json.JSONDecodeError as error:
		raise ValueError(
			"Failed to parse second-pass sentiment/theme response as JSON.\n\n"
			f"ENTRY ID: {getattr(entry, 'id', 'UNKNOWN')}\n"
			f"PROVIDER: {provider}\n"
			f"MODEL: {model}\n"
			f"JSON ERROR: {error}\n\n"
			"RAW SENTIMENT/THEME RESPONSE:\n"
			f"{raw_content}"
		) from error

	sentiment_result = parse_sentiment_result(
		sentiment_data=data.get("sentiment") or {},
		provider=provider,
		model=model,
	)

	theme_result = parse_theme_result(
		theme_data=data.get("theme") or {},
		theme_catalog=theme_catalog,
		provider=provider,
		model=model,
		entry_id=entry.id,
	)

	metadata = dict(data.get("metadata") or {})

	return sentiment_result, theme_result, metadata


def parse_issue_analyzer_response(
	*,
	raw_content: str,
	finish_reason: str | None,
	usage: Any,
	entry,
	issue_catalog: list[IssueResult],
	provider: str,
	model: str,
) -> IssueResult | None:
	try:
		data = json.loads(raw_content)

	except json.JSONDecodeError as error:
		raise ValueError(
			"Failed to parse second-pass issue response as JSON.\n\n"
			f"ENTRY ID: {getattr(entry, 'id', 'UNKNOWN')}\n"
			f"PROVIDER: {provider}\n"
			f"MODEL: {model}\n"
			f"FINISH REASON: {finish_reason}\n"
			f"USAGE: {usage}\n"
			f"RESPONSE LENGTH: {len(raw_content)}\n"
			f"JSON ERROR: {error}\n\n"
			"RAW ISSUE RESPONSE:\n"
			f"{raw_content}"
		) from error

	return parse_issue_result(
		issue_data=data.get("issue") or {},
		issue_catalog=issue_catalog,
		provider=provider,
		model=model,
		entry_id=entry.id,
	)

# Because Claude is fucking special with its prefix and suffix
def clean_json_response(raw_content: str) -> str:
	raw_content = raw_content.strip()

	if raw_content.startswith("```json"):
		raw_content = raw_content.removeprefix("```json")
	elif raw_content.startswith("```"):
		raw_content = raw_content.removeprefix("```")

	if raw_content.endswith("```"):
		raw_content = raw_content.removesuffix("```")

	return raw_content.strip()


def analyze_entry_with_llm(
	entry,
	theme_catalog: list[ThemeResult],
	issue_catalog: list[IssueResult],
	provider: str,
	model: str | None = None,
) -> EntryAnalysisResult | None:

	client, selected_model, response_format = connect_to_llm(
		provider=provider,
		model=model,
	)

	# ---------------------------------------------------------
	# FIRST PASS: SENTIMENT + THEMES
	# ---------------------------------------------------------

	sentiment_theme_prompt = build_sentiment_theme_prompt(
		entry=entry,
		theme_catalog=theme_catalog,
	)

	sentiment_theme_messages = [
		{
			"role": "system",
			"content": SENTIMENT_THEME_ANALYZER_SYSTEM_PROMPT,
		},
		{
			"role": "user",
			"content": sentiment_theme_prompt,
		},
	]

	wait_for_token_capacity(
		provider=provider,
		messages=sentiment_theme_messages,
		max_output_tokens=MAX_OUTPUT_TOKENS,
	)

	sentiment_theme_request = {
		"model": selected_model,
		"temperature": 0,
		"max_tokens": MAX_OUTPUT_TOKENS,
		"messages": sentiment_theme_messages,
	}

	if response_format is not None:
		sentiment_theme_request["response_format"] = response_format

	sentiment_theme_response = create_completion_with_retry(
		client=client,
		request=sentiment_theme_request,
		provider=provider,
		messages=sentiment_theme_messages,
	)

	# If all retries failed, abandon this entry.
	if sentiment_theme_response is None:
		print(
			f"LLM sentiment/theme analysis failed for entry {entry.id} "
			f"after {MAX_LLM_RETRIES} attempts. "
			f"Provider={provider}, model={selected_model}. "
			"Skipping entry."
		)
		return None

	sentiment_theme_raw_content = clean_json_response(
		sentiment_theme_response.choices[0].message.content or "{}"
	)

	sentiment_result, theme_result, metadata = parse_sentiment_theme_response(
		raw_content=sentiment_theme_raw_content,
		entry=entry,
		theme_catalog=theme_catalog,
		provider=provider,
		model=selected_model,
	)

	# ---------------------------------------------------------
	# SECOND PASS: ISSUES
	# ---------------------------------------------------------

	issue_prompt = build_issue_analyzer_prompt(
		entry=entry,
		issue_catalog=issue_catalog,
	)

	issue_messages = [
		{
			"role": "system",
			"content": ISSUE_ANALYZER_SYSTEM_PROMPT,
		},
		{
			"role": "user",
			"content": issue_prompt,
		},
	]

	wait_for_token_capacity(
		provider=provider,
		messages=issue_messages,
		max_output_tokens=MAX_OUTPUT_TOKENS,
	)

	issue_request = {
		"model": selected_model,
		"temperature": 0,
		"max_tokens": MAX_OUTPUT_TOKENS,
		"messages": issue_messages,
	}

	if response_format is not None:
		issue_request["response_format"] = response_format

	issue_response = create_completion_with_retry(
		client=client,
		request=issue_request,
		provider=provider,
		messages=issue_messages,
	)

	# If issue analysis fails, preserve the successful
	# sentiment/theme analysis and continue without issues.
	if issue_response is None:
		print(
			f"LLM issue analysis failed for entry {entry.id} "
			f"after {MAX_LLM_RETRIES} attempts. "
			f"Provider={provider}, model={selected_model}. "
			"Keeping sentiment/theme result without issues."
		)

		metadata["model"] = selected_model
		metadata["provider"] = provider
		metadata["issue_analysis_failed"] = True

		return EntryAnalysisResult(
			entry_id=entry.id,
			sentiment=sentiment_result,
			themes=[theme_result],
			issues=[],
			metadata=metadata,
		)

	issue_choice = issue_response.choices[0]

	issue_raw_content = clean_json_response(
		issue_choice.message.content or "{}"
	)

	issue_result = parse_issue_analyzer_response(
		raw_content=issue_raw_content,
		finish_reason=issue_choice.finish_reason,
		usage=getattr(issue_response, "usage", None),
		entry=entry,
		issue_catalog=issue_catalog,
		provider=provider,
		model=selected_model,
	)

	metadata["model"] = selected_model
	metadata["provider"] = provider

	return EntryAnalysisResult(
		entry_id=entry.id,
		sentiment=sentiment_result,
		themes=[theme_result],
		issues=[issue_result] if issue_result else [],
		metadata=metadata,
	)


def create_completion_with_retry(
	client,
	request: dict,
	provider: str,
	messages: list[dict],
):
	for attempt in range(1, MAX_LLM_RETRIES + 1):

		if attempt > 1:
			wait_for_token_capacity(
				provider=provider,
				messages=messages,
				max_output_tokens=MAX_OUTPUT_TOKENS,
			)

		response = client.chat.completions.create(**request)

		choice = response.choices[0]
		finish_reason = getattr(choice, "finish_reason", None)

		if finish_reason != "error":
			return response

		if attempt == MAX_LLM_RETRIES:
			raw_content = choice.message.content or ""

			raise RuntimeError(
				"LLM inference failed after retries.\n\n"
				f"PROVIDER: {provider}\n"
				f"MODEL: {request.get('model')}\n"
				f"ATTEMPTS: {MAX_LLM_RETRIES}\n"
				f"FINISH REASON: {finish_reason}\n"
				f"USAGE: {getattr(response, 'usage', None)}\n\n"
				"RAW RESPONSE:\n"
				f"{raw_content}"
			)

		delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))

		print(
			f"LLM inference returned finish_reason='error' "
			f"for provider={provider}, model={request.get('model')}. "
			f"Retrying attempt {attempt + 1}/{MAX_LLM_RETRIES}..."
		)

		time.sleep(delay)

