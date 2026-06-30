import json
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


ENTRY_ANALYZER_SYSTEM_PROMPT = """
You are acting as an entry-level analyzer of diary study entries for UX research.

Your task is to analyze ONE diary entry at a time.

You must:
1. Analyze the sentiment of the entry, and score its polarity between -1 and 1.
2. Assign the best matching theme from the provided theme catalog.
3. Detect whether the entry describes a UX/usability issue.
4. If an issue is present, assign the best matching issue from the provided issue catalog.

Rules:
- You must only use theme_id values from the provided theme catalog.
- You must only use issue_id values from the provided issue catalog.
- The theme catalog may include canonical themes and suggested themes. Both are valid choices.
- The issue catalog may include canonical issues and suggested issues. Both are valid choices.
- Do not create new themes at entry-analysis time.
- Do not create new issues at entry-analysis time.
- Do not invent recommendations.
- If no UX/usability issue is clearly present, return null for issue.
- Return only valid JSON.
""".strip()


def format_theme_catalog(theme_catalog: list[ThemeResult]) -> str:
    return json.dumps(
        [
            {
                "theme_id": theme.theme_id,
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
                "issue_id": issue.issue_id,
                "label": issue.label,
                "keywords": issue.keywords,
            }
            for issue in issue_catalog
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        cls=DjangoJSONEncoder,
    )


def build_entry_analyzer_prompt(
    entry,
    theme_catalog: list[ThemeResult],
    issue_catalog: list[IssueResult],
) -> str:
    theme_catalog_json = format_theme_catalog(theme_catalog)
    issue_catalog_json = format_issue_catalog(issue_catalog)

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
    "theme_id": 1,
    "weight": 0.75,
    "metadata": {{
      "language": "english",
      "rationale": "short explanation of why this theme matches the entry"
    }}
  }},
  "issue": {{
    "issue_id": 1,
    "weight": 0.80,
    "metadata": {{
      "language": "english",
      "rationale": "short explanation of why this issue matches the entry"
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
- Return theme.theme_id using only the theme_id field from the Theme catalog.
- Do not use issue_id.
- Do not use id.
- Do not use canonical_theme_id.
- Do not use canonical_issue_id.
- If you cannot choose confidently, choose the closest theme_id from the Theme catalog. Never invent a theme_id.

Issue rules:
- Return an issue only if the entry clearly describes UX friction, confusion, failure, accessibility problems, errors, inefficiency, dissatisfaction, or another usability problem.
- If no UX/usability issue is clearly present, return "issue": {{}}
- If an UX/usability issue is clearly present, return issue.issue_id using only the issue_id field from the Issue catalog.
- Do not use theme_id.
- Do not use id.
- Do not use canonical_theme_id.
- Do not use canonical_issue_id.
- Never invent an issue_id.

Metadata rules:
- Keep metadata concise.
- Do not include recommendations.
- Do not include extra top-level keys.

Return a JSON object only, do not return anything else.

Theme catalog:
{theme_catalog_json}

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
        valid_theme_ids = [
            theme.theme_id
            for theme in theme_catalog
        ]

        raise RuntimeError(
            f"LLM returned invalid theme_id '{theme_data.get('theme_id')}' "
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
    issue_data: dict | None,
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
        valid_issue_ids = [
            issue.issue_id
            for issue in issue_catalog
        ]

        raise RuntimeError(
            f"LLM returned invalid issue_id '{issue_data.get('issue_id')}' "
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


def parse_entry_analyzer_response(
    raw_content: str,
    entry,
    theme_catalog: list[ThemeResult],
    issue_catalog: list[IssueResult],
    provider: str,
    model: str,
) -> EntryAnalysisResult:
    data = json.loads(raw_content)

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

    issue_result = parse_issue_result(
        issue_data=data.get("issue"),
        issue_catalog=issue_catalog,
        provider=provider,
        model=model,
        entry_id=entry.id,
    )

    metadata = dict(data.get("metadata") or {})
    metadata["model"] = model
    metadata["provider"] = provider

    return EntryAnalysisResult(
        entry_id=entry.id,
        sentiment=sentiment_result,
        themes=[theme_result] if theme_result else [],
        issues=[issue_result] if issue_result else [],
        metadata=metadata,
    )


def analyze_entry_with_llm(
    entry,
    theme_catalog: list[ThemeResult],
    issue_catalog: list[IssueResult],
    provider: str,
    model: str | None = None,
) -> EntryAnalysisResult:
    client, selected_model = connect_to_llm(provider=provider, model=model)

    prompt = build_entry_analyzer_prompt(
        entry=entry,
        theme_catalog=theme_catalog,
        issue_catalog=issue_catalog,
    )

    response = client.chat.completions.create(
        model=selected_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ENTRY_ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw_content = response.choices[0].message.content or "{}"

    return parse_entry_analyzer_response(
        raw_content=raw_content,
        entry=entry,
        theme_catalog=theme_catalog,
        issue_catalog=issue_catalog,
        provider=provider,
        model=selected_model,
    )