import json

from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import F, Q

from studies.models import CanonicalTheme, ThemeAndIssueStatus, ThemeAndIssueSource

from studies.services.contracts import ThemeResult
from studies.services.llm.client import connect_to_llm


THEME_EXPLORER_SYSTEM_PROMPT = """
You are acting as a theme extractor for diary study entries in UX research.

Your task is to identify common themes across all entries.
Themes should represent recurring experiences, behaviors, frustrations, needs, expectations, or reactions (both positive or negative).

You must:
1. Use the provided canonical theme catalog as the primary theme catalog.
2. Identify whether the diary entries contain meaningful recurring themes that are not covered by the canonical catalog.
3. Suggest additional themes that are clearly grounded in the diary entries only if there is no existing canonical theme that reasonably covers it.

Do not analyze sentiment.
Do not invent issues or recommendations.
Do not suggest a new theme if an existing canonical theme reasonably covers it.

Return only valid JSON.
""".strip()


MAX_THEME_ENTRY_CHARS = 1200


def compact_text(text: str, max_chars: int = MAX_THEME_ENTRY_CHARS) -> str:
    text = " ".join((text or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def get_canonical_theme_catalog() -> list[dict]:
    return list(
        CanonicalTheme.objects
        .filter(is_active=True)
        .filter(
            Q(status=ThemeAndIssueStatus.APPROVED)
            | Q(status=ThemeAndIssueStatus.SUGGESTED)
            | Q(status__isnull=True)
            | Q(status="")
        )
        .annotate(canonical_theme_id=F("id"))
        .order_by("name")
        .values(
            "canonical_theme_id",
            "name",
            "description",
            "aliases",
            "examples",
            "source",
            "status",
            "is_active",
            "created_by_id",
        )
    )


def format_entries_for_theme_exploration(entries) -> str:
    return json.dumps(
        [
            {
                "entry_id": entry.id,
                "text": compact_text(entry.content),
            }
            for entry in entries
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        cls=DjangoJSONEncoder,
    )


def build_theme_explorer_prompt(
    entries,
    canonical_theme_catalog: list[dict],
    max_themes: int = 8,
) -> str:
    entries_json = format_entries_for_theme_exploration(entries)

    canonical_theme_catalog_json = json.dumps(
        canonical_theme_catalog,
        ensure_ascii=False,
        separators=(",", ":"),
        cls=DjangoJSONEncoder,
    )

    return f"""
Analyze the following diary study entries against the canonical theme catalog.

Return up to {max_themes} additional suggested themes that are NOT already covered by the canonical theme catalog.

Return the response using this exact JSON structure:

{{
  "suggested_themes": [
    {{
      "name": "short human-readable theme name",
      "description": "brief explanation of what this theme captures",
      "aliases": ["keyword-1", "keyword-2", "keyword-3"],
      "examples": "short example phrase or sentence grounded in the diary entries"
    }}
  ]
}}


Rules:
- Return only suggested themes that are clearly grounded in the entries.
- Do not include canonical themes in suggested_themes.
- Do not suggest a new theme if an existing canonical theme reasonably covers the meaning.
- Do not include theme_id, id, canonical_theme_id, source, status, is_active, created_by, created_at, or updated_at.
- Return at most {max_themes} suggested themes.
- name should be concise and useful to a UX evaluator.
- description should explain what the theme captures.
- aliases should contain short searchable phrases found in or strongly supported by the entries.
- examples should contain one short example phrase or sentence grounded in the diary entries.
- If no additional themes are needed, return "suggested_themes": [].
- Return a JSON object only.

Canonical theme catalog:
{canonical_theme_catalog_json}

Diary entries:
{entries_json}
""".strip()


def canonical_theme_catalog_to_theme_results(
    canonical_theme_catalog: list[dict],
    provider: str,
    model: str,
) -> list[ThemeResult]:
    return [
        ThemeResult(
            theme_id=theme["canonical_theme_id"],
            weight=0.0,
            label=theme.get("name", ""),
            keywords=theme.get("aliases") or [],
            method="llm",
            metadata={
                **theme,
                "model": model,
                "provider": provider,
                "match_type": "canonical",
                "is_catalog_suggestion": False,
                "canonical_theme_id": theme["canonical_theme_id"],
                "catalog_match_weight": 1.0,
            },
        )
        for theme in canonical_theme_catalog
    ]


def parse_theme_explorer_response(raw_content: str) -> list[dict]:
    data = json.loads(raw_content)
    suggested_themes = data.get("suggested_themes", [])

    parsed_themes = []

    for theme in suggested_themes:
        name = (theme.get("name") or "").strip()

        if not name:
            continue

        aliases = theme.get("aliases") or []

        if isinstance(aliases, str):
            aliases = [aliases]

        parsed_themes.append(
            {
                "name": name,
                "description": (theme.get("description") or "").strip(),
                "aliases": aliases,
                "examples": (theme.get("examples") or "").strip(),
                "source": ThemeAndIssueSource.LLM,
                "status": ThemeAndIssueStatus.SUGGESTED,
                "is_active": True,
            }
        )

    return parsed_themes


def create_suggested_canonical_themes(
    suggested_themes: list[dict],
) -> list[CanonicalTheme]:
    canonical_themes = []

    for suggested_theme in suggested_themes:
        name = suggested_theme["name"]

        canonical_theme, _ = CanonicalTheme.objects.get_or_create(
            name=name,
            defaults={
                "description": suggested_theme.get("description", ""),
                "aliases": suggested_theme.get("aliases", []),
                "examples": suggested_theme.get("examples", ""),
                "source": suggested_theme.get("source", ThemeAndIssueSource.LLM),
                "status": suggested_theme.get("status", ThemeAndIssueStatus.SUGGESTED),
                "is_active": suggested_theme.get("is_active", True),
            },
        )

        canonical_themes.append(canonical_theme)

    return canonical_themes


def explore_themes_with_llm(
    entries,
    provider: str,
    model: str | None = None,
    max_themes: int = 8,
) -> list[ThemeResult]:
    client, selected_model = connect_to_llm(provider=provider, model=model)

    canonical_theme_catalog = get_canonical_theme_catalog()

    prompt = build_theme_explorer_prompt(
        entries=entries,
        canonical_theme_catalog=canonical_theme_catalog,
        max_themes=max_themes,
    )

    response = client.chat.completions.create(
        model=selected_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": THEME_EXPLORER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw_content = response.choices[0].message.content or "{}"

    suggested_themes = parse_theme_explorer_response(
        raw_content=raw_content,
    )

    create_suggested_canonical_themes(
        suggested_themes=suggested_themes,
    )

    canonical_theme_catalog = get_canonical_theme_catalog()

    return canonical_theme_catalog_to_theme_results(
        canonical_theme_catalog=canonical_theme_catalog,
        provider=provider,
        model=selected_model,
    )