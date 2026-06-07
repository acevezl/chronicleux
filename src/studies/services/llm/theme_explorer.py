import json

from studies.services.contracts import ThemeResult
from studies.services.llm.client import connect_to_llm


THEME_EXPLORER_SYSTEM_PROMPT = """
You are acting as a thematic analyzer of diary study entries for UX research.

Your task is to identify common themes across all entries.
Themes should represent recurring experiences, behaviors, frustrations, needs, expectations, or reactions (both positive or negative).

You can only identify themes that exist in the entries provided.
Do not add or invent themes that are not grounded in the entries provided.
Do not analyze sentiment. 
Use short labels.
Use keywords or short phrases that appear in the entries as much as possible.
Do not invent issues or recommendations.

Return only valid JSON.
""".strip()

MAX_THEME_ENTRY_CHARS = 1200

def compact_text(text: str, max_chars: int = MAX_THEME_ENTRY_CHARS) -> str:
    text = " ".join((text or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."

def format_entries_for_theme_exploration(entries) -> str:
    formatted_entries = []

    for entry in entries:
        formatted_entries.append(
            {
                "entry_id": entry.id,
                "text": compact_text(entry.content),
            }
        )

    return json.dumps(
        formatted_entries,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_theme_explorer_prompt(entries, max_themes: int = 8) -> str:
    entries_json = format_entries_for_theme_exploration(entries)

    return f"""
Analyze the following diary study entries and identify up to {max_themes} shared themes.

Return the response using this exact JSON structure:

{{
  "themes": [
    {{
      "theme_id": 1,
      "weight": 0.0,
      "label": "short human-readable theme label",
      "keywords": ["keyword-1", "keyword-2", "keyword-3"],
      "method": "llm",
      "metadata": {{
        "language": "english",
        "num_keywords": 3
      }}
    }}
  ]
}}

Rules:
- theme_id must start at 1 and increment by 1.
- label should be concise and useful to a UX evaluator.
- keywords should contain short phrases found in or strongly supported by the entries.
- weight should be 0.0 for now. Entry-level weights will be assigned later.
- Do not include entry-level analysis.
- Do not include sentiment.
- Return a JSON object only, do not return anything else.
- Prefer themes that are supported by more than one entry.

Diary entries:
{entries_json}
""".strip()


def parse_theme_explorer_response(raw_content: str, provider: str, model: str) -> list[ThemeResult]:
    data = json.loads(raw_content)
    themes = data.get("themes", [])

    theme_results = []

    for theme in themes:
        keywords = theme.get("keywords", [])

        metadata = theme.get("metadata", {})
        metadata["model"] = model
        metadata["provider"] = provider
        metadata["num_keywords"] = len(keywords)

        theme_results.append(
            ThemeResult(
                theme_id=theme.get("theme_id"),
                weight=theme.get("weight", 0.0),
                label=theme.get("label", ""),
                keywords=keywords,
                method="llm",
                metadata=metadata,
            )
        )

    return theme_results


def explore_themes_with_llm(
    entries,
    provider: str,
    model: str | None = None,
    max_themes: int = 8,
) -> list[ThemeResult]:
    client, selected_model = connect_to_llm(provider=provider, model=model)

    prompt = build_theme_explorer_prompt(
        entries=entries,
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

    return parse_theme_explorer_response(
        raw_content=raw_content,
        provider=provider,
        model=selected_model,
    )