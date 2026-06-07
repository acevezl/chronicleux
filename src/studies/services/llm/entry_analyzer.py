import json

from studies.services.contracts import EntryAnalysisResult, SentimentResult, ThemeResult
from studies.services.llm.client import connect_to_llm
from studies.services.nlp.sentiment._thresholds import map_sentiment_score_to_label


ENTRY_ANALYZER_SYSTEM_PROMPT = """
You are acting as an entry-level analyzer of diary study entries for UX research.

Your task is to analyze ONE diary entry at a time.

You must:
1. Analyze the sentiment of the entry, and score its polarity between -1 (most negative) and 1 (most positive).
2. Analyze the theme of the entry, and assign the best matching theme from the provided theme catalog.

You must not create new themes.
You must not invent issues or recommendations.
You must only use one of the provided theme_id values.
You must return only valid JSON.
""".strip()


def format_theme_catalog(theme_catalog: list[ThemeResult]) -> str:
    formatted_themes = []

    for theme in theme_catalog:
        formatted_themes.append(
            {
                "theme_id": theme.theme_id,
                "label": theme.label,
                "keywords": theme.keywords,
                "metadata": theme.metadata,
            }
        )

    return json.dumps(formatted_themes, ensure_ascii=False, separators=(",", ":"))


def build_entry_analyzer_prompt(entry, theme_catalog: list[ThemeResult]) -> str:
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
    "theme_id": 1,
    "weight": 0.75,
    "label": "theme label from provided catalog",
    "keywords": ["keyword-1", "keyword-2", "keyword-3"],
    "method": "llm",
    "metadata": {{
      "language": "english",
      "num_keywords": 3
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
- do not return a sentiment label, I will assign it based on the score.

Theme rules:
- You must choose exactly one theme from the provided theme catalog.
- Do not create a new theme.
- theme_id must match one of the provided theme_id values.
- label must match the selected theme label.
- keywords must come from the selected theme.
- weight should represent how strongly this entry matches the selected theme, from 0.0 to 1.0.

Return a JSON object only, do not return anything else.

Theme catalog:
{theme_catalog_json}

Diary entry:
{{
  "entry_id": {entry.id},
  "text": {json.dumps(entry.content, ensure_ascii=False)}
}}
""".strip()


def get_theme_by_id(theme_catalog: list[ThemeResult], theme_id) -> ThemeResult | None:
    try:
        theme_id = int(theme_id)
    except (TypeError, ValueError):
        return None

    for theme in theme_catalog:
        if theme.theme_id == theme_id:
            return theme

    return None


def parse_entry_analyzer_response(
    raw_content: str,
    entry,
    theme_catalog: list[ThemeResult],
    provider: str,
    model: str,
) -> EntryAnalysisResult:
    data = json.loads(raw_content)

    # Get the data
    sentiment_data = data.get("sentiment", {})
    theme_data = data.get("theme", {})
    metadata = data.get("metadata", {})

    # Sentiment data
    sentiment_score = float(sentiment_data.get("score", 0.0))

    # In case the LLM hallucinates the score outside the range
    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    # I map the labels based on the score, I do not use the LLM's recommendation 
    # b/c they can give me something like Score: 0.2 and Label: Negative
    # Just reducing the points of failure
    sentiment_label = map_sentiment_score_to_label(sentiment_score) 

    sentiment_metadata = sentiment_data.get("metadata", {})
    sentiment_metadata["model"] = model
    sentiment_metadata["provider"] = provider

    # Theme data
    try:
        selected_theme_id = int(theme_data.get("theme_id"))
    except (TypeError, ValueError):
        selected_theme_id = None
    selected_theme = get_theme_by_id(theme_catalog, selected_theme_id)

    if selected_theme:
        theme_label = selected_theme.label
        theme_keywords = selected_theme.keywords
        theme_metadata = dict(selected_theme.metadata or {})
    else:
        theme_label = theme_data.get("label", "")
        theme_keywords = theme_data.get("keywords", [])
        theme_metadata = theme_data.get("metadata", {})

    theme_metadata.update(theme_data.get("metadata", {}))
    theme_metadata["model"] = model
    theme_metadata["provider"] = provider
    theme_metadata["num_keywords"] = len(theme_keywords)

    metadata["model"] = model
    metadata["provider"] = provider

    return EntryAnalysisResult(
        entry_id=entry.id,
        sentiment=SentimentResult(
            score=sentiment_score,
            label=sentiment_label,
            method="llm",
            metadata=sentiment_metadata,
        ),
        theme=ThemeResult(
            theme_id=selected_theme_id,
            weight=theme_data.get("weight", 0.0),
            label=theme_label,
            keywords=theme_keywords,
            method="llm",
            metadata=theme_metadata,
        ),
        metadata=metadata,
    )


def analyze_entry_with_llm(
    entry,
    theme_catalog: list[ThemeResult],
    provider: str,
    model: str | None = None,
) -> EntryAnalysisResult:
    client, selected_model = connect_to_llm(provider=provider, model=model)

    prompt = build_entry_analyzer_prompt(
        entry=entry,
        theme_catalog=theme_catalog,
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
        provider=provider,
        model=selected_model,
    )