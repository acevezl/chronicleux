import json

from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import F, Q

from studies.models import CanonicalIssue, ThemeAndIssueStatus, ThemeAndIssueSource

from studies.services.contracts import IssueResult
from studies.services.llm.client import connect_to_llm


ISSUE_EXPLORER_SYSTEM_PROMPT = """
You are acting as a usability issue extractor for diary study entries in UX research.

Your task is to identify common UX/usability issues across all entries.
Issues should represent recurring friction, confusion, failures, errors, accessibility barriers, inefficiencies, dissatisfaction, or problems that affect the user experience.

You must:
1. Use the provided canonical issue catalog as the primary issue catalog.
2. Identify whether the diary entries contain meaningful recurring issues that are not covered by the canonical catalog.
3. Suggest additional issues that are clearly grounded in the diary entries only if there is no existing canonical issue that reasonably covers it.

Do not analyze sentiment.
Do not invent themes or recommendations.
Do not suggest a new issue if an existing canonical issue reasonably covers it.

Return only valid JSON.
""".strip()


MAX_ISSUE_ENTRY_CHARS = 1200


def compact_text(text: str, max_chars: int = MAX_ISSUE_ENTRY_CHARS) -> str:
    text = " ".join((text or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def get_canonical_issue_catalog() -> list[dict]:
    fields = [
        field.name
        for field in CanonicalIssue._meta.concrete_fields
    ]

    return list(
        CanonicalIssue.objects
        .filter(is_active=True)
        .filter(
            Q(status=ThemeAndIssueStatus.APPROVED)
            | Q(status=ThemeAndIssueStatus.SUGGESTED)
            | Q(status__isnull=True)
            | Q(status="")
        )
        .annotate(canonical_issue_id=F("id"))
        .order_by("name")
        .values("canonical_issue_id", *fields)
    )


def format_entries_for_issue_exploration(entries) -> str:
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
    )


def build_issue_explorer_prompt(
    entries,
    canonical_issue_catalog: list[dict],
    max_issues: int = 8,
) -> str:
    entries_json = format_entries_for_issue_exploration(entries)

    canonical_issue_catalog_json = json.dumps(
        canonical_issue_catalog,
        ensure_ascii=False,
        separators=(",", ":"),
        cls=DjangoJSONEncoder,
    )

    return f"""
Analyze the following diary study entries against the canonical issue catalog.

Return up to {max_issues} additional suggested issues that are NOT already covered by the canonical issue catalog.

Return the response using this exact JSON structure:

{{
  "suggested_issues": [
    {{
      "issue_id": 1,
      "weight": 0.0,
      "label": "short human-readable suggested issue label",
      "keywords": ["keyword-1", "keyword-2", "keyword-3"],
      "method": "llm",
      "metadata": {{
        "language": "english",
        "num_keywords": 3,
        "match_type": "suggested",
        "is_catalog_suggestion": true,
        "rationale": "why this issue is needed and not covered by the canonical catalog"
      }}
    }}
  ]
}}

Rules:
- Return only suggested issues that are clearly grounded in the entries.
- Do not include canonical issues in suggested_issues.
- Do not suggest a new issue if an existing canonical issue reasonably covers the meaning.
- Return at most {max_issues} suggested issues.
- Label should be concise and useful to a UX evaluator.
- Keywords should contain short phrases found in or strongly supported by the entries.
- Weight should be 0.0 for now. Entry-level weights will be assigned later.
- If no additional issues are needed, return "suggested_issues": [].
- Do not include entry-level analysis.
- Do not include sentiment.
- Do not include themes.
- Do not include recommendations.
- Return a JSON object only, do not return anything else.

Canonical issue catalog:
{canonical_issue_catalog_json}

Diary entries:
{entries_json}
""".strip()


def canonical_issue_catalog_to_issue_results(
    canonical_issue_catalog: list[dict],
    provider: str,
    model: str,
) -> list[IssueResult]:
    return [
        IssueResult(
            issue_id=issue["canonical_issue_id"],
            weight=0.0,
            label=issue.get("name", ""),
            keywords=issue.get("aliases") or [],
            method="llm",
            metadata={
                **issue,
                "model": model,
                "provider": provider,
                "match_type": "canonical",
                "is_catalog_suggestion": False,
                "canonical_issue_id": issue["canonical_issue_id"],
                "catalog_match_weight": 1.0,
            },
        )
        for issue in canonical_issue_catalog
    ]


def parse_issue_explorer_response(
    raw_content: str,
    provider: str,
    model: str,
) -> list[IssueResult]:
    data = json.loads(raw_content)
    suggested_issues = data.get("suggested_issues", [])

    issue_results = []

    for index, issue in enumerate(suggested_issues, start=1):
        keywords = issue.get("keywords", [])

        metadata = issue.get("metadata", {})
        metadata["model"] = model
        metadata["provider"] = provider
        metadata["num_keywords"] = len(keywords)
        metadata["match_type"] = "suggested"
        metadata["is_catalog_suggestion"] = True

        issue_results.append(
            IssueResult(
                issue_id=index,
                weight=issue.get("weight", 0.0),
                label=issue.get("label", ""),
                keywords=keywords,
                method="llm",
                metadata=metadata,
            )
        )

    return issue_results


def create_suggested_canonical_issues(
    suggested_issue_results: list[IssueResult],
) -> None:
    for issue_result in suggested_issue_results:
        name = (issue_result.label or "").strip()

        if not name:
            continue

        metadata = issue_result.metadata or {}

        CanonicalIssue.objects.get_or_create(
            name=name,
            defaults={
                "description": metadata.get("description")
                or metadata.get("rationale")
                or "",
                "aliases": issue_result.keywords or [],
                "examples": "",
                "source": ThemeAndIssueSource.LLM,
                "status": ThemeAndIssueStatus.SUGGESTED,
                "is_active": True,
            },
        )


def explore_issues_with_llm(
    entries,
    provider: str,
    model: str | None = None,
    max_issues: int = 8,
) -> list[IssueResult]:
    client, selected_model = connect_to_llm(provider=provider, model=model)

    canonical_issue_catalog = get_canonical_issue_catalog()

    prompt = build_issue_explorer_prompt(
        entries=entries,
        canonical_issue_catalog=canonical_issue_catalog,
        max_issues=max_issues,
    )

    response = client.chat.completions.create(
        model=selected_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ISSUE_EXPLORER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw_content = response.choices[0].message.content or "{}"

    suggested_issue_results = parse_issue_explorer_response(
        raw_content=raw_content,
        provider=provider,
        model=selected_model,
    )

    create_suggested_canonical_issues(
        suggested_issue_results=suggested_issue_results,
    )

    canonical_issue_catalog = get_canonical_issue_catalog()

    return canonical_issue_catalog_to_issue_results(
        canonical_issue_catalog=canonical_issue_catalog,
        provider=provider,
        model=selected_model,
    )
