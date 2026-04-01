import re

from django import template
from django.utils.html import conditional_escape, format_html
from django.utils.safestring import mark_safe

register = template.Library()


@register.filter
def highlight(text, query):
    if not text:
        return ""

    if not query:
        return conditional_escape(text)

    escaped_text = conditional_escape(text)
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    highlighted = pattern.sub(
        lambda m: f'<mark class="bg-yellow-200 text-gray-900 px-0.5 rounded">{m.group(0)}</mark>',
        str(escaped_text),
    )
    return mark_safe(highlighted)


@register.simple_tag
def content_snippet(text, query, radius=80):
    if not text:
        return ""

    if not query:
        snippet = text[: radius * 2]
        if len(text) > radius * 2:
            snippet += "..."
        return snippet

    match = re.search(re.escape(query), text, re.IGNORECASE)

    if not match:
        snippet = text[: radius * 2]
        if len(text) > radius * 2:
            snippet += "..."
        return snippet

    start = max(match.start() - radius, 0)
    end = min(match.end() + radius, len(text))

    snippet = text[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    escaped_snippet = conditional_escape(snippet)

    highlighted = pattern.sub(
        lambda m: f'<mark class="bg-yellow-200 text-gray-900 px-0.5 rounded">{m.group(0)}</mark>',
        str(escaped_snippet),
    )

    return mark_safe(highlighted)