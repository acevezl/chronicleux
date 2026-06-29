from __future__ import annotations

from dataclasses import dataclass
import re

from bertopic import BERTopic
from django.db.models import Q
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from studies.models import CanonicalTheme, ThemeAndIssueSource, ThemeAndIssueStatus
from studies.services.contracts import BaseThemeExtractor, ThemeResult

MODEL_NAME = "BERTopic + Canonical Theme Matching"
THEME_WEIGHT_THRESHOLD = 0.30

# Debug imports
from collections import Counter
# Debug imports

@dataclass(frozen=True)
class CanonicalThemeDocument:
    canonical_theme_id: int
    label: str
    text: str
    keywords: list[str]


@dataclass(frozen=True)
class BertopicThemeCandidate:
    bertopic_theme_id: int
    label: str
    weight: float | None
    keywords: list[str]
    match_text: str


class BertopicThemeExtractor(BaseThemeExtractor):
    method_name = "bertopic"

    def __init__(
        self,
        num_keywords: int = 8,
        max_features: int = 1000,
        theme_weight_threshold: float = THEME_WEIGHT_THRESHOLD,
    ):
        self.num_keywords = num_keywords
        self.max_features = max_features
        self.theme_weight_threshold = theme_weight_threshold

    def extract(self, documents: list[str]) -> tuple[list[ThemeResult], list[dict]]:
        """
        Extract themes with BERTopic, then resolve those BERTopic themes against
        the canonical theme catalogue.

        Rule:
        1. BERTopic proposes themes from the diary entries.
        2. Each proposed BERTopic theme is compared against the canonical catalogue.
        3. If the catalogue match weight is greater than THEME_WEIGHT_THRESHOLD,
           the catalogue theme is used.
        4. Otherwise, the BERTopic theme remains a suggested theme.
        """
        valid_items = [
            (index, document.strip())
            for index, document in enumerate(documents)
            if document and document.strip()
        ]

        if not valid_items:
            return [], []

        original_indices = [item[0] for item in valid_items]
        valid_documents = [item[1] for item in valid_items]

        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            min_df=1,
            max_df=1.0,
        )

        topic_model = BERTopic(
            vectorizer_model=vectorizer_model,
            language="english",
            calculate_probabilities=True,
            min_topic_size=3,
            verbose=False,
        )

        topics, probabilities = topic_model.fit_transform(valid_documents)

        # Debug
        print("BERTopic topic counts:", Counter(topics))
        # Debug

        bertopic_candidates = self._build_bertopic_theme_candidates(
            topic_model=topic_model,
            topics=topics,
        )

        if not bertopic_candidates:
            return [], []

        bertopic_theme_id_to_examples = self._build_examples_by_bertopic_theme(
            topics=topics,
            valid_documents=valid_documents,
            bertopic_candidates=bertopic_candidates,
        )

        canonical_documents = self._get_canonical_theme_documents()

        themes, bertopic_theme_id_to_resolved_theme_id = self._resolve_bertopic_themes(
            bertopic_candidates=bertopic_candidates,
            canonical_documents=canonical_documents,
            bertopic_theme_id_to_examples=bertopic_theme_id_to_examples,
        )

        assignments = self._build_assignments(
            topics=topics,
            probabilities=probabilities,
            original_indices=original_indices,
            bertopic_candidates=bertopic_candidates,
            bertopic_theme_id_to_resolved_theme_id=bertopic_theme_id_to_resolved_theme_id,
        )

        return themes, assignments

    def _build_bertopic_theme_candidates(
        self,
        topic_model: BERTopic,
        topics: list[int],
    ) -> list[BertopicThemeCandidate]:
        candidates: list[BertopicThemeCandidate] = []

        topic_ids = sorted({
            topic_id
            for topic_id in topics
            if topic_id != -1
        })

        for bertopic_theme_id in topic_ids:
            topic_terms = topic_model.get_topic(bertopic_theme_id) or []

            keywords = [
                term
                for term, _weight in topic_terms[:self.num_keywords]
                if term
            ]

            if not keywords:
                continue

            term_weights = [
                float(weight)
                for _term, weight in topic_terms[:self.num_keywords]
                if weight is not None
            ]

            weight = (
                float(sum(term_weights) / len(term_weights))
                if term_weights
                else None
            )

            label = self._make_suggestion_label(keywords)

            candidates.append(
                BertopicThemeCandidate(
                    bertopic_theme_id=bertopic_theme_id,
                    label=label,
                    weight=weight,
                    keywords=keywords,
                    match_text=" ".join([label, *keywords]).strip(),
                )
            )

        return candidates

    def _build_examples_by_bertopic_theme(
        self,
        topics: list[int],
        valid_documents: list[str],
        bertopic_candidates: list[BertopicThemeCandidate],
    ) -> dict[int, list[str]]:
        candidate_theme_ids = {
            candidate.bertopic_theme_id
            for candidate in bertopic_candidates
        }

        examples_by_theme_id = {
            candidate.bertopic_theme_id: []
            for candidate in bertopic_candidates
        }

        for document_row_index, bertopic_theme_id in enumerate(topics):
            if bertopic_theme_id == -1:
                continue

            if bertopic_theme_id not in candidate_theme_ids:
                continue

            document = valid_documents[document_row_index]

            if document:
                examples_by_theme_id.setdefault(bertopic_theme_id, []).append(document)

        return examples_by_theme_id

    def _resolve_bertopic_themes(
        self,
        bertopic_candidates: list[BertopicThemeCandidate],
        canonical_documents: list[CanonicalThemeDocument],
        bertopic_theme_id_to_examples: dict[int, list[str]],
    ) -> tuple[list[ThemeResult], dict[int, int]]:
        themes: list[ThemeResult] = []
        bertopic_theme_id_to_resolved_theme_id: dict[int, int] = {}
        resolved_theme_key_to_theme_id: dict[tuple[str, str], int] = {}

        catalog_matches = self._match_bertopic_candidates_to_catalog(
            bertopic_candidates=bertopic_candidates,
            canonical_documents=canonical_documents,
        )

        for candidate in bertopic_candidates:
            catalog_match = catalog_matches.get(candidate.bertopic_theme_id)

            if catalog_match and catalog_match["theme_weight"] > self.theme_weight_threshold:
                canonical_theme = catalog_match["canonical_theme"]
                resolved_key = ("canonical", str(canonical_theme.canonical_theme_id))

                if resolved_key not in resolved_theme_key_to_theme_id:
                    resolved_theme_id = len(themes)
                    resolved_theme_key_to_theme_id[resolved_key] = resolved_theme_id

                    themes.append(
                        ThemeResult(
                            theme_id=resolved_theme_id,
                            label=canonical_theme.label,
                            weight=catalog_match["theme_weight"],
                            keywords=canonical_theme.keywords,
                            method=self.method_name,
                            metadata={
                                "model": MODEL_NAME,
                                "match_type": "canonical",
                                "canonical_theme_id": canonical_theme.canonical_theme_id,
                                "catalog_match_weight": catalog_match["theme_weight"],
                                "theme_weight_threshold": self.theme_weight_threshold,
                                "original_bertopic_theme": {
                                    "theme_id": candidate.bertopic_theme_id,
                                    "label": candidate.label,
                                    "weight": candidate.weight,
                                    "keywords": candidate.keywords,
                                },
                                "language": "english",
                                "topic_model": "bertopic",
                                "vectorizer": "count",
                                "stop_words": "english",
                                "ngram_range": [2, 3],
                            },
                        )
                    )

                bertopic_theme_id_to_resolved_theme_id[candidate.bertopic_theme_id] = resolved_theme_key_to_theme_id[resolved_key]
                continue

            examples = bertopic_theme_id_to_examples.get(candidate.bertopic_theme_id, [])

            suggested_theme = self._get_or_create_suggested_canonical_theme(
                candidate=candidate,
                examples=examples,
            )

            resolved_key = ("suggested", str(suggested_theme.id))

            if resolved_key not in resolved_theme_key_to_theme_id:
                resolved_theme_id = len(themes)
                resolved_theme_key_to_theme_id[resolved_key] = resolved_theme_id

                themes.append(
                    ThemeResult(
                        theme_id=resolved_theme_id,
                        label=suggested_theme.name,
                        weight=candidate.weight,
                        keywords=candidate.keywords,
                        method=self.method_name,
                        metadata={
                            "model": MODEL_NAME,
                            "match_type": "suggested",
                            "is_catalog_suggestion": True,
                            "canonical_theme_id": suggested_theme.id,
                            "catalog_match_weight": (
                                catalog_match["theme_weight"]
                                if catalog_match
                                else None
                            ),
                            "theme_weight_threshold": self.theme_weight_threshold,
                            "suggested_theme": {
                                "id": suggested_theme.id,
                                "name": suggested_theme.name,
                                "description": suggested_theme.description,
                                "aliases": suggested_theme.aliases or [],
                                "source": suggested_theme.source,
                                "status": suggested_theme.status,
                            },
                            "original_bertopic_theme": {
                                "theme_id": candidate.bertopic_theme_id,
                                "label": candidate.label,
                                "weight": candidate.weight,
                                "keywords": candidate.keywords,
                            },
                            "language": "english",
                            "topic_model": "bertopic",
                            "vectorizer": "count",
                            "stop_words": "english",
                            "num_keywords": len(candidate.keywords),
                            "ngram_range": [2, 3],
                        },
                    )
                )

            bertopic_theme_id_to_resolved_theme_id[candidate.bertopic_theme_id] = resolved_theme_key_to_theme_id[resolved_key]

        return themes, bertopic_theme_id_to_resolved_theme_id

    def _get_or_create_suggested_canonical_theme(
        self,
        candidate: BertopicThemeCandidate,
        examples: list[str],
    ) -> CanonicalTheme:
        existing_theme = (
            CanonicalTheme.objects
            .filter(name__iexact=candidate.label)
            .first()
        )

        if existing_theme:
            return existing_theme

        return CanonicalTheme.objects.create(
            name=candidate.label,
            description=(
                "NLP-suggested theme produced by BERTopic because "
                "the generated theme did not strongly match the canonical theme catalog."
            ),
            aliases=candidate.keywords,
            examples="\n".join(examples[:5]),
            source=ThemeAndIssueSource.NLP,
            status=ThemeAndIssueStatus.SUGGESTED,
            is_active=True,
        )

    def _match_bertopic_candidates_to_catalog(
        self,
        bertopic_candidates: list[BertopicThemeCandidate],
        canonical_documents: list[CanonicalThemeDocument],
    ) -> dict[int, dict]:
        if not bertopic_candidates or not canonical_documents:
            return {}

        candidate_texts = [candidate.match_text for candidate in bertopic_candidates]
        catalog_texts = [theme.text for theme in canonical_documents]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 3),
        )

        matrix = vectorizer.fit_transform([*candidate_texts, *catalog_texts])

        candidate_matrix = matrix[:len(candidate_texts)]
        catalog_matrix = matrix[len(candidate_texts):]

        similarity_matrix = cosine_similarity(candidate_matrix, catalog_matrix)

        matches: dict[int, dict] = {}

        for candidate_index, similarities in enumerate(similarity_matrix):
            best_catalog_index = int(similarities.argmax())
            best_score = float(similarities[best_catalog_index])
            candidate = bertopic_candidates[candidate_index]

            top_match_indices = similarities.argsort()[-5:][::-1]

            print("\nBERTOPIC CATALOG MATCH DEBUG")
            print("--------------------------------")
            print(f"BERTopic candidate id: {candidate.bertopic_theme_id}")
            print(f"BERTopic candidate label: {candidate.label}")
            print(
                "BERTopic candidate weight: "
                f"{candidate.weight:.4f}" if candidate.weight is not None else "BERTopic candidate weight: None"
            )
            print(f"BERTopic candidate keywords: {candidate.keywords}")
            print(f"BERTopic candidate match text: {candidate.match_text}")
            print(f"Theme threshold: {self.theme_weight_threshold:.4f}")
            print("Top catalogue matches:")

            for rank, catalog_index in enumerate(top_match_indices, start=1):
                catalog_theme = canonical_documents[int(catalog_index)]
                score = float(similarities[int(catalog_index)])

                print(
                    f"  {rank}. "
                    f"id={catalog_theme.canonical_theme_id} | "
                    f"label={catalog_theme.label!r} | "
                    f"score={score:.4f} | "
                    f"keywords={catalog_theme.keywords}"
                )

            print("--------------------------------\n")

            matches[candidate.bertopic_theme_id] = {
                "theme_weight": best_score,
                "canonical_theme": canonical_documents[best_catalog_index],
            }

        return matches

    def _build_assignments(
        self,
        topics: list[int],
        probabilities,
        original_indices: list[int],
        bertopic_candidates: list[BertopicThemeCandidate],
        bertopic_theme_id_to_resolved_theme_id: dict[int, int],
    ) -> list[dict]:
        assignments: list[dict] = []

        bertopic_theme_ids = sorted([
            candidate.bertopic_theme_id
            for candidate in bertopic_candidates
        ])

        candidate_by_bertopic_theme_id = {
            candidate.bertopic_theme_id: candidate
            for candidate in bertopic_candidates
        }

        for valid_document_index, bertopic_theme_id in enumerate(topics):
            if bertopic_theme_id == -1:
                continue

            resolved_theme_id = bertopic_theme_id_to_resolved_theme_id.get(bertopic_theme_id)

            if resolved_theme_id is None:
                continue

            candidate = candidate_by_bertopic_theme_id.get(bertopic_theme_id)

            if candidate is None:
                continue

            topic_weight = None

            if probabilities is not None:
                try:
                    topic_position = bertopic_theme_ids.index(bertopic_theme_id)
                    topic_weight = float(
                        probabilities[valid_document_index][topic_position]
                    )
                except Exception:
                    topic_weight = None

            assignments.append(
                {
                    "document_index": original_indices[valid_document_index],
                    "theme_id": resolved_theme_id,
                    "theme_weight": topic_weight,
                    "source_bertopic_theme_id": candidate.bertopic_theme_id,
                    "source_bertopic_theme_label": candidate.label,
                    "source_bertopic_theme_weight": candidate.weight,
                }
            )

        return assignments

    def _get_canonical_theme_documents(self) -> list[CanonicalThemeDocument]:
        canonical_themes = (
            CanonicalTheme.objects
            .filter(is_active=True)
            .filter(
                Q(status=ThemeAndIssueStatus.APPROVED)
                | Q(status=ThemeAndIssueStatus.SUGGESTED)
                | Q(status__isnull=True)
                | Q(status="")
            )
            .order_by("name")
        )

        documents: list[CanonicalThemeDocument] = []

        for theme in canonical_themes:
            aliases = self._clean_list(theme.aliases)
            examples = self._split_examples(theme.examples)
            keywords = [theme.name, *aliases]

            text_parts = [
                theme.name,
                theme.description,
                " ".join(aliases),
                " ".join(examples),
            ]

            documents.append(
                CanonicalThemeDocument(
                    canonical_theme_id=theme.id,
                    label=theme.name,
                    text=" ".join(part for part in text_parts if part).strip(),
                    keywords=keywords,
                )
            )

        return documents

    def _make_suggestion_label(self, keywords: list[str]) -> str:
        weak_words = {
            "day",
            "today",
            "wanted",
            "want",
            "quick",
            "task",
            "thing",
            "use",
            "used",
            "using",
            "app",
        }

        cleaned_keywords = []

        for keyword in keywords:
            words = [
                word
                for word in re.sub(r"[^a-zA-Z0-9\s]", " ", keyword.lower()).split()
                if word not in weak_words
            ]

            if len(words) >= 2:
                cleaned_keywords.append(" ".join(words))

        if cleaned_keywords:
            return cleaned_keywords[0].title()

        for keyword in keywords:
            label = re.sub(r"\s+", " ", keyword).strip()

            if label:
                return label.title()

        return "Suggested Theme"

    def _clean_list(self, value) -> list[str]:
        if not value:
            return []

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]

        return []

    def _split_examples(self, value: str) -> list[str]:
        if not value:
            return []

        return [
            example.strip()
            for example in re.split(r"[\n;]+", value)
            if example.strip()
        ]
