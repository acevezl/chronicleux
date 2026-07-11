from __future__ import annotations

from dataclasses import dataclass
import re

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from studies.models import CanonicalTheme, ThemeAndIssueSource, ThemeAndIssueStatus
from studies.services.contracts import BaseThemeExtractor, ThemeResult


MODEL_NAME = "Count-LDA + Evidence-Vote Canonical Theme Matching"
THEME_WEIGHT_THRESHOLD = 0.15
NUM_THEMES = 12
NUM_KEYWORDS = 10
MAX_FEATURES = 1000
MAX_MATCHING_EXAMPLES = 5

SUGGESTION_CONTEXT_TERMS = {
    "netflix",
    "phone",
    "smartphone",
    "laptop",
    "tablet",
    "tv",
    "smart",
    "screen",
    "evening",
    "tonight",
    "today",
    "yesterday",
    "session",
    "started",
    "start",
    "did",
}


@dataclass(frozen=True)
class CanonicalThemeDocument:
    canonical_theme_id: int
    label: str
    text: str
    keywords: list[str]


@dataclass(frozen=True)
class LdaThemeCandidate:
    lda_theme_id: int
    label: str | None
    weight: float
    keywords: list[str]
    match_text: str


class CountLdaThemeExtractor(BaseThemeExtractor):
    method_name = "tfidf_lda"

    def __init__(
        self,
        num_themes: int = NUM_THEMES,
        num_keywords: int = NUM_KEYWORDS,
        max_features: int = MAX_FEATURES,
        theme_weight_threshold: float = THEME_WEIGHT_THRESHOLD,
    ):
        self.num_themes = num_themes
        self.num_keywords = num_keywords
        self.max_features = max_features
        self.theme_weight_threshold = theme_weight_threshold

    def extract(self, documents: list[str]) -> tuple[list[ThemeResult], list[dict]]:
        """
        Extract latent themes with count-based LDA and resolve them against
        the approved canonical theme catalogue.

        Each topic is matched first against approved canonical themes using
        topic keywords and representative-entry votes. A weak catalogue match
        is rejected, and only then is the topic added as an NLP suggestion.
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

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=self.max_features,
            min_df=2,
            max_df=0.75,
            ngram_range=(1, 3),
        )

        try:
            matrix = vectorizer.fit_transform(valid_documents)
        except ValueError:
            return [], []

        actual_num_themes = min(
            self.num_themes,
            matrix.shape[0],
            matrix.shape[1],
        )

        if actual_num_themes < 1:
            return [], []

        model = LatentDirichletAllocation(
            n_components=actual_num_themes,
            random_state=42,
            max_iter=75,
            learning_method="batch",
            doc_topic_prior=0.10,
            topic_word_prior=0.05,
        )

        document_theme_matrix = model.fit_transform(matrix)
        feature_names = vectorizer.get_feature_names_out()

        lda_candidates = self._build_lda_theme_candidates(
            model=model,
            feature_names=feature_names,
        )

        lda_theme_id_to_examples = self._build_examples_by_lda_theme(
            document_theme_matrix=document_theme_matrix,
            valid_documents=valid_documents,
            lda_candidates=lda_candidates,
        )

        canonical_documents = self._get_canonical_theme_documents()

        themes, lda_theme_id_to_resolved_theme_id = self._resolve_lda_themes(
            lda_candidates=lda_candidates,
            canonical_documents=canonical_documents,
            lda_theme_id_to_examples=lda_theme_id_to_examples,
        )

        assignments = self._build_assignments(
            document_theme_matrix=document_theme_matrix,
            original_indices=original_indices,
            lda_candidates=lda_candidates,
            lda_theme_id_to_resolved_theme_id=lda_theme_id_to_resolved_theme_id,
        )

        return themes, assignments

    def _build_lda_theme_candidates(
        self,
        model: LatentDirichletAllocation,
        feature_names,
    ) -> list[LdaThemeCandidate]:
        candidates: list[LdaThemeCandidate] = []

        for lda_theme_id, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-self.num_keywords:][::-1]
            keywords = [feature_names[index] for index in top_indices]
            label = keywords[0].title() if keywords else "Suggested Theme"
            weight = float(topic[top_indices].mean()) if len(top_indices) else 0.0

            candidates.append(
                LdaThemeCandidate(
                    lda_theme_id=lda_theme_id,
                    label=label,
                    weight=weight,
                    keywords=keywords,
                    match_text=" ".join([label, *keywords]).strip(),
                )
            )

        return candidates

    def _build_examples_by_lda_theme(
        self,
        document_theme_matrix,
        valid_documents: list[str],
        lda_candidates: list[LdaThemeCandidate],
    ) -> dict[int, list[str]]:
        """Return the strongest representative diary entries for each topic."""
        weighted_examples: dict[int, list[tuple[float, str]]] = {
            candidate.lda_theme_id: []
            for candidate in lda_candidates
        }

        for document_row_index, theme_weights in enumerate(document_theme_matrix):
            lda_theme_id = int(theme_weights.argmax())
            topic_weight = float(theme_weights[lda_theme_id])
            document = valid_documents[document_row_index]

            if document:
                weighted_examples.setdefault(lda_theme_id, []).append(
                    (topic_weight, document)
                )

        examples_by_theme_id: dict[int, list[str]] = {}

        for lda_theme_id, examples in weighted_examples.items():
            examples.sort(key=lambda item: item[0], reverse=True)
            examples_by_theme_id[lda_theme_id] = [
                document
                for _, document in examples
            ]

        return examples_by_theme_id

    def _is_catalog_match_accepted(self, catalog_match: dict | None) -> bool:
        """
        Decide whether the best approved-catalogue match is strong enough.

        Matching is always attempted first. A topic is mapped to the approved
        catalogue only when the combined score reaches the configured
        threshold and the match has support from either the LDA keywords or
        multiple representative diary entries. Otherwise, the topic falls
        through to NLP suggestion creation.
        """
        if not catalog_match:
            return False

        combined_score = float(catalog_match.get("theme_weight") or 0.0)
        keyword_score = float(
            catalog_match.get("keyword_match_weight") or 0.0
        )
        example_vote_share = float(
            catalog_match.get("example_vote_share") or 0.0
        )
        example_votes = int(catalog_match.get("example_votes") or 0)

        if combined_score < self.theme_weight_threshold:
            return False

        has_keyword_support = keyword_score >= 0.05
        has_repeated_entry_support = (
            example_votes >= 2
            and example_vote_share >= 0.40
        )

        return has_keyword_support or has_repeated_entry_support

    def _resolve_lda_themes(
        self,
        lda_candidates: list[LdaThemeCandidate],
        canonical_documents: list[CanonicalThemeDocument],
        lda_theme_id_to_examples: dict[int, list[str]],
    ) -> tuple[list[ThemeResult], dict[int, int]]:
        themes: list[ThemeResult] = []
        lda_theme_id_to_resolved_theme_id: dict[int, int] = {}
        resolved_theme_key_to_theme_id: dict[tuple[str, str], int] = {}

        catalog_matches = self._match_lda_candidates_to_catalog(
            lda_candidates=lda_candidates,
            canonical_documents=canonical_documents,
            lda_theme_id_to_examples=lda_theme_id_to_examples,
        )

        for candidate in lda_candidates:
            catalog_match = catalog_matches.get(candidate.lda_theme_id)

            if self._is_catalog_match_accepted(catalog_match):
                canonical_theme = catalog_match["canonical_theme"]
                resolved_key = (
                    "canonical",
                    str(canonical_theme.canonical_theme_id),
                )

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
                                "original_lda_theme": {
                                    "theme_id": candidate.lda_theme_id,
                                    "label": candidate.label,
                                    "weight": candidate.weight,
                                    "keywords": candidate.keywords,
                                },
                                "language": "english",
                                "vectorizer": "count",
                                "topic_model": "lda",
                                "num_themes": self.num_themes,
                                "num_keywords": self.num_keywords,
                                "ngram_range": [1, 3],
                            },
                        )
                    )

                lda_theme_id_to_resolved_theme_id[candidate.lda_theme_id] = (
                    resolved_theme_key_to_theme_id[resolved_key]
                )
                continue

            examples = lda_theme_id_to_examples.get(
                candidate.lda_theme_id,
                [],
            )
            suggestion_label = self._make_suggestion_label(
                candidate.keywords,
                examples,
            )

            suggested_theme = self._get_or_create_suggested_canonical_theme(
                candidate=candidate,
                suggestion_label=suggestion_label,
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
                            "attempted_canonical_theme_id": (
                                catalog_match["canonical_theme"].canonical_theme_id
                                if catalog_match
                                else None
                            ),
                            "attempted_canonical_theme_label": (
                                catalog_match["canonical_theme"].label
                                if catalog_match
                                else None
                            ),
                            "keyword_match_weight": (
                                catalog_match.get("keyword_match_weight")
                                if catalog_match
                                else None
                            ),
                            "example_match_weight": (
                                catalog_match.get("example_match_weight")
                                if catalog_match
                                else None
                            ),
                            "example_vote_share": (
                                catalog_match.get("example_vote_share")
                                if catalog_match
                                else None
                            ),
                            "example_votes": (
                                catalog_match.get("example_votes")
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
                            "original_lda_theme": {
                                "theme_id": candidate.lda_theme_id,
                                "label": candidate.label,
                                "weight": candidate.weight,
                                "keywords": candidate.keywords,
                            },
                            "language": "english",
                            "vectorizer": "count",
                            "topic_model": "lda",
                            "num_themes": self.num_themes,
                            "num_keywords": self.num_keywords,
                            "ngram_range": [1, 3],
                        },
                    )
                )

            lda_theme_id_to_resolved_theme_id[candidate.lda_theme_id] = (
                resolved_theme_key_to_theme_id[resolved_key]
            )

        unresolved_topic_ids = {
            candidate.lda_theme_id
            for candidate in lda_candidates
            if candidate.lda_theme_id not in lda_theme_id_to_resolved_theme_id
        }

        if unresolved_topic_ids:
            raise RuntimeError(
                "Every LDA topic must resolve to a canonical or suggested "
                f"theme. Unresolved topic ids: {sorted(unresolved_topic_ids)}"
            )

        return themes, lda_theme_id_to_resolved_theme_id

    def _get_or_create_suggested_canonical_theme(
        self,
        candidate: LdaThemeCandidate,
        suggestion_label: str,
        examples: list[str],
    ) -> CanonicalTheme:
        existing_theme = (
            CanonicalTheme.objects
            .filter(name__iexact=suggestion_label)
            .first()
        )

        if existing_theme:
            return existing_theme

        return CanonicalTheme.objects.create(
            name=suggestion_label,
            description=(
                "NLP-suggested theme produced by TF-IDF + LDA because "
                "the generated topic did not strongly match an approved "
                "canonical theme."
            ),
            aliases=candidate.keywords,
            examples="\n".join(examples[:MAX_MATCHING_EXAMPLES]),
            source=ThemeAndIssueSource.NLP,
            status=ThemeAndIssueStatus.SUGGESTED,
            is_active=True,
        )

    def _match_lda_candidates_to_catalog(
        self,
        lda_candidates: list[LdaThemeCandidate],
        canonical_documents: list[CanonicalThemeDocument],
        lda_theme_id_to_examples: dict[int, list[str]],
    ) -> dict[int, dict]:
        """
        Match LDA topics to approved canonical themes using evidence voting.

        Topic keywords are compared separately from representative diary
        entries. Each diary entry votes for its strongest canonical match.
        This avoids diluting canonical signals inside one oversized text blob.
        """
        if not lda_candidates or not canonical_documents:
            return {}

        catalog_texts = [theme.text for theme in canonical_documents]
        matches: dict[int, dict] = {}

        for candidate in lda_candidates:
            examples = lda_theme_id_to_examples.get(
                candidate.lda_theme_id,
                [],
            )[:MAX_MATCHING_EXAMPLES]

            keyword_text = " ".join(candidate.keywords).strip()
            evidence_texts = [keyword_text, *examples]

            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=self.max_features,
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 3),
                sublinear_tf=True,
            )

            try:
                matrix = vectorizer.fit_transform(
                    [*evidence_texts, *catalog_texts]
                )
            except ValueError:
                continue

            evidence_matrix = matrix[:len(evidence_texts)]
            catalog_matrix = matrix[len(evidence_texts):]
            similarities = cosine_similarity(
                evidence_matrix,
                catalog_matrix,
            )

            keyword_scores = similarities[0]
            example_scores = similarities[1:]

            vote_counts = [0] * len(canonical_documents)
            vote_strengths = [0.0] * len(canonical_documents)

            for row in example_scores:
                best_index = int(row.argmax())
                best_score = float(row[best_index])
                vote_counts[best_index] += 1
                vote_strengths[best_index] += best_score

            ranked_matches: list[dict] = []

            for catalog_index, canonical_theme in enumerate(canonical_documents):
                keyword_score = float(keyword_scores[catalog_index])
                votes = vote_counts[catalog_index]
                average_vote_strength = (
                    vote_strengths[catalog_index] / votes
                    if votes
                    else 0.0
                )
                vote_share = (
                    votes / len(examples)
                    if examples
                    else 0.0
                )

                combined_score = (
                    (0.25 * keyword_score)
                    + (0.50 * average_vote_strength)
                    + (0.25 * vote_share)
                )

                ranked_matches.append(
                    {
                        "canonical_theme": canonical_theme,
                        "theme_weight": combined_score,
                        "keyword_match_weight": keyword_score,
                        "example_match_weight": average_vote_strength,
                        "example_vote_share": vote_share,
                        "example_votes": votes,
                    }
                )

            ranked_matches.sort(
                key=lambda item: item["theme_weight"],
                reverse=True,
            )

            best_match = ranked_matches[0]

            print("\nCOUNT_LDA EVIDENCE-VOTE MATCH DEBUG")
            print("------------------------------------")
            print(f"LDA candidate id: {candidate.lda_theme_id}")
            print(f"LDA candidate label: {candidate.label}")
            print(f"LDA candidate keywords: {candidate.keywords}")
            print(f"Representative examples: {len(examples)}")
            print(f"Threshold: {self.theme_weight_threshold:.4f}")
            print("Top catalogue matches:")

            for rank, result in enumerate(ranked_matches[:5], start=1):
                theme = result["canonical_theme"]
                print(
                    f"  {rank}. id={theme.canonical_theme_id} | "
                    f"label={theme.label!r} | "
                    f"combined={result['theme_weight']:.4f} | "
                    f"keyword={result['keyword_match_weight']:.4f} | "
                    f"entry={result['example_match_weight']:.4f} | "
                    f"votes={result['example_votes']}/{len(examples)}"
                )

            print("------------------------------------\n")

            matches[candidate.lda_theme_id] = best_match

        return matches

    def _build_assignments(
        self,
        document_theme_matrix,
        original_indices: list[int],
        lda_candidates: list[LdaThemeCandidate],
        lda_theme_id_to_resolved_theme_id: dict[int, int],
    ) -> list[dict]:
        assignments: list[dict] = []

        for document_row_index, theme_weights in enumerate(document_theme_matrix):
            lda_theme_id = int(theme_weights.argmax())
            lda_theme_weight = float(theme_weights[lda_theme_id])
            weight_total = float(theme_weights.sum())
            assignment_weight = (
                lda_theme_weight / weight_total
                if weight_total
                else 0.0
            )

            resolved_theme_id = lda_theme_id_to_resolved_theme_id.get(lda_theme_id)

            if resolved_theme_id is None:
                continue

            candidate = lda_candidates[lda_theme_id]

            assignments.append(
                {
                    "document_index": original_indices[document_row_index],
                    "theme_id": resolved_theme_id,
                    "theme_weight": assignment_weight,
                    "source_lda_theme_id": candidate.lda_theme_id,
                    "source_lda_theme_label": candidate.label,
                    "source_lda_theme_weight": lda_theme_weight,
                }
            )

        return assignments

    def _get_canonical_theme_documents(self) -> list[CanonicalThemeDocument]:
        """Build matching documents from approved, active catalogue themes only."""
        canonical_themes = (
            CanonicalTheme.objects
            .filter(
                is_active=True,
                status=ThemeAndIssueStatus.APPROVED,
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
                    text=" ".join(
                        part
                        for part in text_parts
                        if part
                    ).strip(),
                    keywords=keywords,
                )
            )

        return documents

    def _make_suggestion_label(
        self,
        keywords: list[str],
        examples: list[str] | None = None,
    ) -> str:
        """
        Create a deterministic label for an unmatched LDA topic.

        Catalogue matching has already failed before this method is called.
        The label is always derived from the topic's own LDA keywords:

        1. Prefer a repeated, meaningful multi-word phrase.
        2. Otherwise prefer the strongest meaningful multi-word phrase.
        3. Otherwise combine the two strongest meaningful unigrams.
        4. As a final fallback, use the strongest available LDA keyword.

        This guarantees that every LDA topic can resolve to either a canonical
        theme or a suggested theme, so no diary entry loses its assignment.
        """
        examples = examples or []
        cleaned_keywords = [
            re.sub(r"\s+", " ", keyword).strip()
            for keyword in keywords
            if keyword and keyword.strip()
        ]
        normalized_examples = [
            re.sub(r"\s+", " ", example).casefold()
            for example in examples
            if example and example.strip()
        ]

        # Prefer a phrase that is repeated in representative diary evidence.
        for keyword in cleaned_keywords:
            tokens = keyword.casefold().split()

            if len(tokens) < 2:
                continue

            if all(token in SUGGESTION_CONTEXT_TERMS for token in tokens):
                continue

            occurrence_count = sum(
                1
                for example in normalized_examples
                if keyword.casefold() in example
            )

            if occurrence_count >= 2:
                return keyword.title()

        # Otherwise use the strongest meaningful multi-word LDA phrase.
        for keyword in cleaned_keywords:
            tokens = keyword.casefold().split()

            if (
                len(tokens) >= 2
                and any(
                    token not in SUGGESTION_CONTEXT_TERMS
                    for token in tokens
                )
            ):
                return keyword.title()

        # Otherwise combine the two strongest meaningful unigrams.
        meaningful_unigrams: list[str] = []

        for keyword in cleaned_keywords:
            tokens = keyword.casefold().split()

            if (
                len(tokens) == 1
                and tokens[0] not in SUGGESTION_CONTEXT_TERMS
                and tokens[0] not in meaningful_unigrams
            ):
                meaningful_unigrams.append(tokens[0])

            if len(meaningful_unigrams) == 2:
                return " ".join(meaningful_unigrams).title()

        # Final fallback: preserve the strongest LDA keyword rather than
        # dropping the topic and leaving entries without an assignment.
        if cleaned_keywords:
            return cleaned_keywords[0].title()

        return "Suggested Theme"

    def _clean_list(self, value) -> list[str]:
        if not value:
            return []

        if isinstance(value, (list, tuple, set)):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        if isinstance(value, str):
            return [
                item.strip()
                for item in re.split(r"[|,;\n]+", value)
                if item.strip()
            ]

        return []

    def _split_examples(self, value) -> list[str]:
        if not value:
            return []

        if isinstance(value, (list, tuple, set)):
            return [
                str(example).strip()
                for example in value
                if str(example).strip()
            ]

        return [
            example.strip()
            for example in re.split(r"[\n;]+", str(value))
            if example.strip()
        ]
