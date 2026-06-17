from __future__ import annotations

from dataclasses import dataclass
import re

from django.db.models import Q
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from studies.models import CanonicalTheme, ThemeAndIssueSource, ThemeAndIssueStatus
from studies.services.contracts import BaseThemeExtractor, ThemeResult

MODEL_NAME = "TF-IDF + LDA + Canonical Theme Matching"
THEME_WEIGHT_THRESHOLD = 0.30


@dataclass(frozen=True)
class CanonicalThemeDocument:
    canonical_theme_id: int
    label: str
    text: str
    keywords: list[str]


@dataclass(frozen=True)
class LdaThemeCandidate:
    lda_theme_id: int
    label: str
    weight: float
    keywords: list[str]
    match_text: str


class TfidfLdaThemeExtractor(BaseThemeExtractor):
    method_name = "tfidf_lda"

    def __init__(
        self,
        num_themes: int = 5,
        num_keywords: int = 8,
        max_features: int = 1000,
        theme_weight_threshold: float = THEME_WEIGHT_THRESHOLD,
    ):
        self.num_themes = num_themes
        self.num_keywords = num_keywords
        self.max_features = max_features
        self.theme_weight_threshold = theme_weight_threshold

    def extract(self, documents: list[str]) -> tuple[list[ThemeResult], list[dict]]:
        """
        Extract themes with TF-IDF + LDA, then resolve those LDA themes against
        the canonical theme catalogue.

        Rule:
        1. TF-IDF + LDA proposes themes from the diary entries.
        2. Each proposed LDA theme is compared against the canonical catalogue.
        3. If the catalogue match weight is greater than THEME_WEIGHT_THRESHOLD,
           the catalogue theme is used.
        4. Otherwise, the LDA theme remains a suggested theme.
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

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            min_df=1,
            max_df=0.95,
            ngram_range=(2, 3),
        )

        matrix = vectorizer.fit_transform(valid_documents)

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
            max_iter=20,
            learning_method="batch",
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
            label = self._make_suggestion_label(keywords)
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
        examples_by_theme_id = {
            candidate.lda_theme_id: []
            for candidate in lda_candidates
        }

        for document_row_index, theme_weights in enumerate(document_theme_matrix):
            lda_theme_id = int(theme_weights.argmax())
            document = valid_documents[document_row_index]

            if document:
                examples_by_theme_id.setdefault(lda_theme_id, []).append(document)

        return examples_by_theme_id

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
        )

        for candidate in lda_candidates:
            catalog_match = catalog_matches.get(candidate.lda_theme_id)

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
                                "original_lda_theme": {
                                    "theme_id": candidate.lda_theme_id,
                                    "label": candidate.label,
                                    "weight": candidate.weight,
                                    "keywords": candidate.keywords,
                                },
                                "language": "english",
                                "vectorizer": "tfidf",
                                "topic_model": "lda",
                                "ngram_range": [2, 3],
                            },
                        )
                    )

                lda_theme_id_to_resolved_theme_id[candidate.lda_theme_id] = resolved_theme_key_to_theme_id[resolved_key]
                continue

            examples = lda_theme_id_to_examples.get(candidate.lda_theme_id, [])

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
                            "original_lda_theme": {
                                "theme_id": candidate.lda_theme_id,
                                "label": candidate.label,
                                "weight": candidate.weight,
                                "keywords": candidate.keywords,
                            },
                            "language": "english",
                            "vectorizer": "tfidf",
                            "topic_model": "lda",
                            "num_keywords": len(candidate.keywords),
                            "ngram_range": [2, 3],
                        },
                    )
                )

            lda_theme_id_to_resolved_theme_id[candidate.lda_theme_id] = resolved_theme_key_to_theme_id[resolved_key]

        return themes, lda_theme_id_to_resolved_theme_id

    def _get_or_create_suggested_canonical_theme(
        self,
        candidate: LdaThemeCandidate,
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
                "NLP-suggested theme produced by TF-IDF + LDA because "
                "the generated theme did not strongly match the canonical theme catalog."
            ),
            aliases=candidate.keywords,
            examples="\n".join(examples[:5]),
            source=ThemeAndIssueSource.NLP,
            status=ThemeAndIssueStatus.SUGGESTED,
            is_active=True,
        )

    def _match_lda_candidates_to_catalog(
        self,
        lda_candidates: list[LdaThemeCandidate],
        canonical_documents: list[CanonicalThemeDocument],
    ) -> dict[int, dict]:
        if not lda_candidates or not canonical_documents:
            return {}

        candidate_texts = [candidate.match_text for candidate in lda_candidates]
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
            candidate = lda_candidates[candidate_index]

            top_match_indices = similarities.argsort()[-5:][::-1]

            print("\nTFIDF_LDA CATALOG MATCH DEBUG")
            print("--------------------------------")
            print(f"LDA candidate id: {candidate.lda_theme_id}")
            print(f"LDA candidate label: {candidate.label}")
            print(f"LDA candidate weight: {candidate.weight:.4f}")
            print(f"LDA candidate keywords: {candidate.keywords}")
            print(f"LDA candidate match text: {candidate.match_text}")
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

            matches[candidate.lda_theme_id] = {
                "theme_weight": best_score,
                "canonical_theme": canonical_documents[best_catalog_index],
            }

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
