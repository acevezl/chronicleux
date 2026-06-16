from __future__ import annotations

from dataclasses import dataclass
import re

from django.db.models import Q
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from studies.models import CanonicalTheme, ThemeAndIssueStatus
from studies.services.contracts import BaseThemeExtractor, ThemeResult

MODEL_NAME = "TF-IDF + Canonical Themes + NMF Suggestions"


@dataclass(frozen=True)
class CanonicalThemeDocument:
    local_theme_id: int
    canonical_theme_id: int
    label: str
    text: str
    keywords: list[str]


class TfidfNmfThemeExtractor(BaseThemeExtractor):
    method_name = "tfidf_nmf"

    def __init__(
        self,
        num_themes: int = 5,
        num_keywords: int = 8,
        max_features: int = 1000,
        canonical_confidence_threshold: float = 0.12,
        suggestion_confidence_threshold: float = 0.35,
        suggestion_margin: float = 0.08,
    ):
        self.num_themes = num_themes
        self.num_keywords = num_keywords
        self.max_features = max_features
        self.canonical_confidence_threshold = canonical_confidence_threshold
        self.suggestion_confidence_threshold = suggestion_confidence_threshold
        self.suggestion_margin = suggestion_margin

    def extract(self, documents: list[str]) -> tuple[list[ThemeResult], list[dict]]:
        valid_items = [
            (index, document.strip())
            for index, document in enumerate(documents)
            if document and document.strip()
        ]

        if not valid_items:
            return [], []

        original_indices = [item[0] for item in valid_items]
        valid_documents = [item[1] for item in valid_items]

        canonical_documents = self._get_canonical_theme_documents()

        themes: list[ThemeResult] = []
        assignments: list[dict] = []

        canonical_assignments_by_valid_index: dict[int, dict] = {}
        weak_valid_indices: list[int] = list(range(len(valid_documents)))

        if canonical_documents:
            (
                canonical_themes,
                canonical_assignments_by_valid_index,
                weak_valid_indices,
            ) = self._assign_canonical_themes(
                valid_documents=valid_documents,
                original_indices=original_indices,
                canonical_documents=canonical_documents,
            )

            themes.extend(canonical_themes)

            assignments.extend(
                assignment
                for valid_index, assignment in canonical_assignments_by_valid_index.items()
                if valid_index not in weak_valid_indices
            )

        suggestion_themes, suggestion_assignments = self._suggest_themes_for_weak_documents(
            valid_documents=valid_documents,
            original_indices=original_indices,
            weak_valid_indices=weak_valid_indices,
            starting_theme_id=len(themes),
            canonical_assignments_by_valid_index=canonical_assignments_by_valid_index,
        )

        themes.extend(suggestion_themes)

        suggestion_assignment_doc_indices = {
            assignment["document_index"]
            for assignment in suggestion_assignments
        }

        assignments.extend(suggestion_assignments)

        # Keep weak canonical matches only when NMF did not produce a stronger suggestion.
        assignments.extend(
            assignment
            for valid_index, assignment in canonical_assignments_by_valid_index.items()
            if (
                valid_index in weak_valid_indices
                and assignment["document_index"] not in suggestion_assignment_doc_indices
            )
        )

        return themes, assignments

    def _get_canonical_theme_documents(self) -> list[CanonicalThemeDocument]:
        canonical_themes = (
            CanonicalTheme.objects
            .filter(is_active=True)
            .filter(
                Q(status=ThemeAndIssueStatus.APPROVED)
                | Q(status__isnull=True)
                | Q(status="")
            )
            .order_by("name")
        )

        documents: list[CanonicalThemeDocument] = []

        for local_theme_id, theme in enumerate(canonical_themes):
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
                    local_theme_id=local_theme_id,
                    canonical_theme_id=theme.id,
                    label=theme.name,
                    text=" ".join(part for part in text_parts if part).strip(),
                    keywords=keywords,
                )
            )

        return documents

    def _assign_canonical_themes(
        self,
        valid_documents: list[str],
        original_indices: list[int],
        canonical_documents: list[CanonicalThemeDocument],
    ) -> tuple[list[ThemeResult], dict[int, dict], list[int]]:
        catalog_texts = [theme.text for theme in canonical_documents]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 3),
        )

        matrix = vectorizer.fit_transform([*catalog_texts, *valid_documents])

        canonical_matrix = matrix[:len(catalog_texts)]
        document_matrix = matrix[len(catalog_texts):]

        similarity_matrix = cosine_similarity(document_matrix, canonical_matrix)

        themes = [
            ThemeResult(
                theme_id=theme.local_theme_id,
                label=theme.label,
                weight=0.0,
                keywords=theme.keywords,
                method=self.method_name,
                metadata={
                    "model": MODEL_NAME,
                    "match_type": "canonical",
                    "canonical_theme_id": theme.canonical_theme_id,
                    "canonical_confidence_threshold": self.canonical_confidence_threshold,
                    "language": "english",
                    "vectorizer": "tfidf",
                    "ngram_range": [1, 3],
                },
            )
            for theme in canonical_documents
        ]

        assignments_by_valid_index: dict[int, dict] = {}
        weak_valid_indices: list[int] = []

        for valid_index, similarities in enumerate(similarity_matrix):
            best_canonical_index = int(similarities.argmax())
            best_score = float(similarities[best_canonical_index])
            canonical_theme = canonical_documents[best_canonical_index]

            if best_score < self.canonical_confidence_threshold:
                weak_valid_indices.append(valid_index)

            assignments_by_valid_index[valid_index] = {
                "document_index": original_indices[valid_index],
                "theme_id": canonical_theme.local_theme_id,
                "theme_weight": best_score,
                "match_type": "canonical",
                "canonical_theme_id": canonical_theme.canonical_theme_id,
                "is_weak_match": best_score < self.canonical_confidence_threshold,
            }

        return themes, assignments_by_valid_index, weak_valid_indices

    def _suggest_themes_for_weak_documents(
        self,
        valid_documents: list[str],
        original_indices: list[int],
        weak_valid_indices: list[int],
        starting_theme_id: int,
        canonical_assignments_by_valid_index: dict[int, dict],
    ) -> tuple[list[ThemeResult], list[dict]]:
        weak_documents = [valid_documents[index] for index in weak_valid_indices]

        if not weak_documents:
            return [], []

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            min_df=1,
            max_df=0.95,
            ngram_range=(2, 3),
        )

        matrix = vectorizer.fit_transform(weak_documents)

        actual_num_themes = min(
            self.num_themes,
            matrix.shape[0],
            matrix.shape[1],
        )

        if actual_num_themes < 1:
            return [], []

        model = NMF(
            n_components=actual_num_themes,
            random_state=42,
            init="nndsvda",
            max_iter=500,
        )

        document_theme_matrix = model.fit_transform(matrix)
        feature_names = vectorizer.get_feature_names_out()

        themes: list[ThemeResult] = []

        for nmf_theme_index, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-self.num_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            label = self._make_suggestion_label(keywords)
            local_theme_id = starting_theme_id + nmf_theme_index

            themes.append(
                ThemeResult(
                    theme_id=local_theme_id,
                    label=label,
                    weight=float(topic[top_indices].mean()) if len(top_indices) else 0.0,
                    keywords=keywords,
                    method=self.method_name,
                    metadata={
                        "model": MODEL_NAME,
                        "match_type": "suggested",
                        "is_catalog_suggestion": True,
                        "suggested_theme": {
                            "name": label,
                            "description": (
                                "NLP-suggested theme produced because one or more entries "
                                "did not strongly match the approved canonical theme catalog."
                            ),
                            "aliases": keywords,
                            "source": "NLP",
                            "status": "SUGGESTED",
                        },
                        "suggestion_confidence_threshold": self.suggestion_confidence_threshold,
                        "suggestion_margin": self.suggestion_margin,
                        "language": "english",
                        "vectorizer": "tfidf",
                        "topic_model": "nmf",
                        "num_keywords": len(keywords),
                        "ngram_range": [2, 3],
                    },
                )
            )

        assignments: list[dict] = []

        for weak_row_index, theme_weights in enumerate(document_theme_matrix):
            valid_index = weak_valid_indices[weak_row_index]
            original_document_index = original_indices[valid_index]

            top_nmf_theme_index = int(theme_weights.argmax())
            top_nmf_theme_weight = float(theme_weights[top_nmf_theme_index])

            weight_total = float(theme_weights.sum())
            suggestion_confidence = (
                top_nmf_theme_weight / weight_total
                if weight_total
                else 0.0
            )

            canonical_assignment = canonical_assignments_by_valid_index.get(valid_index)
            canonical_confidence = 0.0

            if canonical_assignment:
                canonical_confidence = float(canonical_assignment.get("theme_weight") or 0.0)

            is_stronger_than_canonical = (
                suggestion_confidence >= self.suggestion_confidence_threshold
                and suggestion_confidence >= canonical_confidence + self.suggestion_margin
            )

            if not is_stronger_than_canonical:
                continue

            assignments.append(
                {
                    "document_index": original_document_index,
                    "theme_id": starting_theme_id + top_nmf_theme_index,
                    "theme_weight": suggestion_confidence,
                    "match_type": "suggested",
                    "canonical_confidence": canonical_confidence,
                }
            )

        return themes, assignments

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