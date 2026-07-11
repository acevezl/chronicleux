from __future__ import annotations

from dataclasses import dataclass
import re

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from studies.models import CanonicalTheme, ThemeAndIssueSource, ThemeAndIssueStatus
from studies.services.contracts import BaseThemeExtractor, ThemeResult


MODEL_NAME = "TF-IDF + NMF + Canonical Theme Matching"
THEME_WEIGHT_THRESHOLD = 0.15
NUM_THEMES = 12
NUM_KEYWORDS = 10
MAX_FEATURES = 1000
MAX_MATCHING_EXAMPLES = 5


@dataclass(frozen=True)
class CanonicalThemeDocument:
    canonical_theme_id: int
    label: str
    text: str
    keywords: list[str]


@dataclass(frozen=True)
class NmfThemeCandidate:
    nmf_theme_id: int
    label: str
    weight: float
    keywords: list[str]
    match_text: str


class TfidfNmfThemeExtractor(BaseThemeExtractor):
    method_name = "tfidf_nmf"

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
        Extract latent themes with TF-IDF + NMF and resolve them against the
        approved canonical theme catalogue.

        NMF discovers recurring vocabulary patterns. Canonical matching then
        compares each topic's label, keywords, and strongest diary examples
        against the catalogue. Topics below the matching threshold remain NLP
        suggestions for evaluator review.
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
            min_df=2,
            max_df=0.85,
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

        model = NMF(
            n_components=actual_num_themes,
            random_state=42,
            init="nndsvda",
            max_iter=500,
        )

        document_theme_matrix = model.fit_transform(matrix)
        feature_names = vectorizer.get_feature_names_out()

        nmf_candidates = self._build_nmf_theme_candidates(
            model=model,
            feature_names=feature_names,
        )

        nmf_theme_id_to_examples = self._build_examples_by_nmf_theme(
            document_theme_matrix=document_theme_matrix,
            valid_documents=valid_documents,
            nmf_candidates=nmf_candidates,
        )

        canonical_documents = self._get_canonical_theme_documents()

        themes, nmf_theme_id_to_resolved_theme_id = self._resolve_nmf_themes(
            nmf_candidates=nmf_candidates,
            canonical_documents=canonical_documents,
            nmf_theme_id_to_examples=nmf_theme_id_to_examples,
        )

        assignments = self._build_assignments(
            document_theme_matrix=document_theme_matrix,
            original_indices=original_indices,
            nmf_candidates=nmf_candidates,
            nmf_theme_id_to_resolved_theme_id=nmf_theme_id_to_resolved_theme_id,
        )

        return themes, assignments

    def _build_nmf_theme_candidates(
        self,
        model: NMF,
        feature_names,
    ) -> list[NmfThemeCandidate]:
        candidates: list[NmfThemeCandidate] = []

        for nmf_theme_id, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-self.num_keywords:][::-1]
            keywords = [feature_names[index] for index in top_indices]
            label = self._make_suggestion_label(keywords)
            weight = float(topic[top_indices].mean()) if len(top_indices) else 0.0

            candidates.append(
                NmfThemeCandidate(
                    nmf_theme_id=nmf_theme_id,
                    label=label,
                    weight=weight,
                    keywords=keywords,
                    match_text=" ".join([label, *keywords]).strip(),
                )
            )

        return candidates

    def _build_examples_by_nmf_theme(
        self,
        document_theme_matrix,
        valid_documents: list[str],
        nmf_candidates: list[NmfThemeCandidate],
    ) -> dict[int, list[str]]:
        """Return the strongest representative diary entries for each topic."""
        weighted_examples: dict[int, list[tuple[float, str]]] = {
            candidate.nmf_theme_id: []
            for candidate in nmf_candidates
        }

        for document_row_index, theme_weights in enumerate(document_theme_matrix):
            nmf_theme_id = int(theme_weights.argmax())
            topic_weight = float(theme_weights[nmf_theme_id])
            document = valid_documents[document_row_index]

            if document:
                weighted_examples.setdefault(nmf_theme_id, []).append(
                    (topic_weight, document)
                )

        examples_by_theme_id: dict[int, list[str]] = {}

        for nmf_theme_id, examples in weighted_examples.items():
            examples.sort(key=lambda item: item[0], reverse=True)
            examples_by_theme_id[nmf_theme_id] = [
                document
                for _, document in examples
            ]

        return examples_by_theme_id

    def _resolve_nmf_themes(
        self,
        nmf_candidates: list[NmfThemeCandidate],
        canonical_documents: list[CanonicalThemeDocument],
        nmf_theme_id_to_examples: dict[int, list[str]],
    ) -> tuple[list[ThemeResult], dict[int, int]]:
        themes: list[ThemeResult] = []
        nmf_theme_id_to_resolved_theme_id: dict[int, int] = {}
        resolved_theme_key_to_theme_id: dict[tuple[str, str], int] = {}

        catalog_matches = self._match_nmf_candidates_to_catalog(
            nmf_candidates=nmf_candidates,
            canonical_documents=canonical_documents,
            nmf_theme_id_to_examples=nmf_theme_id_to_examples,
        )

        for candidate in nmf_candidates:
            catalog_match = catalog_matches.get(candidate.nmf_theme_id)

            if (
                catalog_match
                and catalog_match["theme_weight"] >= self.theme_weight_threshold
            ):
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
                                "original_nmf_theme": {
                                    "theme_id": candidate.nmf_theme_id,
                                    "label": candidate.label,
                                    "weight": candidate.weight,
                                    "keywords": candidate.keywords,
                                },
                                "language": "english",
                                "vectorizer": "tfidf",
                                "topic_model": "nmf",
                                "num_themes": self.num_themes,
                                "num_keywords": self.num_keywords,
                                "ngram_range": [1, 3],
                            },
                        )
                    )

                nmf_theme_id_to_resolved_theme_id[candidate.nmf_theme_id] = (
                    resolved_theme_key_to_theme_id[resolved_key]
                )
                continue

            examples = nmf_theme_id_to_examples.get(candidate.nmf_theme_id, [])
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
                            "original_nmf_theme": {
                                "theme_id": candidate.nmf_theme_id,
                                "label": candidate.label,
                                "weight": candidate.weight,
                                "keywords": candidate.keywords,
                            },
                            "language": "english",
                            "vectorizer": "tfidf",
                            "topic_model": "nmf",
                            "num_themes": self.num_themes,
                            "num_keywords": self.num_keywords,
                            "ngram_range": [1, 3],
                        },
                    )
                )

            nmf_theme_id_to_resolved_theme_id[candidate.nmf_theme_id] = (
                resolved_theme_key_to_theme_id[resolved_key]
            )

        return themes, nmf_theme_id_to_resolved_theme_id

    def _get_or_create_suggested_canonical_theme(
        self,
        candidate: NmfThemeCandidate,
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
                "NLP-suggested theme produced by TF-IDF + NMF because "
                "the generated topic did not strongly match an approved "
                "canonical theme."
            ),
            aliases=candidate.keywords,
            examples="\n".join(examples[:MAX_MATCHING_EXAMPLES]),
            source=ThemeAndIssueSource.NLP,
            status=ThemeAndIssueStatus.SUGGESTED,
            is_active=True,
        )

    def _match_nmf_candidates_to_catalog(
        self,
        nmf_candidates: list[NmfThemeCandidate],
        canonical_documents: list[CanonicalThemeDocument],
        nmf_theme_id_to_examples: dict[int, list[str]],
    ) -> dict[int, dict]:
        if not nmf_candidates or not canonical_documents:
            return {}

        candidate_texts: list[str] = []

        for candidate in nmf_candidates:
            examples = nmf_theme_id_to_examples.get(
                candidate.nmf_theme_id,
                [],
            )

            candidate_text = " ".join(
                [
                    candidate.label,
                    *candidate.keywords,
                    *examples[:MAX_MATCHING_EXAMPLES],
                ]
            ).strip()

            candidate_texts.append(candidate_text)

        catalog_texts = [theme.text for theme in canonical_documents]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 3),
            sublinear_tf=True,
        )

        try:
            matrix = vectorizer.fit_transform([*candidate_texts, *catalog_texts])
        except ValueError:
            return {}

        candidate_matrix = matrix[:len(candidate_texts)]
        catalog_matrix = matrix[len(candidate_texts):]
        similarity_matrix = cosine_similarity(candidate_matrix, catalog_matrix)

        matches: dict[int, dict] = {}

        for candidate_index, similarities in enumerate(similarity_matrix):
            best_catalog_index = int(similarities.argmax())
            best_score = float(similarities[best_catalog_index])
            candidate = nmf_candidates[candidate_index]
            top_match_indices = similarities.argsort()[-5:][::-1]

            print("\nTFIDF_NMF CATALOG MATCH DEBUG")
            print("--------------------------------")
            print(f"NMF candidate id: {candidate.nmf_theme_id}")
            print(f"NMF candidate label: {candidate.label}")
            print(f"NMF candidate weight: {candidate.weight:.4f}")
            print(f"NMF candidate keywords: {candidate.keywords}")
            print(f"NMF candidate match text: {candidate_texts[candidate_index]}")
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

            matches[candidate.nmf_theme_id] = {
                "theme_weight": best_score,
                "canonical_theme": canonical_documents[best_catalog_index],
            }

        return matches

    def _build_assignments(
        self,
        document_theme_matrix,
        original_indices: list[int],
        nmf_candidates: list[NmfThemeCandidate],
        nmf_theme_id_to_resolved_theme_id: dict[int, int],
    ) -> list[dict]:
        assignments: list[dict] = []

        for document_row_index, theme_weights in enumerate(document_theme_matrix):
            nmf_theme_id = int(theme_weights.argmax())
            nmf_theme_weight = float(theme_weights[nmf_theme_id])
            weight_total = float(theme_weights.sum())
            assignment_weight = (
                nmf_theme_weight / weight_total
                if weight_total
                else 0.0
            )

            resolved_theme_id = nmf_theme_id_to_resolved_theme_id.get(nmf_theme_id)

            if resolved_theme_id is None:
                continue

            candidate = nmf_candidates[nmf_theme_id]

            assignments.append(
                {
                    "document_index": original_indices[document_row_index],
                    "theme_id": resolved_theme_id,
                    "theme_weight": assignment_weight,
                    "source_nmf_theme_id": candidate.nmf_theme_id,
                    "source_nmf_theme_label": candidate.label,
                    "source_nmf_theme_weight": nmf_theme_weight,
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

    def _make_suggestion_label(self, keywords: list[str]) -> str:
        """Create a readable fallback label from the first useful NMF phrase."""
        ignored_labels = {
            "netflix",
            "used netflix",
            "smart",
            "smart tv",
            "phone",
            "smartphone",
            "laptop",
            "tablet",
            "screen",
            "feel",
            "feel like",
        }

        for keyword in keywords:
            label = re.sub(r"\s+", " ", keyword).strip()

            if label and label.casefold() not in ignored_labels:
                return label.title()

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
