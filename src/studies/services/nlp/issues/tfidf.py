from __future__ import annotations

import re

from django.db.models import Q
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from studies.models import CanonicalIssue, ThemeAndIssueSource, ThemeAndIssueStatus
from studies.services.contracts import BaseIssueDetector, IssueResult


MODEL_NAME = "TF-IDF + Canonical Issue Matching"
ISSUE_MATCH_THRESHOLD = 0.25


ISSUE_CUES = [
    "problem","issue","bug","broken","confusing","confused",
    "frustrating","frustrated","hard","hard to","difficult",
    "annoying","slow","lag","crash","stuck","couldn't","cannot",
    "can't","unclear","missing","failed","error",
    "didn't work","did not work","doesn't work","does not work","trouble",
    "not obvious","not intuitive","too many steps","took too long",
    "clutter","cluttered","hidden","lacks"
]


class TfidfIssueDetector(BaseIssueDetector):
    method_name = "tfidf"

    def __init__(
        self,
        num_keywords: int = 8,
        max_features: int = 1000,
        issue_match_threshold: float = ISSUE_MATCH_THRESHOLD,
    ):
        self.num_keywords = num_keywords
        self.max_features = max_features
        self.issue_match_threshold = issue_match_threshold

    def detect(self, documents: list[str]) -> tuple[list[IssueResult], list[dict]]:
        """
        Detect UX issues with this workflow:

        1. detect: keep only entries that appear to describe UX friction.
        2. match: compare each issue-like entry against the canonical issue catalogue.
        3. suggest if unmatched: create a suggested CanonicalIssue when no strong match exists.

        This method does not use NMF. It does not generate issue clusters.
        """

        issue_keyword_catalog = self._get_issue_keyword_catalog()

        issue_candidates = self._build_issue_candidates(
            documents=documents,
            issue_keyword_catalog=issue_keyword_catalog,
        )

        if not issue_candidates:
            return [], []

        canonical_documents = self._get_canonical_issue_documents()

        catalog_matches = self._match_issue_candidates_to_catalog(
            issue_candidates=issue_candidates,
            canonical_documents=canonical_documents,
        )

        issues, candidate_id_to_resolved_issue_id = self._resolve_issue_candidates(
            issue_candidates=issue_candidates,
            canonical_documents=canonical_documents,
            catalog_matches=catalog_matches,
        )

        assignments = self._build_assignments(
            issue_candidates=issue_candidates,
            issues=issues,
            candidate_id_to_resolved_issue_id=candidate_id_to_resolved_issue_id,
        )

        return issues, assignments
    
    def _get_issue_keyword_catalog(self) -> dict[int, dict]:
        canonical_issues = (
            CanonicalIssue.objects
            .filter(is_active=True)
            .filter(
                Q(status=ThemeAndIssueStatus.APPROVED)
                | Q(status=ThemeAndIssueStatus.SUGGESTED)
                | Q(status__isnull=True)
                | Q(status="")
            )
            .order_by("name")
        )

        catalog = {}

        for issue in canonical_issues:
            aliases = self._clean_list(issue.aliases)

            keywords = [
                issue.name,
                *aliases,
            ]

            keywords = [
                keyword.strip().lower()
                for keyword in keywords
                if keyword and keyword.strip()
            ]

            catalog[issue.id] = {
                "canonical_issue": issue,
                "label": issue.name,
                "keywords": keywords,
            }

        return catalog
    
    def _get_catalogue_keyword_hits(
        self,
        document: str,
        issue_keyword_catalog: dict[int, dict],
    ) -> list[dict]:
        text = f" {self._normalize_text(document).lower()} "

        hits = []

        for canonical_issue_id, issue_data in issue_keyword_catalog.items():
            matched_keywords = [
                keyword
                for keyword in issue_data["keywords"]
                if keyword and keyword in text
            ]

            if not matched_keywords:
                continue

            hits.append(
                {
                    "canonical_issue_id": canonical_issue_id,
                    "canonical_issue": issue_data["canonical_issue"],
                    "label": issue_data["label"],
                    "matched_keywords": matched_keywords,
                    "keyword_score": len(matched_keywords),
                }
            )

        return sorted(
            hits,
            key=lambda hit: hit["keyword_score"],
            reverse=True,
        )

    def _build_issue_candidates(
        self,
        documents: list[str],
        issue_keyword_catalog: dict[int, dict],
    ) -> list[dict]:
        valid_items = []

        for index, document in enumerate(documents):
            if not document or not document.strip():
                continue

            catalogue_hits = self._get_catalogue_keyword_hits(
                document=document,
                issue_keyword_catalog=issue_keyword_catalog,
            )

            has_issue_cue = self._has_issue_cue(document)

            if not has_issue_cue and not catalogue_hits:
                continue

            valid_items.append(
                {
                    "document_index": index,
                    "document": document.strip(),
                    "catalogue_hits": catalogue_hits,
                    "has_issue_cue": has_issue_cue,
                }
            )

        if not valid_items:
            return []

        valid_documents = [
            item["document"]
            for item in valid_items
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 3),
        )

        matrix = vectorizer.fit_transform(valid_documents)
        feature_names = vectorizer.get_feature_names_out()

        candidates = []

        for row_index, item in enumerate(valid_items):
            keywords = self._extract_keywords_from_row(
                matrix=matrix,
                row_index=row_index,
                feature_names=feature_names,
            )

            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "document_index": item["document_index"],
                    "document": item["document"],
                    "label": self._make_suggestion_label(item["document"]),
                    "keywords": keywords,
                    "match_text": " ".join([item["document"], *keywords]).strip(),
                    "catalogue_hits": item["catalogue_hits"],
                    "has_issue_cue": item["has_issue_cue"],
                }
            )

        return candidates

    def _match_issue_candidates_to_catalog(
        self,
        issue_candidates: list[dict],
        canonical_documents: list[dict],
    ) -> dict[int, dict]:
        if not issue_candidates or not canonical_documents:
            return {}

        candidate_texts = [
            candidate["match_text"]
            for candidate in issue_candidates
        ]

        catalog_texts = [
            issue["text"]
            for issue in canonical_documents
        ]

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

        matches = {}

        for candidate_index, similarities in enumerate(similarity_matrix):
            best_catalog_index = int(similarities.argmax())
            best_score = float(similarities[best_catalog_index])
            candidate = issue_candidates[candidate_index]

            matches[candidate["candidate_id"]] = {
                "issue_weight": best_score,
                "canonical_issue": canonical_documents[best_catalog_index],
            }

        return matches

    def _resolve_issue_candidates(
        self,
        issue_candidates: list[dict],
        canonical_documents: list[dict],
        catalog_matches: dict[int, dict],
    ) -> tuple[list[IssueResult], dict[int, int]]:
        issues = []
        candidate_id_to_resolved_issue_id = {}
        resolved_issue_key_to_issue_id = {}

        for candidate in issue_candidates:
            catalogue_hits = candidate.get("catalogue_hits", [])

            # 1. First priority: direct catalogue keyword / alias match.
            # If the entry directly contains a canonical issue name or alias,
            # trust the catalogue before TF-IDF.
            if catalogue_hits:
                best_hit = catalogue_hits[0]
                canonical_issue = best_hit["canonical_issue"]

                resolved_key = ("canonical", str(canonical_issue.id))

                if resolved_key not in resolved_issue_key_to_issue_id:
                    resolved_issue_id = len(issues)
                    resolved_issue_key_to_issue_id[resolved_key] = resolved_issue_id

                    issues.append(
                        IssueResult(
                            issue_id=resolved_issue_id,
                            label=canonical_issue.name,
                            weight=1.0,
                            keywords=best_hit["matched_keywords"],
                            method=self.method_name,
                            metadata={
                                "model": MODEL_NAME,
                                "match_type": "canonical",
                                "match_strategy": "catalogue_keyword",
                                "canonical_issue_id": canonical_issue.id,
                                "matched_keywords": best_hit["matched_keywords"],
                                "catalog_match_weight": 1.0,
                                "issue_match_threshold": self.issue_match_threshold,
                                "original_candidate": {
                                    "candidate_id": candidate["candidate_id"],
                                    "label": candidate["label"],
                                    "keywords": candidate["keywords"],
                                    "document_index": candidate["document_index"],
                                },
                                "language": "english",
                                "vectorizer": "tfidf",
                                "ngram_range": [1, 3],
                            },
                        )
                    )

                candidate_id_to_resolved_issue_id[candidate["candidate_id"]] = (
                    resolved_issue_key_to_issue_id[resolved_key]
                )
                continue

            catalog_match = catalog_matches.get(candidate["candidate_id"])

            # 2. Second priority: TF-IDF match against canonical issue documents.
            if (
                catalog_match
                and catalog_match["issue_weight"] > self.issue_match_threshold
            ):
                canonical_issue = catalog_match["canonical_issue"]
                resolved_key = ("canonical", str(canonical_issue["canonical_issue_id"]))

                if resolved_key not in resolved_issue_key_to_issue_id:
                    resolved_issue_id = len(issues)
                    resolved_issue_key_to_issue_id[resolved_key] = resolved_issue_id

                    issues.append(
                        IssueResult(
                            issue_id=resolved_issue_id,
                            label=canonical_issue["label"],
                            weight=catalog_match["issue_weight"],
                            keywords=canonical_issue["keywords"],
                            method=self.method_name,
                            metadata={
                                "model": MODEL_NAME,
                                "match_type": "canonical",
                                "match_strategy": "tfidf_similarity",
                                "canonical_issue_id": canonical_issue["canonical_issue_id"],
                                "catalog_match_weight": catalog_match["issue_weight"],
                                "issue_match_threshold": self.issue_match_threshold,
                                "original_candidate": {
                                    "candidate_id": candidate["candidate_id"],
                                    "label": candidate["label"],
                                    "keywords": candidate["keywords"],
                                    "document_index": candidate["document_index"],
                                },
                                "language": "english",
                                "vectorizer": "tfidf",
                                "ngram_range": [1, 3],
                            },
                        )
                    )

                candidate_id_to_resolved_issue_id[candidate["candidate_id"]] = (
                    resolved_issue_key_to_issue_id[resolved_key]
                )
                continue

            # 3. Fallback: issue-like entry, but no confident catalogue match.
            # Create a suggested CanonicalIssue for evaluator review.
            suggested_issue = self._get_or_create_suggested_canonical_issue(
                candidate=candidate,
                catalog_match=catalog_match,
            )

            resolved_key = ("suggested", str(suggested_issue.id))

            if resolved_key not in resolved_issue_key_to_issue_id:
                resolved_issue_id = len(issues)
                resolved_issue_key_to_issue_id[resolved_key] = resolved_issue_id

                issues.append(
                    IssueResult(
                        issue_id=resolved_issue_id,
                        label=suggested_issue.name,
                        weight=(
                            catalog_match["issue_weight"]
                            if catalog_match
                            else 0.0
                        ),
                        keywords=candidate["keywords"],
                        method=self.method_name,
                        metadata={
                            "model": MODEL_NAME,
                            "match_type": "suggested",
                            "match_strategy": "suggested_catalogue_issue",
                            "is_catalog_suggestion": True,
                            "canonical_issue_id": suggested_issue.id,
                            "catalog_match_weight": (
                                catalog_match["issue_weight"]
                                if catalog_match
                                else None
                            ),
                            "issue_match_threshold": self.issue_match_threshold,
                            "suggested_issue": {
                                "id": suggested_issue.id,
                                "name": suggested_issue.name,
                                "description": suggested_issue.description,
                                "aliases": suggested_issue.aliases or [],
                                "source": suggested_issue.source,
                                "status": suggested_issue.status,
                            },
                            "nearest_canonical_issue": (
                                {
                                    "id": catalog_match["canonical_issue"]["canonical_issue_id"],
                                    "name": catalog_match["canonical_issue"]["label"],
                                }
                                if catalog_match
                                else None
                            ),
                            "original_candidate": {
                                "candidate_id": candidate["candidate_id"],
                                "label": candidate["label"],
                                "keywords": candidate["keywords"],
                                "document_index": candidate["document_index"],
                            },
                            "language": "english",
                            "vectorizer": "tfidf",
                            "ngram_range": [1, 3],
                        },
                    )
                )

            candidate_id_to_resolved_issue_id[candidate["candidate_id"]] = (
                resolved_issue_key_to_issue_id[resolved_key]
            )

        return issues, candidate_id_to_resolved_issue_id

    def _build_assignments(
        self,
        issue_candidates: list[dict],
        issues: list[IssueResult],
        candidate_id_to_resolved_issue_id: dict[int, int],
    ) -> list[dict]:
        assignments = []

        issues_by_id = {
            issue.issue_id: issue
            for issue in issues
        }

        for candidate in issue_candidates:
            resolved_issue_id = candidate_id_to_resolved_issue_id.get(
                candidate["candidate_id"]
            )

            if resolved_issue_id is None:
                continue

            issue = issues_by_id.get(resolved_issue_id)

            if issue is None:
                continue

            assignments.append(
                {
                    "document_index": candidate["document_index"],
                    "issue_id": resolved_issue_id,
                    "issue_detected": True,
                    "issue_weight": issue.weight,
                    "source_candidate_id": candidate["candidate_id"],
                    "source_candidate_label": candidate["label"],
                    "source_candidate_keywords": candidate["keywords"],
                }
            )

        return assignments

    def _get_or_create_suggested_canonical_issue(
        self,
        candidate: dict,
        catalog_match: dict | None,
    ) -> CanonicalIssue:
        existing_issue = (
            CanonicalIssue.objects
            .filter(name__iexact=candidate["label"])
            .first()
        )

        if existing_issue:
            return existing_issue

        nearest_match_text = ""

        if catalog_match:
            nearest_issue = catalog_match["canonical_issue"]
            nearest_match_text = (
                f"\n\nNearest catalogue match: {nearest_issue['label']} "
                f"(score: {catalog_match['issue_weight']:.4f})."
            )

        return CanonicalIssue.objects.create(
            name=candidate["label"],
            description=(
                "NLP-suggested UX issue produced by TF-IDF because the diary entry "
                "appeared to describe a UX issue but did not strongly match the "
                "canonical issue catalogue."
                f"{nearest_match_text}"
            ),
            aliases=candidate["keywords"],
            examples=candidate["document"],
            source=ThemeAndIssueSource.NLP,
            status=ThemeAndIssueStatus.SUGGESTED,
            is_active=True,
        )

    def _get_canonical_issue_documents(self) -> list[dict]:
        canonical_issues = (
            CanonicalIssue.objects
            .filter(is_active=True)
            .filter(
                Q(status=ThemeAndIssueStatus.APPROVED)
                | Q(status=ThemeAndIssueStatus.SUGGESTED)
                | Q(status__isnull=True)
                | Q(status="")
            )
            .order_by("name")
        )

        documents = []

        for issue in canonical_issues:
            aliases = self._clean_list(issue.aliases)
            examples = self._split_examples(issue.examples)
            keywords = [issue.name, *aliases]

            text_parts = [
                issue.name,
                issue.description,
                " ".join(aliases),
                " ".join(examples),
            ]

            documents.append(
                {
                    "canonical_issue_id": issue.id,
                    "label": issue.name,
                    "text": " ".join(part for part in text_parts if part).strip(),
                    "keywords": keywords,
                }
            )

        return documents

    def _has_issue_cue(self, document: str) -> bool:
        text = f" {self._normalize_text(document).lower()} "

        return any(cue in text for cue in ISSUE_CUES)

    def _extract_keywords_from_row(
        self,
        matrix,
        row_index: int,
        feature_names,
    ) -> list[str]:
        row = matrix[row_index].toarray()[0]
        top_indices = row.argsort()[-self.num_keywords:][::-1]

        keywords = []

        for index in top_indices:
            if row[index] <= 0:
                continue

            keyword = feature_names[index].strip()

            if keyword:
                keywords.append(keyword)

        return keywords

    def _make_suggestion_label(self, document: str) -> str:
        text = self._normalize_text(document).lower()

        for cue in ISSUE_CUES:
            if cue in text:
                return f"{cue.title()}"

        return "Other"

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

    def _normalize_text(self, value: str) -> str:
        return " ".join((value or "").split())