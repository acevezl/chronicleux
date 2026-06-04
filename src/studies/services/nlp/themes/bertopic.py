from bertopic import BERTopic

from studies.services.nlp.contracts import BaseThemeExtractor, ThemeResult

MODEL_NAME = "BERTopic"

class BertopicThemeExtractor(BaseThemeExtractor):

    method_name = "bertopic"
    
    def __init__(self, num_keywords: int = 8):
        self.num_keywords = num_keywords

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

        topic_model = BERTopic(
            language="english",
            calculate_probabilities=True,
            verbose=False,
        )

        topics, probabilities = topic_model.fit_transform(valid_documents)

        topic_ids = sorted({
            topic_id
            for topic_id in topics
            if topic_id != -1
        })

        themes = []

        for topic_id in topic_ids:
            topic_terms = topic_model.get_topic(topic_id) or []
            keywords = [
                term
                for term, _weight in topic_terms[:self.num_keywords]
            ]

            if not keywords:
                continue

            themes.append(
                ThemeResult(
                    theme_id=topic_id,
                    weight=None,
                    label=", ".join(keywords[:3]),
                    keywords=keywords,
                    method=self.method_name,
                    metadata={
                        "model": MODEL_NAME,
                        "language": "english",
                        "num_keywords": len(keywords),
                    },
                )
            )

        theme_ids = {
            theme.theme_id
            for theme in themes
        }

        assignments = []

        for valid_document_index, topic_id in enumerate(topics):
            if topic_id == -1:
                continue

            if topic_id not in theme_ids:
                continue

            original_document_index = original_indices[valid_document_index]

            topic_weight = None

            if probabilities is not None:
                try:
                    topic_position = topic_ids.index(topic_id)
                    topic_weight = float(probabilities[valid_document_index][topic_position])
                except Exception:
                    topic_weight = None

            assignments.append(
                {
                    "document_index": original_document_index,
                    "theme_id": topic_id,
                    "theme_weight": topic_weight,
                }
            )

        return themes, assignments