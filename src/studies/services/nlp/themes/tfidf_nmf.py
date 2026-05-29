from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from studies.services.nlp.contracts import BaseThemeExtractor, ThemeResult

class TfidfNmfThemeExtractor (BaseThemeExtractor):
    method_name = "tfidf_nmf"

    def __init__(self, num_themes: int = 5, num_keywords: int = 8, max_features: int = 1000):
        self.num_themes = num_themes
        self.num_keywords = num_keywords
        self.max_features = max_features

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
        
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=self.max_features,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
            )
        
        matrix = vectorizer.fit_transform(valid_documents)

        actual_num_themes = min (
            self.num_themes,
            matrix.shape[0],
            matrix.shape[1],
        )

        if actual_num_themes < 1:
            return [], []
        
        model = NMF(
            n_components=actual_num_themes, 
            random_state=42,
            init='nndsvda',
            max_iter=500,
            )
        
        document_theme_matrix = model.fit_transform(matrix)
        feature_names = vectorizer.get_feature_names_out()

        themes = []

        for theme_index, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-self.num_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            theme_result = ThemeResult(
                theme_id=theme_index,
                label=", ".join(keywords[:3]),
                weight=float(topic[top_indices].mean()),
                keywords=keywords,
                method=self.method_name,
                metadata={
                    "num_keywords": len(keywords),
                }
            )
            themes.append(theme_result)

        assignments = []

        for valid_document_index, theme_weights in enumerate(document_theme_matrix):
            original_document_index = original_indices[valid_document_index]
            top_theme_index = int(theme_weights.argmax())
            top_theme_weight = float(theme_weights[top_theme_index])

            assignment = {
                "document_index": original_document_index,
                "theme_id": top_theme_index,
                "theme_weight": top_theme_weight
            }
            assignments.append(assignment)

        return themes, assignments