# This file defines the data contracts for NLP analysis results in the context of studies. 
# (i.e., defines the "shape" of the system)

from dataclasses import dataclass, field
from typing import Any

@dataclass
class SentimentResult:
    score: float
    label: str
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
# Example usage:
# sentiment = SentimentResult(
# 	score=-0.62,
# 	label="NEGATIVE",
# 	method="vader",
# 	metadata={
# 		"positive": 0.05,
# 		"neutral": 0.55,
# 		"negative": 0.40,
# 		"compound": -0.62,
# 	}
# )

@dataclass
class ThemeResult:
    theme_id: int
    label: str
    keywords: list[str]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

# Example usage:
# theme = ThemeResult(
#   theme_id=0,
# 	label="login frustration",
# 	keywords=["login", "password", "error", "reset", "access"],
# 	method="tfidf_nmf",
# 	metadata={
# 		"topic_weight": 0.74,
# 	}
# )

@dataclass
class EntryAnalysisResult:
    entry_id: int
    sentiment: SentimentResult | None = None
    theme: ThemeResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

# Example usage:
# entry_analysis = EntryAnalysisResult(
# 	entry_id=17,
# 	sentiment=SentimentResult(
# 		score=-0.62,
# 		label="NEGATIVE",
# 		method="vader",
# 		metadata={
# 			"positive": 0.05,
# 			"neutral": 0.55,
# 			"negative": 0.40,
# 			"compound": -0.62,
# 		}
# 	),
# 	theme=ThemeResult(
# 		theme_id=0,
# 		label="login frustration",
# 		keywords=["login", "password", "error", "reset", "access"],
# 		method="tfidf_nmf",
# 		metadata={
# 			"topic_weight": 0.74,
# 		}
# 	),
# 	metadata={
# 		"word_count": 42,
# 		"language": "en",
# 	}
# )

@dataclass
class StudyAnalysisResult:
    study_id: int

    # Sentiment / Opinion Analysis
    dominant_study_sentiment_label: str | None = None
    average_study_sentiment_score: float | None = None
    study_sentiment_distribution: dict[str, int] = field(default_factory=dict)
    
    # Theme / Topic Analysis
    dominant_study_theme_label: str | None = None
    average_study_theme_weight: float | None = None
    study_theme_distribution: dict[str, int] = field(default_factory=dict)

    # All entry results, for traceability
    entry_analysis_results: list[EntryAnalysisResult] = field(default_factory=list)
    
    total_entries: int = 0
    total_themes: int = 0

    methods: dict[str, str] = field(default_factory=dict)

    # Example of methods dict:
    # methods = {
    # 	"sentiment": "vader",
    # 	"theme": "tfidf_nmf",
    # }

    metadata: dict[str, Any] = field(default_factory=dict)

# Example usage:
# study_analysis = StudyAnalysisResult(
# 	study_id=5,
# 	dominant_study_sentiment_label="NEGATIVE",
# 	average_study_sentiment_score=-0.32,
# 	study_sentiment_distribution={
# 		"NEGATIVE": 2,
# 		"POSITIVE": 1,
# 	},
# 	dominant_study_theme_label="login frustration",
# 	average_study_theme_weight=0.77,
# 	study_theme_distribution={
# 		"login frustration": 2,
# 		"interface clarity": 1,
# 	},
# 	entry_analysis_results=[
# 		EntryAnalysisResult(
# 			entry_id=17,
# 			sentiment=SentimentResult(
# 				score=-0.62,
# 				label="NEGATIVE",
# 				method="vader",
# 				metadata={
# 					"compound": -0.62,
# 				}
# 			),
# 			theme=ThemeResult(
# 				theme_id=0,
# 				label="login frustration",
# 				keywords=["login", "password", "error", "reset", "access"],
# 				method="tfidf_nmf",
# 				metadata={
# 					"topic_weight": 0.74,
# 				}
# 			)
# 		),
# 		EntryAnalysisResult(
# 			entry_id=18,
# 			sentiment=SentimentResult(
# 				score=-0.45,
# 				label="NEGATIVE",
# 				method="vader",
# 				metadata={
# 					"compound": -0.45,
# 				}
# 			),
# 			theme=ThemeResult(
# 				theme_id=0,
# 				label="login frustration",
# 				keywords=["login", "password", "error", "reset", "access"],
# 				method="tfidf_nmf",
# 				metadata={
# 					"topic_weight": 0.80,
# 				}
# 			)
# 		),
# 		EntryAnalysisResult(
# 			entry_id=19,
# 			sentiment=SentimentResult(
# 				score=0.21,
# 				label="POSITIVE",
# 				method="vader",
# 				metadata={
# 					"compound": 0.21,
# 				}
# 			),
# 			theme=ThemeResult(
# 				theme_id=1,
# 				label="interface clarity",
# 				keywords=["clear", "simple", "button", "screen"],
# 				method="tfidf_nmf",
# 				metadata={
# 					"topic_weight": 0.69,
# 				}
# 			)
# 		),
# 	],
# 	total_entries=3,
# 	total_themes=2,
# 	methods={
# 		"sentiment": "vader",
# 		"theme": "tfidf_nmf",
# 	},
# 	metadata={
# 		"num_themes_requested": 5,
# 		"num_themes_generated": 2,
# 	}
# )

class BaseSentimentAnalyzer:
    method_name: str = 'base'

    def analyze(self, text: str) -> SentimentResult:
        raise NotImplementedError("Subclasses must implement this method")
    
class BaseThemeExtractor:
    method_name: str = 'base'

    def extract(self, documents: list[str]) -> tuple[list[ThemeResult], list[dict]]:
        raise NotImplementedError("Subclasses must implement this method")
    