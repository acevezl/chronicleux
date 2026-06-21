from django.contrib import admin
from unfold.admin import ModelAdmin

from .models import (
    Study,
    StudyMembership,
    Prompt,
    DiaryEntry,
    PromptResponse,
    StudyAnalysisRun,
    DiaryEntryAnalysis,
    CanonicalTheme,
    CanonicalIssue,
    DiaryEntryAnalysisCanonicalTheme,
    DiaryEntryAnalysisCanonicalIssue,
    DiaryEntryAnalysisEvaluatorTheme,
    DiaryEntryAnalysisEvaluatorIssue,
    StudyAnalysisRunCanonicalTheme,
    StudyAnalysisRunCanonicalIssue,
)


class PromptInline(admin.TabularInline):
    model = Prompt
    extra = 1


class StudyMembershipInline(admin.TabularInline):
    model = StudyMembership
    extra = 1


class PromptResponseInline(admin.TabularInline):
    model = PromptResponse
    extra = 0


class DiaryEntryAnalysisCanonicalThemeInline(admin.TabularInline):
    model = DiaryEntryAnalysisCanonicalTheme
    extra = 0
    autocomplete_fields = ("canonical_theme", "assigned_by")


class DiaryEntryAnalysisCanonicalIssueInline(admin.TabularInline):
    model = DiaryEntryAnalysisCanonicalIssue
    extra = 0
    autocomplete_fields = ("canonical_issue", "assigned_by")


class DiaryEntryAnalysisEvaluatorThemeInline(admin.TabularInline):
	model = DiaryEntryAnalysisEvaluatorTheme
	extra = 0
	autocomplete_fields = ("canonical_theme", "assigned_by")


class DiaryEntryAnalysisEvaluatorIssueInline(admin.TabularInline):
	model = DiaryEntryAnalysisEvaluatorIssue
	extra = 0
	autocomplete_fields = ("canonical_issue", "assigned_by")


class StudyAnalysisRunCanonicalThemeInline(admin.TabularInline):
    model = StudyAnalysisRunCanonicalTheme
    extra = 0
    autocomplete_fields = ("canonical_theme",)


class StudyAnalysisRunCanonicalIssueInline(admin.TabularInline):
    model = StudyAnalysisRunCanonicalIssue
    extra = 0
    autocomplete_fields = ("canonical_issue",)


@admin.register(Study)
class StudyAdmin(ModelAdmin):
    list_display = (
        "id",
        "title",
        "status",
        "owner",
        "entry_frequency",
        "data_collection_start",
        "data_collection_end",
        "created_at",
    )
    list_filter = ("status", "entry_frequency", "created_at")
    search_fields = ("title", "description", "goal", "owner__username", "owner__email")
    autocomplete_fields = ("owner", "selected_study_run")
    inlines = [PromptInline, StudyMembershipInline]


@admin.register(StudyMembership)
class StudyMembershipAdmin(ModelAdmin):
    list_display = ("id", "study", "user", "role", "created_at")
    list_filter = ("role", "created_at")
    search_fields = ("study__title", "user__username", "user__email")
    autocomplete_fields = ("study", "user")


@admin.register(Prompt)
class PromptAdmin(ModelAdmin):
    list_display = ("id", "study", "order", "prompt_type", "is_required", "text", "created_at")
    list_filter = ("prompt_type", "is_required", "created_at")
    search_fields = ("study__title", "text")
    autocomplete_fields = ("study",)


@admin.register(DiaryEntry)
class DiaryEntryAdmin(ModelAdmin):
    list_display = (
        "id",
        "study",
        "participant",
        "participant_display_name",
        "source",
        "sentiment_self_report",
        "issue_encountered",
        "created_at",
    )
    list_filter = ("source", "study", "sentiment_self_report", "issue_encountered", "created_at")
    search_fields = (
        "content",
        "study__title",
        "participant__username",
        "participant__email",
        "participant_external_id",
        "participant_display_name",
        "participant_email",
    )
    autocomplete_fields = ("study", "participant", "selected_entry_run")
    inlines = [PromptResponseInline]


@admin.register(PromptResponse)
class PromptResponseAdmin(ModelAdmin):
    list_display = ("id", "diary_entry", "prompt", "likert_value", "text_value")
    list_filter = ("prompt__prompt_type",)
    search_fields = ("diary_entry__content", "prompt__text", "text_value")
    autocomplete_fields = ("diary_entry", "prompt")


@admin.register(StudyAnalysisRun)
class StudyAnalysisRunAdmin(ModelAdmin):
    list_display = (
        "id",
        "study",
        "status",
        "analysis_model",
        "analysis_version",
        "started_at",
        "completed_at",
        "total_entries",
        "total_themes",
        "dominant_sentiment_label",
        "dominant_sentiment_score",
    )
    list_filter = (
        "status",
        "analysis_model",
        "analysis_version",
        "dominant_sentiment_label",
        "started_at",
    )
    search_fields = ("study__title", "analysis_model", "analysis_version", "error_message")
    autocomplete_fields = ("study", "created_by")
    inlines = [
        StudyAnalysisRunCanonicalThemeInline,
        StudyAnalysisRunCanonicalIssueInline,
    ]


@admin.register(DiaryEntryAnalysis)
class DiaryEntryAnalysisAdmin(ModelAdmin):
    list_display = (
        "id",
        "run",
        "entry",
        "analyzed_at",
        "sentiment_label",
        "sentiment_score",
        "participant_sentiment_label",
        "participant_confusion_matrix_outcome",
        "evaluator_sentiment_label",
        "evaluator_confusion_matrix_outcome",
    )
    list_filter = (
        "sentiment_label",
        "participant_sentiment_label",
        "participant_confusion_matrix_outcome",
        "evaluator_sentiment_label",
        "evaluator_confusion_matrix_outcome",
        "analyzed_at",
    )
    search_fields = (
        "entry__content",
        "entry__participant_display_name",
        "entry__study__title",
        "entry_summary",
    )
    autocomplete_fields = ("run", "entry")
    inlines = [
        DiaryEntryAnalysisCanonicalThemeInline,
        DiaryEntryAnalysisCanonicalIssueInline,
        DiaryEntryAnalysisEvaluatorThemeInline,
        DiaryEntryAnalysisEvaluatorIssueInline,
    ]


@admin.register(CanonicalTheme)
class CanonicalThemeAdmin(ModelAdmin):
    list_display = (
        "id",
        "name",
        "source",
        "status",
        "is_active",
        "created_by",
        "created_at",
        "updated_at",
    )
    list_filter = ("source", "status", "is_active", "created_at")
    search_fields = ("name", "description", "examples")
    autocomplete_fields = ("created_by",)


@admin.register(CanonicalIssue)
class CanonicalIssueAdmin(ModelAdmin):
    list_display = (
        "id",
        "name",
        "source",
        "status",
        "is_active",
        "created_by",
        "created_at",
        "updated_at",
    )
    list_filter = ("source", "status", "is_active", "created_at")
    search_fields = ("name", "description", "examples")
    autocomplete_fields = ("created_by",)


@admin.register(DiaryEntryAnalysisCanonicalTheme)
class DiaryEntryAnalysisCanonicalThemeAdmin(ModelAdmin):
    list_display = (
        "id",
        "diary_entry_analysis",
        "canonical_theme",
        "confidence_score",
        "assigned_by",
        "assigned_at",
    )
    list_filter = ("canonical_theme", "assigned_at")
    search_fields = (
        "canonical_theme__name",
        "diary_entry_analysis__entry__content",
        "rationale",
    )
    autocomplete_fields = ("diary_entry_analysis", "canonical_theme", "assigned_by")


@admin.register(DiaryEntryAnalysisCanonicalIssue)
class DiaryEntryAnalysisCanonicalIssueAdmin(ModelAdmin):
    list_display = (
        "id",
        "diary_entry_analysis",
        "canonical_issue",
        "confidence_score",
        "assigned_by",
        "assigned_at",
    )
    list_filter = ("canonical_issue", "assigned_at")
    search_fields = (
        "canonical_issue__name",
        "diary_entry_analysis__entry__content",
        "rationale",
    )
    autocomplete_fields = ("diary_entry_analysis", "canonical_issue", "assigned_by")


@admin.register(StudyAnalysisRunCanonicalTheme)
class StudyAnalysisRunCanonicalThemeAdmin(ModelAdmin):
    list_display = (
        "id",
        "run",
        "canonical_theme",
        "entry_count",
        "average_confidence_score",
    )
    list_filter = ("canonical_theme",)
    search_fields = ("run__study__title", "canonical_theme__name")
    autocomplete_fields = ("run", "canonical_theme")


@admin.register(StudyAnalysisRunCanonicalIssue)
class StudyAnalysisRunCanonicalIssueAdmin(ModelAdmin):
    list_display = (
        "id",
        "run",
        "canonical_issue",
        "entry_count",
        "average_confidence_score",
    )
    list_filter = ("canonical_issue",)
    search_fields = ("run__study__title", "canonical_issue__name")
    autocomplete_fields = ("run", "canonical_issue")