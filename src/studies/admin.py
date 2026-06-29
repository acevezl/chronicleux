from django.contrib import admin
from unfold.admin import ModelAdmin

from .models import (
    Study,
    StudyMembership,
    Prompt,
    Entry,
    PromptResponse,
    StudyAnalysis,
    EntryAnalysis,
    CanonicalTheme,
    CanonicalIssue,
    EntryAnalysisTheme,
    EntryAnalysisIssue,
    EntryEvaluation,
    EntryEvaluationTheme,
    EntryEvaluationIssue,
    StudyEvaluationTheme,
    StudyEvaluationIssue,
    StudyAnalysisTheme,
    StudyAnalysisIssue,
    UXFramework,
    UXFrameworkCriterion,
    CanonicalIssueToFrameworkMapping,
    CanonicalThemeToFrameworkMapping,
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


class EntryAnalysisThemeInline(admin.TabularInline):
    model = EntryAnalysisTheme
    extra = 0
    autocomplete_fields = ("theme",)
    readonly_fields = ("assigned_at",)


class EntryAnalysisIssueInline(admin.TabularInline):
    model = EntryAnalysisIssue
    extra = 0
    autocomplete_fields = ("issue",)
    readonly_fields = ("assigned_at",)


class EntryEvaluationThemeInline(admin.TabularInline):
    model = EntryEvaluationTheme
    extra = 0
    autocomplete_fields = ("theme", "assigned_by")
    readonly_fields = ("assigned_at",)


class EntryEvaluationIssueInline(admin.TabularInline):
    model = EntryEvaluationIssue
    extra = 0
    autocomplete_fields = ("issue", "assigned_by")
    readonly_fields = ("assigned_at",)


class StudyEvaluationThemeInline(admin.TabularInline):
    model = StudyEvaluationTheme
    extra = 0
    autocomplete_fields = ("theme",)


class StudyEvaluationIssueInline(admin.TabularInline):
    model = StudyEvaluationIssue
    extra = 0
    autocomplete_fields = ("issue",)


class StudyAnalysisThemeInline(admin.TabularInline):
    model = StudyAnalysisTheme
    extra = 0
    autocomplete_fields = ("theme",)


class StudyAnalysisIssueInline(admin.TabularInline):
    model = StudyAnalysisIssue
    extra = 0
    autocomplete_fields = ("issue",)


class UXFrameworkCriterionInline(admin.TabularInline):
    model = UXFrameworkCriterion
    extra = 0


class CanonicalIssueToFrameworkMappingInline(admin.TabularInline):
    model = CanonicalIssueToFrameworkMapping
    extra = 0
    autocomplete_fields = ("criterion", "created_by", "approved_by")


class CanonicalThemeToFrameworkMappingInline(admin.TabularInline):
    model = CanonicalThemeToFrameworkMapping
    extra = 0
    autocomplete_fields = ("criterion", "created_by", "approved_by")


@admin.register(Study)
class StudyAdmin(ModelAdmin):
    list_display = (
        "id",
        "title",
        "status",
        "owner",
        "entry_frequency",
        "evaluator_theme_count",
        "evaluator_issue_count",
        "data_collection_start",
        "data_collection_end",
        "created_at",
    )
    list_filter = ("status", "entry_frequency", "created_at")
    search_fields = ("title", "description", "goal", "owner__username", "owner__email")
    autocomplete_fields = ("owner", "selected_study_run")
    inlines = [
        PromptInline,
        StudyMembershipInline,
        StudyEvaluationThemeInline,
        StudyEvaluationIssueInline,
    ]

    @admin.display(description="Evaluator themes")
    def evaluator_theme_count(self, obj):
        return obj.evaluator_themes.count()

    @admin.display(description="Evaluator issues")
    def evaluator_issue_count(self, obj):
        return obj.evaluator_issues.count()


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


@admin.register(Entry)
class EntryAdmin(ModelAdmin):
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
    list_display = ("id", "entry", "prompt", "likert_value", "text_value")
    list_filter = ("prompt__prompt_type",)
    search_fields = ("entry__content", "prompt__text", "text_value")
    autocomplete_fields = ("entry", "prompt")


@admin.register(StudyAnalysis)
class AnalysisAdmin(ModelAdmin):
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
        "total_issues",
        "average_sentiment_label",
        "average_sentiment_score",
        "dominant_sentiment_label",
        "dominant_sentiment_score",
    )
    list_filter = (
        "status",
        "analysis_model",
        "analysis_version",
        "average_sentiment_label",
        "dominant_sentiment_label",
        "started_at",
    )
    search_fields = ("study__title", "analysis_model", "analysis_version", "error_message")
    autocomplete_fields = ("study", "created_by", "dominant_theme")
    inlines = [
        StudyAnalysisThemeInline,
        StudyAnalysisIssueInline,
    ]


@admin.register(EntryAnalysis)
class EntryAnalysisAdmin(ModelAdmin):
    list_display = (
        "id",
        "run",
        "entry",
        "analyzed_at",
        "sentiment_label",
        "sentiment_score",
        "participant_sentiment_label",
        "participant_confusion_matrix_outcome",
        "evaluator_confusion_matrix_outcome",
    )
    list_filter = (
        "sentiment_label",
        "participant_sentiment_label",
        "participant_confusion_matrix_outcome",
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
        EntryAnalysisThemeInline,
        EntryAnalysisIssueInline,
    ]


@admin.register(EntryEvaluation)
class EntryEvaluationAdmin(ModelAdmin):
    list_display = (
        "id",
        "entry",
        "evaluator_sentiment_label",
        "evaluated_by",
        "created_at",
        "updated_at",
    )
    list_filter = (
        "evaluator_sentiment_label",
        "created_at",
        "updated_at",
    )
    search_fields = (
        "entry__content",
        "entry__participant_display_name",
        "entry__study__title",
        "evaluator_notes",
    )
    autocomplete_fields = ("entry", "evaluated_by")
    inlines = [
        EntryEvaluationThemeInline,
        EntryEvaluationIssueInline,
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
    search_fields = ("name", "description", "aliases", "examples")
    autocomplete_fields = ("created_by",)
    inlines = [CanonicalThemeToFrameworkMappingInline]


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
    search_fields = ("name", "description", "aliases", "examples")
    autocomplete_fields = ("created_by",)
    inlines = [CanonicalIssueToFrameworkMappingInline]


@admin.register(EntryAnalysisTheme)
class EntryAnalysisThemeAdmin(ModelAdmin):
    list_display = (
        "id",
        "entry_analysis",
        "theme",
        "confidence_score",
        "assigned_by_method",
        "assigned_at",
    )
    list_filter = ("theme", "assigned_at")
    search_fields = (
        "theme__name",
        "entry_analysis__entry__content",
        "rationale",
    )
    autocomplete_fields = ("entry_analysis", "theme")


@admin.register(EntryAnalysisIssue)
class EntryAnalysisIssueAdmin(ModelAdmin):
    list_display = (
        "id",
        "entry_analysis",
        "issue",
        "confidence_score",
        "assigned_by_method",
        "assigned_at",
    )
    list_filter = ("issue", "assigned_at")
    search_fields = (
        "issue__name",
        "entry_analysis__entry__content",
        "rationale",
    )
    autocomplete_fields = ("entry_analysis", "issue")


@admin.register(EntryEvaluationTheme)
class EntryEvaluationThemeAdmin(ModelAdmin):
    list_display = (
        "id",
        "entry_evaluation",
        "theme",
        "assigned_by",
        "assigned_at",
    )
    list_filter = ("theme", "assigned_at")
    search_fields = (
        "theme__name",
        "entry_evaluation__entry__content",
        "rationale",
    )
    autocomplete_fields = ("entry_evaluation", "theme", "assigned_by")


@admin.register(EntryEvaluationIssue)
class EntryEvaluationIssueAdmin(ModelAdmin):
    list_display = (
        "id",
        "entry_evaluation",
        "issue",
        "assigned_by",
        "assigned_at",
    )
    list_filter = ("issue", "assigned_at")
    search_fields = (
        "issue__name",
        "entry_evaluation__entry__content",
        "rationale",
    )
    autocomplete_fields = ("entry_evaluation", "issue", "assigned_by")


@admin.register(StudyEvaluationTheme)
class StudyEvaluationThemeAdmin(ModelAdmin):
    list_display = (
        "id",
        "study",
        "theme",
        "entry_count",
        "average_confidence_score",
    )
    list_filter = ("theme",)
    search_fields = ("study__title", "theme__name")
    autocomplete_fields = ("study", "theme")


@admin.register(StudyEvaluationIssue)
class StudyEvaluationIssueAdmin(ModelAdmin):
    list_display = (
        "id",
        "study",
        "issue",
        "entry_count",
        "average_confidence_score",
    )
    list_filter = ("issue",)
    search_fields = ("study__title", "issue__name")
    autocomplete_fields = ("study", "issue")


@admin.register(StudyAnalysisTheme)
class StudyAnalysisThemeAdmin(ModelAdmin):
    list_display = (
        "id",
        "run",
        "theme",
        "entry_count",
        "average_confidence_score",
        "average_sentiment_score",
    )
    list_filter = ("theme",)
    search_fields = ("run__study__title", "theme__name")
    autocomplete_fields = ("run", "theme")


@admin.register(StudyAnalysisIssue)
class StudyAnalysisIssueAdmin(ModelAdmin):
    list_display = (
        "id",
        "run",
        "issue",
        "entry_count",
        "average_confidence_score",
        "average_sentiment_score",
    )
    list_filter = ("issue",)
    search_fields = ("run__study__title", "issue__name")
    autocomplete_fields = ("run", "issue")


@admin.register(UXFramework)
class UXFrameworkAdmin(ModelAdmin):
    list_display = (
        "id",
        "name",
        "framework_type",
        "version",
        "source",
        "is_active",
        "created_by",
        "created_at",
        "updated_at",
    )
    list_filter = ("framework_type", "is_active", "created_at")
    search_fields = ("name", "description", "source", "version")
    autocomplete_fields = ("created_by",)
    inlines = [UXFrameworkCriterionInline]


@admin.register(UXFrameworkCriterion)
class UXFrameworkCriterionAdmin(ModelAdmin):
    list_display = (
        "id",
        "framework",
        "code",
        "name",
        "is_active",
        "created_at",
        "updated_at",
    )
    list_filter = ("framework", "is_active", "created_at")
    search_fields = (
        "framework__name",
        "code",
        "name",
        "description",
        "aliases",
        "examples",
        "recommendation_guidance",
    )
    autocomplete_fields = ("framework",)


@admin.register(CanonicalIssueToFrameworkMapping)
class CanonicalIssueToFrameworkMappingAdmin(ModelAdmin):
    list_display = (
        "id",
        "issue",
        "criterion",
        "method",
        "status",
        "score",
        "created_by",
        "approved_by",
        "approved_at",
        "created_at",
        "updated_at",
    )
    list_filter = ("method", "status", "created_at", "approved_at")
    search_fields = (
        "issue__name",
        "criterion__framework__name",
        "criterion__name",
        "rationale",
    )
    autocomplete_fields = ("issue", "criterion", "created_by", "approved_by")


@admin.register(CanonicalThemeToFrameworkMapping)
class CanonicalThemeToFrameworkMappingAdmin(ModelAdmin):
    list_display = (
        "id",
        "theme",
        "criterion",
        "method",
        "status",
        "score",
        "created_by",
        "approved_by",
        "approved_at",
        "created_at",
        "updated_at",
    )
    list_filter = ("method", "status", "created_at", "approved_at")
    search_fields = (
        "theme__name",
        "criterion__framework__name",
        "criterion__name",
        "rationale",
    )
    autocomplete_fields = ("theme", "criterion", "created_by", "approved_by")
