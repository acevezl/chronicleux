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
)


class PromptInline(admin.TabularInline):
    model = Prompt
    extra = 1


class StudyMembershipInline(admin.TabularInline):
    model = StudyMembership
    extra = 1


@admin.register(Study)
class StudyAdmin(admin.ModelAdmin):
    list_display = ("title", "status", "owner", "data_collection_start", "data_collection_end")
    list_filter = ("status", "entry_frequency")
    inlines = [PromptInline, StudyMembershipInline]


@admin.register(DiaryEntry)
class DiaryEntryAdmin(admin.ModelAdmin):
    list_display = ("study", "participant", "created_at")
    list_filter = ("study",)


@admin.register(StudyAnalysisRun)
class StudyAnalysisRunAdmin(admin.ModelAdmin):
    list_display = ("id", "study", "status", "analysis_model", "analysis_version", "started_at", "completed_at", "dominant_sentiment_label", "dominant_sentiment_score", "dominant_theme_weight", "dominant_theme_label", "error_message")
    list_filter = ("status", "analysis_model", "analysis_version", "dominant_sentiment_label")
    search_fields = ("study__title",)


@admin.register(DiaryEntryAnalysis)
class DiaryEntryAnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "entry", "analyzed_at", "sentiment_label", "sentiment_score", "raw_sentiment_result", "theme_label", "theme_weight", "raw_theme_result", "issue_detected", "issues", "entry_summary", "methods", "metadata")
    list_filter = ("sentiment_label", "theme_label", "issues", "analyzed_at")
    search_fields = ("entry__content", "entry__participant_display_name", "entry__study__title")


admin.site.register(Prompt)
admin.site.register(StudyMembership)
admin.site.register(PromptResponse)


