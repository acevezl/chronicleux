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
    list_display = ("id", "study", "status", "analysis_model", "analysis_version", "started_at", "completed_at")
    list_filter = ("status", "analysis_model", "analysis_version")
    search_fields = ("study__title",)


@admin.register(DiaryEntryAnalysis)
class DiaryEntryAnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "entry", "run", "sentiment_category", "issue_detected", "analyzed_at")
    list_filter = ("sentiment_category", "issue_detected", "run__analysis_model", "run__analysis_version")
    search_fields = ("entry__content", "entry__participant_display_name", "entry__study__title")


admin.site.register(Prompt)
admin.site.register(StudyMembership)
admin.site.register(PromptResponse)


