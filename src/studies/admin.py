from django.contrib import admin
from unfold.admin import ModelAdmin

from .models import (
    Study,
    StudyMembership,
    Prompt,
    DiaryEntry,
    PromptResponse,
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


admin.site.register(Prompt)
admin.site.register(StudyMembership)
admin.site.register(PromptResponse)

