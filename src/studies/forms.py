from django import forms
from .models import Study
from .models import DiaryEntry

class StudyForm(forms.ModelForm):
    class Meta:
        model = Study
        fields = [
            "title",
            "description",
            "goal",
            "hypotheses",
            "status",
            "entry_frequency",
            "data_collection_start",
            "data_collection_end",
            "participant_instructions",
            "context",
            "tags",
        ]

        help_texts = {
            "title": "A short, descriptive name for the diary study. The title should clearly communicate the focus of the research and help distinguish it from other studies (i.e., what the study is called).",
            "goal": "The main objective of the diary study. This should describe what the research aims to understand about user behavior, experiences, or attitudes over the course of the study (i.e., what your research wants to lear).",
            "description": "A brief overview of the study and its context. Describe what is being studied, why it matters, and any relevant background that helps evaluators and participants understand the purpose of the research (i.e., what the study is about).",
            "hypotheses": "A hypothesis is a testable prediction about participant behavior or experiences during the study. It helps guide what patterns you will look for when analyzing the diary entries (i.e., what you expect to happen). You may write 1-3 hypotheses.",
            "participant_instructions": "The guidance participants will see when completing their diary entries. Use this field to explain what participants should observe, reflect on, or report during each entry (i.e., what participants should do).",
            "context": "The situation or conditions in which participants should record their diary entries. Use this field to describe relevant factors such as environment, timing, activity, or device that may influence the experience being studied. (i.e., under what conditions entries should be recorded)",
            "tags": "Enter tags, separated by commas"
        }

        widgets = {
            "title": forms.TextInput(attrs={
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "goal": forms.TextInput(attrs={
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "description": forms.Textarea(attrs={
                "rows": 4,
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "hypotheses": forms.Textarea(attrs={
                "rows": 4,
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "status": forms.Select(attrs={
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "entry_frequency": forms.Select(attrs={
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "data_collection_start": forms.DateInput(attrs={
                "type": "date",
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "data_collection_end": forms.DateInput(attrs={
                "type": "date",
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "participant_instructions": forms.Textarea(attrs={
                "rows": 4,
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "context": forms.Textarea(attrs={
                "rows": 4,
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
            }),
            "tags": forms.TextInput(attrs={
                "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
                "placeholder": "e.g. user experience, workflow analysis, engagement",
                "data-role": "tags-input",
            })
        }

    def clean(self):
        cleaned_data = super().clean()
        start = cleaned_data.get("data_collection_start")
        end = cleaned_data.get("data_collection_end")

        if start and end and start > end:
            raise forms.ValidationError("The start date cannot be after the end date.")

        return cleaned_data

class DiaryEntryForm(forms.ModelForm):
    class Meta:
        model = DiaryEntry
        fields = ["sentiment_self_report", "issue_encountered", "content"]
        widgets = {
            "sentiment_self_report": forms.Select(
                attrs={
                    "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
                }
            ),
            "issue_encountered": forms.CheckboxInput(
                attrs={
                    "class": "mt-1 h-4 w-4 rounded border-gray-300 text-gray-700 focus:ring-gray-700",
                }
            ),
            "content": forms.Textarea(
                attrs={
                    "rows": 10,
                    "class": "text-gray-600 dark:text-gray-200 mt-1 px-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700",
                    "placeholder": "Write about your experience...",
                }
            ),
        }
        labels = {
            "sentiment_self_report": "Overall sentiment",
            "issue_encountered": "I encountered an issue",
            "content": "Diary entry",
        }