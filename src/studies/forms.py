from django import forms

from django.utils import timezone

from .models import (
    Study,
    CanonicalIssue,
    CanonicalTheme,
    DiaryEntry,
    DiaryEntryAnalysis,
    DiaryEntryAnalysisEvaluatorTheme,
    DiaryEntryAnalysisEvaluatorIssue,
)


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
                "class": "w-full",
            }),
            "goal": forms.TextInput(attrs={
                "class": "w-full",
            }),
            "description": forms.Textarea(attrs={
                "rows": 4,
                "class": "",
            }),
            "hypotheses": forms.Textarea(attrs={
                "rows": 4,
                "class": "",
            }),
            "status": forms.Select(attrs={
                "class": "",
            }),
            "entry_frequency": forms.Select(attrs={
                "class": "",
            }),
            "data_collection_start": forms.DateInput(attrs={
                "type": "date",
                "class": "",
            }),
            "data_collection_end": forms.DateInput(attrs={
                "type": "date",
                "class": "",
            }),
            "participant_instructions": forms.Textarea(attrs={
                "rows": 4,
                "class": "",
            }),
            "context": forms.Textarea(attrs={
                "rows": 4,
                "class": "",
            }),
            "tags": forms.TextInput(attrs={
                "class": "w-full",
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
                    "class": "",
                }
            ),
            "issue_encountered": forms.CheckboxInput(
                attrs={
                    "class": "form-checkbox",
                }
            ),
            "content": forms.Textarea(
                attrs={
                    "rows": 10,
                    "class": "",
                    "placeholder": "Write about your experience...",
                }
            ),
        }
        labels = {
            "sentiment_self_report": "Overall sentiment",
            "issue_encountered": "I encountered an issue",
            "content": "Diary entry",
        }

# CANONICAL THEME FORM
class CanonicalThemeForm(forms.ModelForm):

    aliases = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "",
                "rows": 3,
                "placeholder": "Optional aliases, separated by commas or line breaks.",
            }
        ),
        help_text="Alternative names or phrases that should map to this canonical theme.",
    )

    class Meta:
        model = CanonicalTheme
        fields = [
            "name",
            "description",
            "aliases",
            "source",
            "status",
            "examples",
            "is_active",
        ]
        widgets = {
            "name": forms.TextInput(
                attrs={
                    "class": "w-full",
                    "placeholder": "Example: Message organization",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "",
                    "rows": 4,
                    "placeholder": "Describe what this canonical theme means.",
                }
            ),
            "source": forms.Select(
                attrs={
                    "class": "",
                }
            ),
            "status": forms.Select(
                attrs={
                    "class": "",
                }
            ),
            "examples": forms.Textarea(
                attrs={
                    "class": "",
                    "rows": 4,
                    "placeholder": "Optional examples of entries, phrases, or situations that belong to this theme.",
                }
            ),
            "is_active": forms.CheckboxInput(
                attrs={
                    "class": "form-checkbox",
                }
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.instance and self.instance.pk and isinstance(self.instance.aliases, list):
            self.initial["aliases"] = ", ".join(self.instance.aliases)

    def clean_aliases(self):
        aliases_raw = self.cleaned_data.get("aliases", "")

        if not aliases_raw:
            return []

        aliases = []
        for value in aliases_raw.replace("\n", ",").split(","):
            alias = value.strip()
            if alias and alias not in aliases:
                aliases.append(alias)

        return aliases


# CANONICAL ISSUE FORM
class CanonicalIssueForm(forms.ModelForm):

    aliases = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "",
                "rows": 3,
                "placeholder": "Optional aliases, separated by commas or line breaks.",
            }
        ),
        help_text="Alternative names or phrases that should map to this canonical issue.",
    )

    class Meta:
        model = CanonicalIssue
        fields = [
            "name",
            "description",
            "aliases",
            "source",
            "status",
            "examples",
            "is_active",
        ]
        widgets = {
            "name": forms.TextInput(
                attrs={
                    "class": "w-full",
                    "placeholder": "Example: Performance",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "",
                    "rows": 4,
                    "placeholder": "Describe what this canonical issue means.",
                }
            ),
            "source": forms.Select(
                attrs={
                    "class": "",
                }
            ),
            "status": forms.Select(
                attrs={
                    "class": "",
                }
            ),
            "examples": forms.Textarea(
                attrs={
                    "class": "",
                    "rows": 4,
                    "placeholder": "Optional examples of entries, phrases, or situations that belong to this issue.",
                }
            ),
            "is_active": forms.CheckboxInput(
                attrs={
                    "class": "form-checkbox",
                }
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.instance and self.instance.pk and isinstance(self.instance.aliases, list):
            self.initial["aliases"] = ", ".join(self.instance.aliases)

    def clean_aliases(self):
        aliases_raw = self.cleaned_data.get("aliases", "")

        if not aliases_raw:
            return []

        aliases = []
        for value in aliases_raw.replace("\n", ",").split(","):
            alias = value.strip()
            if alias and alias not in aliases:
                aliases.append(alias)

        return aliases
    
class DiaryEntryManualEvaluationForm(forms.ModelForm):

    class Meta:
        model = DiaryEntryAnalysis
        fields = [
            "evaluator_sentiment_label",
            "evaluator_themes",
            "evaluator_issues",
            "evaluator_notes",
        ]

        widgets = {
            "evaluator_sentiment_label": forms.Select(
                attrs={
                    "class": "form-select",
                }
            ),
            "evaluator_themes": forms.CheckboxSelectMultiple(),
            "evaluator_issues": forms.CheckboxSelectMultiple(),
            "evaluator_notes": forms.Textarea(
                attrs={
                    "class": "form-textarea",
                    "rows": 3,
                    "placeholder": "Optional notes about this evaluation...",
                }
            ),
        }

        labels = {
            "evaluator_sentiment_label": "Evaluator sentiment",
            "evaluator_themes": "Evaluator themes",
            "evaluator_issues": "Evaluator issues",
            "evaluator_notes": "Evaluator notes",
        }

    def __init__(self, *args, evaluator=None, **kwargs):
        self.evaluator = evaluator
        super().__init__(*args, **kwargs)

        self.fields["evaluator_themes"].queryset = (
            CanonicalTheme.objects
            .filter(is_active=True)
            .order_by("name")
        )

        self.fields["evaluator_issues"].queryset = (
            CanonicalIssue.objects
            .filter(is_active=True)
            .order_by("name")
        )

        self.fields["evaluator_themes"].required = True
        self.fields["evaluator_issues"].required = False

    def save(self, commit=True):
        instance = super().save(commit=False)

        if self.evaluator:
            instance.evaluated_by = self.evaluator

        instance.evaluated_at = timezone.now()

        if commit:
            instance.save()

            evaluator_themes = self.cleaned_data.get("evaluator_themes") or []
            evaluator_issues = self.cleaned_data.get("evaluator_issues") or []

            DiaryEntryAnalysisEvaluatorTheme.objects.filter(
                diary_entry_analysis=instance
            ).delete()

            DiaryEntryAnalysisEvaluatorIssue.objects.filter(
                diary_entry_analysis=instance
            ).delete()

            for theme in evaluator_themes:
                DiaryEntryAnalysisEvaluatorTheme.objects.create(
                    diary_entry_analysis=instance,
                    canonical_theme=theme,
                    assigned_by=self.evaluator,
                )

            for issue in evaluator_issues:
                DiaryEntryAnalysisEvaluatorIssue.objects.create(
                    diary_entry_analysis=instance,
                    canonical_issue=issue,
                    assigned_by=self.evaluator,
                )

        return instance