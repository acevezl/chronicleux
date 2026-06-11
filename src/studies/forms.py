from django import forms
from .models import Study, DiaryEntry, CanonicalIssue, CanonicalTheme

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