from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q
from django.utils import timezone

User = settings.AUTH_USER_MODEL

class StudyStatus(models.TextChoices):
    PLANNING = "PLANNING", "Not started (Planning)"
    COLLECTING = "COLLECTING", "Collecting diary entries"
    ANALYZING = "ANALYZING", "Analyzing diary entries"
    COMPLETED = "COMPLETED", "Completed"

class EntryFrequency(models.TextChoices):
    DAILY = "DAILY", "Daily"
    WEEKLY = "WEEKLY", "Weekly"
    EVENT_BASED = "EVENT_BASED", "Event-based"
    FREEFORM = "FREEFORM", "Freeform"

class MembershipRole(models.TextChoices):
    EVALUATOR = "EVALUATOR", "Evaluator"
    PARTICIPANT = "PARTICIPANT", "Participant"

class PromptType(models.TextChoices):
    LIKERT_7 = "LIKERT_7", "Likert (7-point)"
    SHORT_TEXT = "SHORT_TEXT", "Short text"

class Study(models.Model):
    # participant-facing
    title = models.CharField(max_length=200) 
    description = models.TextField(blank=True)  
    participant_instructions = models.TextField(blank=True) 

    # evaluator-facing
    objective = models.TextField(blank=True)
    context = models.TextField(blank=True)
    hypotheses = models.TextField(blank=True)
    tags = models.CharField(max_length=300, blank=True)  # optional, comma-separated for PoC

    # owners are evaluators who create the study and manage it
    # other evaluators may evaluate and analyze the study but not change study design or membership
    owner = models.ForeignKey(
        User, on_delete=models.PROTECT, related_name="owned_studies"
    )

    status = models.CharField(
        max_length=20, choices=StudyStatus.choices, default=StudyStatus.PLANNING
    )

    data_collection_start = models.DateTimeField(null=True, blank=True)
    data_collection_end = models.DateTimeField(null=True, blank=True)

    entry_frequency = models.CharField(
        max_length=20, choices=EntryFrequency.choices, default=EntryFrequency.DAILY
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def clean(self):
        # Validate collection window coherence when both are set.
        if self.data_collection_start and self.data_collection_end:
            if self.data_collection_end <= self.data_collection_start:
                raise ValidationError(
                    {"data_collection_end": "Data collection end must be after start."}
                )

    # Prompts become immutable once collection starts, to ensure data integrity.
    @property
    def prompts_locked(self) -> bool:
        """Prompts become immutable once data collection starts (status COLLECTING or later)."""
        return self.status in {
            StudyStatus.COLLECTING,
            StudyStatus.ANALYZING,
            StudyStatus.COMPLETED,
        }

    # Data collection window is determined by the server timezone (not user timezone) to ensure consistency for all participants regardless of location.
    def is_in_collection_window_now(self) -> bool:
        """Uses server timezone (timezone.now())."""
        now = timezone.now()
        if self.status != StudyStatus.COLLECTING:
            return False
        if self.data_collection_start and now < self.data_collection_start:
            return False
        if self.data_collection_end and now > self.data_collection_end:
            return False
        return True

    def __str__(self) -> str:
        return f"{self.title} ({self.status})"


class StudyMembership(models.Model):
    study = models.ForeignKey(Study, on_delete=models.CASCADE, related_name="memberships")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="study_memberships")
    role = models.CharField(max_length=20, choices=MembershipRole.choices)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            # One membership per user per study -> ensures no evaluator+participant dual role.
            models.UniqueConstraint(fields=["study", "user"], name="unique_user_per_study"),
        ]
        indexes = [
            models.Index(fields=["user", "role"]),
            models.Index(fields=["study", "role"]),
        ]

    def clean(self):
        # Owner must always be an evaluator in their study.
        if self.role == MembershipRole.PARTICIPANT and self.study.owner.pk == self.user.pk:
            raise ValidationError("Study owner cannot be a Participant in their own study.")

    def __str__(self) -> str:
        return f"{self.user} · {self.study} · {self.role}"


class Prompt(models.Model):
    study = models.ForeignKey(Study, on_delete=models.CASCADE, related_name="prompts")
    text = models.CharField(max_length=300)
    prompt_type = models.CharField(max_length=20, choices=PromptType.choices)
    is_required = models.BooleanField(default=True)
    order = models.PositiveIntegerField(default=1)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order", "id"]
        constraints = [
            models.CheckConstraint(condition=Q(order__gte=1), name="prompt_order_gte_1"),
        ]

    def clean(self):
        # Freeze prompts once collection starts.
        if self.pk and self.study.prompts_locked:
            raise ValidationError("Prompts are locked once data collection has started.")

    def save(self, *args, **kwargs):
        # Enforce lock at save time too (clean() can be bypassed).
        if self.pk:
            # if updating existing prompt, reload study status from DB for safety
            current = Prompt.objects.select_related("study").get(pk=self.pk)
            if current.study.prompts_locked:
                raise ValidationError("Prompts are locked once data collection has started.")
        else:
            if self.study.prompts_locked:
                raise ValidationError("Prompts are locked once data collection has started.")
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"[{self.study.pk}] {self.order}. {self.text}"


class DiaryEntry(models.Model):
    study = models.ForeignKey(Study, on_delete=models.CASCADE, related_name="entries")
    participant = models.ForeignKey(User, on_delete=models.CASCADE, related_name="diary_entries")

    content = models.TextField()  # unstructured narrative
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["study", "participant", "created_at"]),
        ]

    def clean(self):
        # Must be in collection window
        if not self.study.is_in_collection_window_now():
            raise ValidationError("This study is not currently collecting diary entries.")

        # Participant must be enrolled as PARTICIPANT
        is_participant = StudyMembership.objects.filter(
            study=self.study, user=self.participant, role=MembershipRole.PARTICIPANT
        ).exists()
        if not is_participant:
            raise ValidationError("User is not an enrolled participant in this study.")

        # Enforce max 1 per frequency window (server timezone)
        if self.study.entry_frequency == EntryFrequency.DAILY:
            start = timezone.localtime(timezone.now()).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = start + timedelta(days=1)
            exists = DiaryEntry.objects.filter(
                study=self.study,
                participant=self.participant,
                created_at__gte=start,
                created_at__lt=end,
            )
            if self.pk:
                exists = exists.exclude(pk=self.pk)
            if exists.exists():
                raise ValidationError("You have already submitted an entry for today.")

        elif self.study.entry_frequency == EntryFrequency.WEEKLY:
            now_local = timezone.localtime(timezone.now())
            iso_year, iso_week, _ = now_local.isocalendar()
            # Get Monday 00:00 of ISO week
            monday = now_local - timedelta(days=now_local.weekday())
            week_start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
            week_end = week_start + timedelta(days=7)
            exists = DiaryEntry.objects.filter(
                study=self.study,
                participant=self.participant,
                created_at__gte=week_start,
                created_at__lt=week_end,
            )
            if self.pk:
                exists = exists.exclude(pk=self.pk)
            if exists.exists():
                raise ValidationError("You have already submitted an entry for this week.")

        # EVENT_BASED and FREEFORM are unlimited.

    def __str__(self) -> str:
        return f"Entry {self.pk} · {self.study.pk} · {self.participant.pk} · {self.created_at}"


class PromptResponse(models.Model):
    diary_entry = models.ForeignKey(DiaryEntry, on_delete=models.CASCADE, related_name="prompt_responses")
    prompt = models.ForeignKey(Prompt, on_delete=models.PROTECT, related_name="responses")

    # Only one of these should be set based on prompt_type
    likert_value = models.PositiveSmallIntegerField(null=True, blank=True)
    text_value = models.TextField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["diary_entry", "prompt"], name="unique_prompt_per_entry"),
        ]

    def clean(self):
        if self.prompt.study.pk != self.diary_entry.study.pk:
            raise ValidationError("Prompt does not belong to the same study as the diary entry.")

        if self.prompt.prompt_type == PromptType.LIKERT_7:
            if self.likert_value is None:
                raise ValidationError({"likert_value": "Likert value is required for this prompt."})
            if not (1 <= self.likert_value <= 7):
                raise ValidationError({"likert_value": "Likert value must be between 1 and 7."})
            # Ensure text_value not used
            self.text_value = None

        elif self.prompt.prompt_type == PromptType.SHORT_TEXT:
            if self.prompt.is_required and not (self.text_value and self.text_value.strip()):
                raise ValidationError({"text_value": "Text response is required for this prompt."})
            # Ensure likert_value not used
            self.likert_value = None

    def __str__(self) -> str:
        return f"Response · entry={self.diary_entry.pk} · prompt={self.prompt.pk}"