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
    title = models.CharField(max_length=255) 
    description = models.TextField(blank=False) 
    participant_instructions = models.TextField(blank=True) 

    # evaluator-facing
    goal = models.TextField(blank=True)
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
        verbose_name = "Study"
        verbose_name_plural = "Studies"

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
        verbose_name = "Study Membership"
        verbose_name_plural = "Study Memberships"

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


class DiaryEntrySentiment(models.TextChoices):
    VERY_NEGATIVE = "VERY_NEGATIVE", "Very negative"
    NEGATIVE = "NEGATIVE", "Negative"
    NEUTRAL = "NEUTRAL", "Neutral"
    POSITIVE = "POSITIVE", "Positive"
    VERY_POSITIVE = "VERY_POSITIVE", "Very positive"

class DiaryEntrySource(models.TextChoices):
    INTERNAL = "INTERNAL", "ChronicleUX submission"
    EXTERNAL = "EXTERNAL", "External submission imported into ChronicleUX"

class DiaryEntry(models.Model):

    # Entry SOURCE (Internal vs. External)
    source = models.CharField(
        max_length=20,
        choices=DiaryEntrySource.choices,
        default=DiaryEntrySource.INTERNAL,
    )

    # Entry Study
    study = models.ForeignKey(
        Study, 
        on_delete=models.CASCADE, 
        related_name="entries"
    )

    # Entry participant
    participant = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, # Not CASCADE b/c evaluators need the ability to import external data that may not include system users
        related_name="diary_entries",
        null=True,
        blank=True,
    )

    # These two attributes are needed for external entries imported into ChronicleUX
    participant_external_id = models.CharField(max_length=255, blank=True)
    participant_display_name = models.CharField(max_length=255, blank=True)
    participant_email = models.EmailField(blank=True)

    sentiment_self_report = models.CharField(
        max_length=20,
        choices=DiaryEntrySentiment.choices,
        help_text="How would you describe your overall experience sentiment for this entry?",
    )
    issue_encountered = models.BooleanField(
        help_text="Did you encounter any issue or friction during this experience?",
    )
    content = models.TextField()  # unstructured narrative
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["study", "participant", "created_at"]),
            models.Index(fields=["study","participant_external_id", "created_at"]),
        ]
        verbose_name = "Diary Entry"
        verbose_name_plural = "Diary Entries"

    def clean(self):
        
        # Identify if the participant is internal (i.e., exists in ChronicleUX)
        # Or it is external (i.e., imported entries from external source)
        has_internal_participant = self.participant is not None
        has_external_participant = bool (
            (self.participant_external_id or "").strip() or
            (self.participant_display_name or "").strip()
        )

        # All entries must have a participant identity (internal or external)
        if not has_internal_participant and not has_external_participant:
            raise ValidationError("Entries must have either a linked internal participant or an external participant identity.")

        # Internal entries must always have a linked ChronicleUX participant
        if self.source == DiaryEntrySource.INTERNAL and not has_internal_participant:
            raise ValidationError("Entries submitted through ChronicleUX must have a linked internal participant")

        # Even if entries have internal or external participant, the display name cannae be empty
        # Check if display name is empty... 
        if not (self.participant_display_name and self.participant_display_name.strip()):
            if self.participant:
                # ... then, if it is empty but the user is internal (i.e., self.participant exists), participant_display_name = username
                self.participant_display_name = self.participant.username
            else:
                # Raise error
                raise ValidationError (
                    {"participant_display_name" : "Attribute `participant_display_name` is required"}
                )

        # If there is a linked internal participant, they must belong to the study
        # And if the entry source is also internal, they must follow the temporal validation logic
        if self.participant:    
            # Check if the creator of the entry is a participant of the study
            is_participant = StudyMembership.objects.filter(
                study=self.study,
                user=self.participant,
                role=MembershipRole.PARTICIPANT,
            ).exists()

            # Raise error if user is not participant
            if not is_participant:
                raise ValidationError("User is not an enrolled participant in this study.")
            
            # If entry was created through ChronicleUX (i.e., entry is INTERNAL to the system) follow the temporal validation logic
            if self.source == DiaryEntrySource.INTERNAL:

                # Raise error if study is not collecting diary entries
                if not self.study.is_in_collection_window_now():
                    raise ValidationError("This study is not currently collecting diary entries.")

                # Enforce max 1 per frequency window (server timezone)
                # DAILY
                if self.study.entry_frequency == EntryFrequency.DAILY:
                    start = timezone.localtime(self.created_at).replace(
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
                # WEEKLY
                elif self.study.entry_frequency == EntryFrequency.WEEKLY:
                    now_local = timezone.localtime(self.created_at)
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
        participant_label = None

        if self.participant_id:
            participant_label = f"user={self.participant_id}"
        elif self.participant_external_id:
            participant_label = f"external_id={self.participant_external_id}"
        elif self.participant_display_name:
            participant_label = f"name={self.participant_display_name}"
        else:
            participant_label = "unknown_participant"
        return (
            f"Entry {self.pk} · study={self.study_id} · "
            f"{participant_label} · {self.created_at}"
        )


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
        verbose_name = "Prompt Response"
        verbose_name_plural = "Prompt Responses"

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