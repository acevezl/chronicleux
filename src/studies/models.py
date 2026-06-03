from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q
from django.utils import timezone

User = settings.AUTH_USER_MODEL

# ----------------------- ENUMS ----------------------- #

# STUDY STATUS ENUM
class StudyStatus(models.TextChoices):
	PLANNING = "PLANNING", "Planning"
	COLLECTING = "COLLECTING", "Collecting diary entries"
	MACHINE_ANALYSIS = "MACHINE_ANALYSIS", "Machine analyzing study"
	HUMAN_ANALYSIS = "HUMAN_ANALYSIS", "Pending human analysis"
	COMPLETED = "COMPLETED", "Completed"


# ENTRY FREQUENCY ENUM
class EntryFrequency(models.TextChoices):
    DAILY = "DAILY", "Daily"
    WEEKLY = "WEEKLY", "Weekly"
    EVENT_BASED = "EVENT_BASED", "Event-based"
    FREEFORM = "FREEFORM", "Freeform"


# MEMBERSHIP ROLE ENUM
class MembershipRole(models.TextChoices):
    EVALUATOR = "EVALUATOR", "Evaluator"
    PARTICIPANT = "PARTICIPANT", "Participant"


# PROMPT TYPES ENUM
class PromptType(models.TextChoices):
    LIKERT_7 = "LIKERT_7", "Likert (7-point)"
    SHORT_TEXT = "SHORT_TEXT", "Short text"


# SENTIMENT CATEGORIES (I.E., SENTIMENT LABELS) 
# Used on both Diary Sentiment and Entry Sentiment
class SentimentCategory(models.TextChoices):
    VERY_NEGATIVE = "VERY_NEGATIVE", "Very negative"
    NEGATIVE = "NEGATIVE", "Negative"
    NEUTRAL = "NEUTRAL", "Neutral"
    POSITIVE = "POSITIVE", "Positive"
    VERY_POSITIVE = "VERY_POSITIVE", "Very positive"

# DIARY ENTRY SOURCE ENUM
class DiaryEntrySource(models.TextChoices):
    INTERNAL = "INTERNAL", "ChronicleUX submission"
    EXTERNAL = "EXTERNAL", "External submission imported into ChronicleUX"

# ANALYSIS RUN STATUS
class AnalysisRunStatus(models.TextChoices):
    PENDING = "PENDING", "Pending"
    RUNNING = "RUNNING", "Running"
    COMPLETED = "COMPLETED", "Completed"
    FAILED = "FAILED", "Failed"

# SENTIMENT SCORE THRESHOLDS
# Used at the Study Level to label average sentiment score
SENTIMENT_SCORE_THRESHOLDS = [
    (-1.0, -0.7, SentimentCategory.VERY_NEGATIVE),
    (-0.7, -0.2, SentimentCategory.NEGATIVE),
    (-0.2, 0.2, SentimentCategory.NEUTRAL),
    (0.2, 0.7, SentimentCategory.POSITIVE),
    (0.7, 1.0, SentimentCategory.VERY_POSITIVE),
]


# ----------------------- MODELS ----------------------- #

# STUDY MODEL
class Study(models.Model):

    title = models.CharField(max_length=255) 
    description = models.TextField(blank=False) 
    
    goal = models.TextField(blank=True)
    context = models.TextField(blank=True)
    hypotheses = models.TextField(blank=True)

    participant_instructions = models.TextField(blank=True) 
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

    # Selected Analysis Run
    selected_study_run = models.ForeignKey(
        "StudyAnalysisRun",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="preferred_by_studies",
        help_text="The analysis run selected as the preferred interpretation for this study.",
    )

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
        """Prompts become immutable once data collection starts."""
        return self.status in {
            StudyStatus.COLLECTING,
            StudyStatus.MACHINE_ANALYSIS,
            StudyStatus.HUMAN_ANALYSIS,
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
        return f"{self.title} [{self.get_status_display()}]"

# STUDY MEMBERSHIP MODEL (I.E. WHAT PARTICIPANTS BELONG TO STUDY WITH WHAT ROLE)
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
        # Owner must always be an evaluator in their study, cannot write entries
        if self.role == MembershipRole.PARTICIPANT and self.study.owner.pk == self.user.pk:
            raise ValidationError("Study owner cannot be a Participant in their own study.")

    def __str__(self) -> str:
        return f"{self.user} · {self.study} · {self.role}"

# PROMPT MODEL
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

# DIARY ENTRY
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

    # Default / Selected Machine Analysis
    selected_entry_run = models.ForeignKey(
        "DiaryEntryAnalysis",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
        help_text="The selected/default analysis for this diary entry.",
    )

    # These two attributes are needed for external entries imported into ChronicleUX
    participant_external_id = models.CharField(max_length=255, blank=True)
    participant_display_name = models.CharField(max_length=255, blank=True)
    participant_email = models.EmailField(blank=True)

    # This is the sentiment ground truth
    sentiment_self_report = models.CharField(
        max_length=20,
        choices=SentimentCategory.choices,
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

# PROMPT RESPONSE
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

# STUDY ANALYSIS RUN
# Results of the full study analysis (i.e., all entries)
class StudyAnalysisRun(models.Model):

    study = models.ForeignKey(
        Study, 
        on_delete=models.CASCADE, 
        related_name="analysis_runs"
    )

    status = models.CharField(
        max_length=20,
        choices=AnalysisRunStatus.choices,
        default=AnalysisRunStatus.PENDING,
    )

    # model and version
    analysis_model = models.CharField(max_length=100)
    analysis_version = models.CharField(max_length=50)

    # start and end times
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True) 
    
    # Study-level outputs

    # Sentiment / Opinion Analysis
    average_sentiment_label = models.CharField(max_length=20, choices=SentimentCategory.choices, null=True, blank=True)
    average_sentiment_score = models.FloatField(null=True, blank=True)
    dominant_sentiment_label = models.CharField(max_length=20, choices=SentimentCategory.choices, null=True, blank=True)
    dominant_sentiment_score = models.FloatField(null=True, blank=True)
    sentiment_distribution = models.JSONField(default=dict, blank=True)
    
    # Theme / Topic Analysis
    dominant_theme_label = models.CharField(max_length=255, blank=True, null=True)
    dominant_theme_weight = models.FloatField(null=True, blank=True)
    theme_distribution = models.JSONField(default=dict, blank=True)

    # Issues
    recurring_issues = models.JSONField(default=list, blank=True)
    
    # Evolution of Sentiment, Theme, and Issues over time
    evolution_over_time = models.JSONField(default=list, blank=True)

    # Top quotes
    top_representative_quotes = models.JSONField(default=list, blank=True)

    # All entry results, for traceability
    entry_analysis_results = models.JSONField(default=list, blank=True)
    
    total_entries = models.PositiveIntegerField(default=0)
    total_themes = models.PositiveIntegerField(default=0)

    methods = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    # error message (if analysis failed)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at"]
        verbose_name = "Study Analysis Run"
        verbose_name_plural = "Study Analysis Runs"

    def __str__(self) -> str:
        return f"StudyAnalysisRun {self.pk} · study={self.study_id} · {self.status}"
    

# DIARY ENTRY ANALYSIS
# Results of the analysis of each entry
class DiaryEntryAnalysis(models.Model):
    
    run = models.ForeignKey(
        StudyAnalysisRun,
        on_delete=models.CASCADE,
        related_name="entry_analyses",
    )

    entry = models.ForeignKey(
        DiaryEntry,
        on_delete=models.CASCADE,
        related_name="entry",
    )

    # Sentiment Analysis Outputs
    sentiment_score = models.FloatField(null=True, blank=True)
    sentiment_label = models.CharField(
        max_length=20,
        choices=SentimentCategory.choices,
        null=True,
        blank=True,
    )
    raw_sentiment_result = models.JSONField(default=list, blank=True)

    # Thematic Analysis Outputs
    theme_weight = models.FloatField(null=True, blank=True)
    theme_label = models.CharField(max_length=255, blank=True, null=True)
    raw_theme_result = models.JSONField(default=list, blank=True)

    # Issues Identified
    issue_detected = models.BooleanField(default=False, blank=False)
    issues = models.JSONField(default=list, blank=True)

    methods = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    entry_summary = models.TextField(blank=True)
    analyzed_at = models.DateTimeField(auto_now_add=True)
    raw_response = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-analyzed_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["run", "entry"],
                name="unique_entry_per_run_analysis",
            ),
        ]
        indexes = [
            models.Index(fields=["run"]),
            models.Index(fields=["entry"])
        ]
        verbose_name = "Diary Entry Analysis"
        verbose_name_plural = "Diary Entry Analyses"

    def __str__(self) -> str:
        return f"DiaryEntryAnalysis {self.pk} · entry={self.entry_id} · run={self.run_id}"


