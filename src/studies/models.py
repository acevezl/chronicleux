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
	COLLECTING = "COLLECTING", "Collecting entries"
	ANALYZING = "ANALYZING", "Analyzing study"
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


# SENTIMENT CATEGORIES (I.E., SENTIMENT LABELS) ENUM
# Used on both Diary Sentiment and Entry Sentiment
class SentimentCategory(models.TextChoices):
	VERY_NEGATIVE = "VERY_NEGATIVE", "Very negative"
	NEGATIVE = "NEGATIVE", "Negative"
	NEUTRAL = "NEUTRAL", "Neutral"
	POSITIVE = "POSITIVE", "Positive"
	VERY_POSITIVE = "VERY_POSITIVE", "Very positive"


# SENTIMENT CATEGORIES (I.E., SENTIMENT LABELS) FOR BERT ENUM
# Because BERT's gonna BERT... (i.e., it is binary)
class BinarySentimentCategory(models.TextChoices):
	NEGATIVE = "NEGATIVE", "Negative"
	NOT_NEGATIVE = "NOT_NEGATIVE", "Not Negative"

# SENTIMENT CHOICES ENUM ARRAY
# Used anywhere a stored sentiment label may come from either:
# - a 5-level sentiment model
# - a binary BERT model
SENTIMENT_CHOICES = [
	*SentimentCategory.choices,
	(BinarySentimentCategory.NOT_NEGATIVE, BinarySentimentCategory.NOT_NEGATIVE.label),
]

# ENTRY SOURCE ENUM
class EntrySource(models.TextChoices):
	INTERNAL = "INTERNAL", "ChronicleUX submission"
	EXTERNAL = "EXTERNAL", "External submission imported into ChronicleUX"


# MACHINE ANALYSIS STATUS ENUM
class AnalysisStatus(models.TextChoices):
	QUEUED = "QUEUED", "Queued"
	RUNNING = "RUNNING", "Running"
	COMPLETED = "COMPLETED", "Completed"
	FAILED = "FAILED", "Failed"


# CONFUSION MATRIX OUTCOME ENUM
class ConfusionMatrixOutcome(models.TextChoices):
	TRUE_POSITIVE = "TRUE_POSITIVE", "True Positive"
	FALSE_POSITIVE = "FALSE_POSITIVE", "False Positive"
	TRUE_NEGATIVE = "TRUE_NEGATIVE", "True Negative"
	FALSE_NEGATIVE = "FALSE_NEGATIVE", "False Negative"
	NOT_AVAILABLE = "NOT_AVAILABLE", "Not Available"


# THEME / ISSUE SOURCE ENUM
class ThemeAndIssueSource(models.TextChoices):
	EVALUATOR = "EVALUATOR", "Evaluator"
	NLP = "NLP", "NLP Model"
	LLM = "LLM", "LLM Model"

# THEME / ISSUE STATUS ENUM
class ThemeAndIssueStatus(models.TextChoices):
	SUGGESTED = "SUGGESTED", "Suggested"
	APPROVED = "APPROVED", "Approved"
	REJECTED = "REJECTED", "Rejected"


# FRAMEWORK SOURCE / TYPE ENUM
class UXFrameworkType(models.TextChoices):
	NIELSEN = "NIELSEN", "Nielsen"
	ISO = "ISO", "ISO"
	WCAG = "WCAG", "WCAG"
	CUSTOM = "CUSTOM", "Custom"


# FRAMEWORK MAPPING METHOD ENUM
class FrameworkMappingMethod(models.TextChoices):
	MANUAL = "MANUAL", "Manual"
	TFIDF = "TFIDF", "TF-IDF similarity"
	EMBEDDING = "EMBEDDING", "Embedding similarity"


# FRAMEWORK MAPPING STATUS ENUM
class FrameworkMappingStatus(models.TextChoices):
	SUGGESTED = "SUGGESTED", "Suggested"
	APPROVED = "APPROVED", "Approved"
	REJECTED = "REJECTED", "Rejected"


# SENTIMENT SCORE THRESHOLDS ARRAYS
# Used at the Study Level to label average sentiment score
SENTIMENT_SCORE_THRESHOLDS = [
	(-1.0, -0.7, SentimentCategory.VERY_NEGATIVE),
	(-0.7, -0.2, SentimentCategory.NEGATIVE),
	(-0.2, 0.2, SentimentCategory.NEUTRAL),
	(0.2, 0.7, SentimentCategory.POSITIVE),
	(0.7, 1.0, SentimentCategory.VERY_POSITIVE),
]
# These one's are for binary methods, like BERT
BINARY_SENTIMENT_SCORE_THRESHOLDS = [
	(-1.0, 0.0, BinarySentimentCategory.NEGATIVE),
	(0.0, 1.0, BinarySentimentCategory.NOT_NEGATIVE),
]

# ----------------------- MODELS ----------------------- #

# STUDY MODEL
class Study(models.Model):

	# Title of the study
	title = models.CharField(max_length=255) 
	# Description of the study
	description = models.TextField(blank=False) 
	# Goal of study
	goal = models.TextField(blank=True)
	# Context of study
	context = models.TextField(blank=True)
	# Hypotheses of study
	hypotheses = models.TextField(blank=True) # comma-separated for PoC, future should be a list
	# Participant instructions
	participant_instructions = models.TextField(blank=True) 
	# Tags for study classification
	tags = models.CharField(max_length=300, blank=True)  # optional, comma-separated for PoC.

	# Study owner
	# Owners are evaluators by default
	# other evaluators may evaluate and analyze the study but cannot change study design or membership
	owner = models.ForeignKey(
		User, on_delete=models.PROTECT, related_name="owned_studies"
	)

	# Study status
	status = models.CharField(
		max_length=20, choices=StudyStatus.choices, default=StudyStatus.PLANNING
	)

	# Data collection period, start and end
	data_collection_start = models.DateTimeField(null=True, blank=True)
	data_collection_end = models.DateTimeField(null=True, blank=True)

	# Data entry frequency
	entry_frequency = models.CharField(
		max_length=20, choices=EntryFrequency.choices, default=EntryFrequency.DAILY
	)

	# Creation and update dates
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	# Selected Machine Analysis
	# The ultimately-selected machine analysis run to complete the study
	selected_study_run = models.ForeignKey(
		"StudyAnalysis",
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="preferred_by_studies",
		help_text="The analysis run selected as the preferred interpretation for this study.",
	)

	# Participant-Self reported Sentiment
	participant_reported_average_sentiment_score = models.FloatField(null=True, blank=True)
	participant_reported_average_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	participant_reported_dominant_sentiment_score = models.FloatField(null=True, blank=True)
	participant_reported_dominant_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	participant_sentiment_distribution = models.JSONField(default=dict, blank=True)

	# Evaluator Analysis
	# Consolidation of all EntryEvaluations by Evaluators
	evaluator_average_sentiment_score = models.FloatField(null=True, blank=True)
	evaluator_average_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	evaluator_dominant_sentiment_score = models.FloatField(null=True, blank=True)
	evaluator_dominant_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	evaluator_sentiment_distribution = models.JSONField(default=dict, blank=True)

	# evaluated_themes.evaluated_studies.all()
	evaluator_themes = models.ManyToManyField(
		"CanonicalTheme",
		through="StudyEvaluationTheme",
		related_name="evaluated_in_studies",
		blank=True,
	)

	# evaluated_issues.evaluated_studies.all()
	evaluator_issues = models.ManyToManyField(
		"CanonicalIssue",
		through="StudyEvaluationIssue",
		related_name="evaluated_in_studies",
		blank=True,
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
			StudyStatus.ANALYZING,
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
	

# ENTRY
class Entry(models.Model):

	# Entry Study
	study = models.ForeignKey(
		Study, 
		on_delete=models.CASCADE, 
		related_name="entries"
	)

	# Entry SOURCE (Internal vs. External)
	source = models.CharField(
		max_length=20,
		choices=EntrySource.choices,
		default=EntrySource.INTERNAL,
	)

	# Entry participant (i.e., author of the entry)
	participant = models.ForeignKey(
		User, 
		on_delete=models.SET_NULL, # Not CASCADE b/c evaluators need the ability to import external data that may not include system users
		related_name="entries",
		null=True,
		blank=True,
	)

	# Default / Selected Machine Analysis
	# The ultimately-selected Machine Analysis run for this entry (all entries have to have the same overarching Study Analysis Run)
	selected_entry_run = models.ForeignKey(
		"EntryAnalysis",
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="+",
		help_text="The selected/default analysis for this entry.",
	)

	# When participant is external, we need an id, display name and email (in the case of imported entries)
	participant_external_id = models.CharField(max_length=255, blank=True)
	participant_display_name = models.CharField(max_length=255, blank=True)
	participant_email = models.EmailField(blank=True)

	# Sentiment self-reported by participant (for Reference calculations)
	sentiment_self_report = models.CharField(
		max_length=20,
		choices=SentimentCategory.choices,
		help_text="How would you describe your overall experience sentiment for this entry?",
	)

	# Bool that indicates if issue was encountered.
	# The issue is described within the body of the entry.
	issue_encountered = models.BooleanField(
		help_text="Did you encounter any issue or friction during this experience?",
	)

	# Entry content
	content = models.TextField()  # unstructured narrative
	# Creation date
	created_at = models.DateTimeField(default=timezone.now)

	class Meta:
		ordering = ["-created_at"]
		indexes = [
			models.Index(fields=["study", "participant", "created_at"]),
			models.Index(fields=["study","participant_external_id", "created_at"]),
		]
		verbose_name = "Entry"
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
		if self.source == EntrySource.INTERNAL and not has_internal_participant:
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
			if self.source == EntrySource.INTERNAL:

				# Raise error if study is not collecting entries
				if not self.study.is_in_collection_window_now():
					raise ValidationError("This study is not currently collecting entries.")

				# Enforce max 1 per frequency window (server timezone)
				# DAILY
				if self.study.entry_frequency == EntryFrequency.DAILY:
					start = timezone.localtime(self.created_at).replace(
						hour=0, minute=0, second=0, microsecond=0
					)
					end = start + timedelta(days=1)
					exists = Entry.objects.filter(
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
					exists = Entry.objects.filter(
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


# PROMPT MODEL
# This is an entry prompt (i.e, question that can be added to the entry)
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
	
	
# PROMPT RESPONSE
# This is the response per entry to the prompt
class PromptResponse(models.Model):
	entry = models.ForeignKey(Entry, on_delete=models.CASCADE, related_name="prompt_responses")
	prompt = models.ForeignKey(Prompt, on_delete=models.PROTECT, related_name="responses")

	# Only one of these should be set based on prompt_type
	likert_value = models.PositiveSmallIntegerField(null=True, blank=True)
	text_value = models.TextField(null=True, blank=True)

	class Meta:
		constraints = [
			models.UniqueConstraint(fields=["entry", "prompt"], name="unique_prompt_per_entry"),
		]
		verbose_name = "Prompt Response"
		verbose_name_plural = "Prompt Responses"

	def clean(self):
		if self.prompt.study.pk != self.entry.study.pk:
			raise ValidationError("Prompt does not belong to the same study as the entry.")

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
		return f"Response · entry={self.entry.pk} · prompt={self.prompt.pk}"
	

# STUDY ANALYSIS RUN
# Full-study analysis run
class StudyAnalysis(models.Model):

	# Study
	study = models.ForeignKey(
		Study, 
		on_delete=models.CASCADE, 
		related_name="analysis_runs"
	)

	# Status
	status = models.CharField(
		max_length=20,
		choices=AnalysisStatus.choices,
		default=AnalysisStatus.QUEUED,
	)

	# Model and version
	analysis_model = models.CharField(max_length=100)
	analysis_version = models.CharField(max_length=50)

	# Start and end times
	started_at = models.DateTimeField(auto_now_add=True)
	completed_at = models.DateTimeField(null=True, blank=True) 

	# Study created by [user]
	created_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="created_runs",
		help_text="User who created this analysis run"
	)
	
	# Study-level outputs

	# SENTIMENT
	# Detected average sentiment label and score
	average_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	average_sentiment_score = models.FloatField(null=True, blank=True)
	# Detected dominant sentiment label and score
	dominant_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	dominant_sentiment_score = models.FloatField(null=True, blank=True)
	# Detected sentiment distribution
	sentiment_distribution = models.JSONField(default=dict, blank=True)

	# THEMES
	# Themes identified in this run
	# This also works as theme distribution
	themes = models.ManyToManyField(
		"CanonicalTheme",
		through="StudyAnalysisTheme",
		related_name="analysis_runs",
		blank=True,
	)

	# Dominant Theme identified in this run
	# To be deprecated #fixlater
	dominant_theme = models.ForeignKey(
		"CanonicalTheme",
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="dominant_in_analysis_runs",
	)

	# ISSUES
	# Issues identified in this run
	# This also works as issue distribution
	issues = models.ManyToManyField(
		"CanonicalIssue",
		through="StudyAnalysisIssue",
		related_name="analysis_runs",
		blank=True,
	)
	
	# Evolution of Sentiment, Theme, and Issues over time
	evolution_over_time = models.JSONField(default=list, blank=True)

	# Top quotes
	top_representative_quotes = models.JSONField(default=list, blank=True)

	# All entry results, for traceability
	# entry_analysis_results = models.JSONField(default=list, blank=True)
	
	# Total entries analized, themes extracted, and issues detected
	total_entries = models.PositiveIntegerField(default=0)
	total_themes = models.PositiveIntegerField(default=0)
	total_issues = models.PositiveIntegerField(default=0)

	# methods and metadata
	methods = models.JSONField(default=dict, blank=True)
	metadata = models.JSONField(default=dict, blank=True)

	# error message (if analysis failed)
	error_message = models.TextField(blank=True)

	# Study-level metrics

	# Vs. Participant Reference
	participant_average_sentiment_score = models.FloatField(null=True, blank=True)
	participant_average_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	participant_dominant_sentiment_score = models.FloatField(null=True, blank=True)
	participant_dominant_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	participant_sentiment_distribution = models.JSONField(default=dict, blank=True)

	participant_abs_distance_average_sentiment = models.IntegerField(null=True, blank=True)
	participant_abs_distance_dominant_sentiment = models.IntegerField(null=True, blank=True)
	
	participant_sentiment_true_positives = models.PositiveIntegerField(default=0)
	participant_sentiment_false_positives = models.PositiveIntegerField(default=0)
	participant_sentiment_true_negatives = models.PositiveIntegerField(default=0)
	participant_sentiment_false_negatives = models.PositiveIntegerField(default=0)

	run_accuracy_v_participant = models.FloatField(null=True, blank=True)
	run_precision_v_participant = models.FloatField(null=True, blank=True)
	run_recall_v_participant = models.FloatField(null=True, blank=True)
	run_f1_v_participant = models.FloatField(null=True, blank=True)

	participant_sentiment_metrics_by_category = models.JSONField(
		default=dict,
		blank=True,
		help_text=(
			"One-vs-rest TP, FP, TN, FN, accuracy, precision, recall, and F1 "
			"per sentiment category against participant self-reported sentiment."
		),
	)

	# Vs. Evaluator Reference
	evaluator_average_sentiment_score = models.FloatField(null=True, blank=True)
	evaluator_average_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	evaluator_dominant_sentiment_score = models.FloatField(null=True, blank=True)
	evaluator_dominant_sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, null=True, blank=True)
	evaluator_sentiment_distribution = models.JSONField(default=dict, blank=True)

	evaluator_abs_distance_average_sentiment = models.IntegerField(null=True, blank=True)
	evaluator_abs_distance_dominant_sentiment = models.IntegerField(null=True, blank=True)

	evaluator_sentiment_true_positives = models.PositiveIntegerField(default=0)
	evaluator_sentiment_false_positives = models.PositiveIntegerField(default=0)
	evaluator_sentiment_true_negatives = models.PositiveIntegerField(default=0)
	evaluator_sentiment_false_negatives = models.PositiveIntegerField(default=0)

	run_accuracy_v_evaluator = models.FloatField(null=True, blank=True)
	run_precision_v_evaluator = models.FloatField(null=True, blank=True)
	run_recall_v_evaluator = models.FloatField(null=True, blank=True)
	run_f1_v_evaluator = models.FloatField(null=True, blank=True)

	evaluator_sentiment_metrics_by_category = models.JSONField(
		default=dict,
		blank=True,
		help_text=(
			"One-vs-rest TP, FP, TN, FN, accuracy, precision, recall, and F1 "
			"per sentiment category against evaluator-reviewed sentiment."
		),
	)

	class Meta:
		ordering = ["-started_at"]
		verbose_name = "Study Analysis"
		verbose_name_plural = "Study Analyses"

	def __str__(self) -> str:
		return f"StudyAnalysis {self.pk} · study={self.study_id} · {self.status}"
	

# ENTRY ANALYSIS
# Results of the machine analysis of each entry
class EntryAnalysis(models.Model):
	
	# run.entry_analyses.all() to get all entry analysis for a run
	run = models.ForeignKey(
		StudyAnalysis,
		on_delete=models.CASCADE,
		related_name="entry_analyses",
	)

	# entry.entry_analyses.all() to get all entry analysis for an entry
	entry = models.ForeignKey(
		Entry,
		on_delete=models.CASCADE,
		related_name="entry_analyses",
	)

	# SENTIMENT ANALYSIS
	sentiment_score = models.FloatField(null=True, blank=True)
	sentiment_label = models.CharField(
		max_length=20,
		choices=SENTIMENT_CHOICES,
		null=True,
		blank=True,
	)
	raw_sentiment_result = models.JSONField(default=list, blank=True)

	# THEME EXTRACTION
	# Themes detected (extracted) in this entry
	themes = models.ManyToManyField(
		"CanonicalTheme",
		through="EntryAnalysisTheme",
		related_name="entry_analyses",
		blank=True,
	)

	# ISSUE DETECTION
	# Issues detected in this entry
	issues = models.ManyToManyField(
		"CanonicalIssue",
		through="EntryAnalysisIssue",
		related_name="entry_analyses",
		blank=True,
	)

	# METRICS METADATA
	# Participant Reference
	participant_sentiment_label = models.CharField(
		max_length=20,
		choices=SENTIMENT_CHOICES,
		null=True,
		blank=True,
	)
	participant_confusion_matrix_outcome = models.CharField(
		max_length=32,
		choices=ConfusionMatrixOutcome.choices,
		default=ConfusionMatrixOutcome.NOT_AVAILABLE
	)
	
	# HUMAN EVALUATION MOVED TO SEPARATE CLASS 
	# Linked now directly to the entry, not the entry analysis
	# b/c entries can have many machine analysis and many human evaluators
	
	# Evaluator Confusion Matrix belongs here b/c each analysis is evaluated against the human evaluation(s)
	evaluator_confusion_matrix_outcome = models.CharField(
		max_length=32,
		choices=ConfusionMatrixOutcome.choices,
		default=ConfusionMatrixOutcome.NOT_AVAILABLE
	)

	@property
	def is_manually_evaluated(self):
		return self.entry.entry_evaluations.filter(
			evaluator_sentiment_label__isnull=False,
		).exists()

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
		verbose_name = "Entry Analysis"
		verbose_name_plural = "Entry Analyses"

	def __str__(self) -> str:
		return f"EntryAnalysis {self.pk} · entry={self.entry_id} · run={self.run_id}"
	

# ENTRY EVALUATION
# Results of the human evaluator assessment of an entry.
class EntryEvaluation(models.Model):

	entry = models.ForeignKey(
		Entry,
		on_delete=models.CASCADE,
		related_name="entry_evaluations",
	)

	evaluator_sentiment_label = models.CharField(
		max_length=20,
		choices=SENTIMENT_CHOICES,
		null=True,
		blank=True,
	)

	evaluator_themes = models.ManyToManyField(
		"CanonicalTheme",
		through="EntryEvaluationTheme",
		related_name="entry_evaluations",
		blank=True,
	)

	evaluator_issues = models.ManyToManyField(
		"CanonicalIssue",
		through="EntryEvaluationIssue",
		related_name="entry_evaluations",
		blank=True,
	)

	evaluator_notes = models.TextField(blank=True)

	evaluated_by = models.ForeignKey(
		settings.AUTH_USER_MODEL,
		on_delete=models.SET_NULL,
		blank=True,
		null=True,
		related_name="entry_evaluations",
	)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	@property
	def is_complete(self):
		return (
			bool(self.evaluator_sentiment_label)
			and self.evaluator_themes.exists()
		)

	class Meta:
		ordering = ["-updated_at", "-created_at"]
		constraints = [
			models.UniqueConstraint(
				fields=["entry", "evaluated_by"],
				name="unique_entry_evaluation_per_evaluator",
			),
		]
		indexes = [
			models.Index(fields=["entry"]),
			models.Index(fields=["evaluated_by"]),
			models.Index(fields=["updated_at"]),
		]
		verbose_name = "Entry Evaluation"
		verbose_name_plural = "Entry Evaluations"

	def __str__(self) -> str:
		return f"EntryEvaluation {self.pk} · entry={self.entry_id}"


# CANONICAL THEME
# Global evaluator-defined theme catalog.
class CanonicalTheme(models.Model):

	name = models.CharField(max_length=255, unique=True)
	description = models.TextField(blank=True)
	aliases = models.JSONField(default=list, blank=True)

	source = models.CharField(
		max_length=20,
		choices=ThemeAndIssueSource.choices,
		null=True,
		blank=True,
	)

	examples = models.TextField(
		blank=True,
	)

	status = models.CharField(
		max_length=20,
		choices=ThemeAndIssueStatus.choices,
		null=True,
		blank=True,
	)

	is_active = models.BooleanField(default=True)

	created_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="created_themes",
	)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ["name"]
		indexes = [
			models.Index(fields=["is_active"]),
			models.Index(fields=["name"]),
		]
		verbose_name = "Canonical Theme"
		verbose_name_plural = "Canonical Themes"

	def clean(self):
		if self.name:
			self.name = self.name.strip()

		if not self.name:
			raise ValidationError({"name": "Canonical theme name is required."})

	def __str__(self) -> str:
		return self.name


# CANONICAL ISSUE
# Global evaluator-defined usability issue catalog.
class CanonicalIssue(models.Model):

	name = models.CharField(max_length=255, unique=True)
	description = models.TextField(blank=True)
	aliases = models.JSONField(default=list, blank=True)

	source = models.CharField(
		max_length=20,
		choices=ThemeAndIssueSource.choices,
		null=True,
		blank=True,
	)

	examples = models.TextField(
		blank=True,
	)

	status = models.CharField(
		max_length=20,
		choices=ThemeAndIssueStatus.choices,
		null=True,
		blank=True,
	)

	is_active = models.BooleanField(default=True)

	created_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="created_issues",
	)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ["name"]
		indexes = [
			models.Index(fields=["is_active"]),
			models.Index(fields=["name"]),
		]
		verbose_name = "Canonical Issue"
		verbose_name_plural = "Canonical Issues"

	def clean(self):
		if self.name:
			self.name = self.name.strip()

		if not self.name:
			raise ValidationError({"name": "Canonical issue name is required."})

	def __str__(self) -> str:
		return self.name
	

# ENTRY ANALYSIS THEME
# Themes assigned to an individual entry analysis.
class EntryAnalysisTheme(models.Model):

	entry_analysis = models.ForeignKey(
		EntryAnalysis,
		on_delete=models.CASCADE,
		related_name="theme_assignments",
	)

	theme = models.ForeignKey(
		CanonicalTheme,
		on_delete=models.PROTECT,
		related_name="entry_analysis_assignments",
	)

	confidence_score = models.FloatField(null=True, blank=True)
	rationale = models.TextField(blank=True)

	assigned_by_method = models.CharField(
		max_length=100,
		blank=True,
		help_text="Name of the NLP technique, LLM provider/model, or process that assigned this theme.",
	)

	assigned_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ["theme__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["entry_analysis", "theme"],
				name="unique_theme_per_entry_analysis",
			),
		]
		indexes = [
			models.Index(fields=["entry_analysis"]),
			models.Index(fields=["theme"]),
		]
		verbose_name = "Entry Analysis Theme"
		verbose_name_plural = "Entry Analysis Themes"

	def __str__(self) -> str:
		return f"{self.theme} · analysis={self.entry_analysis_id}"


# ENTRY ANALYSIS ISSUE
# Issues assigned to an individual entry analysis.
class EntryAnalysisIssue(models.Model):

	entry_analysis = models.ForeignKey(
		EntryAnalysis,
		on_delete=models.CASCADE,
		related_name="issue_assignments",
	)

	issue = models.ForeignKey(
		CanonicalIssue,
		on_delete=models.PROTECT,
		related_name="entry_analysis_assignments",
	)

	confidence_score = models.FloatField(null=True, blank=True)
	rationale = models.TextField(blank=True)

	assigned_by_method = models.CharField(
		max_length=100,
		blank=True,
		help_text="NLP technique, LLM provider/model, or process that assigned this issue.",
	)

	assigned_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ["issue__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["entry_analysis", "issue"],
				name="unique_issue_per_entry_analysis",
			),
		]
		indexes = [
			models.Index(fields=["entry_analysis"]),
			models.Index(fields=["issue"]),
		]
		verbose_name = "Entry Analysis Issue"
		verbose_name_plural = "Entry Analysis Issues"

	def __str__(self) -> str:
		return f"{self.issue} · analysis={self.entry_analysis_id}"
	

# ENTRY EVALUATION THEME
# Themes assigned by a human evaluator to an individual entry evaluation.
class EntryEvaluationTheme(models.Model):

	entry_evaluation = models.ForeignKey(
		EntryEvaluation,
		on_delete=models.CASCADE,
		related_name="evaluator_theme_assignments",
	)

	theme = models.ForeignKey(
		CanonicalTheme,
		on_delete=models.PROTECT,
		related_name="entry_evaluation_assignments",
	)

	rationale = models.TextField(blank=True)

	assigned_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="assigned_evaluator_entry_evaluation_themes",
	)

	assigned_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ["theme__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["entry_evaluation", "theme"],
				name="unique_evaluator_theme_per_entry_evaluation",
			),
		]
		indexes = [
			models.Index(fields=["entry_evaluation"]),
			models.Index(fields=["theme"]),
		]
		verbose_name = "Entry Evaluation Theme"
		verbose_name_plural = "Entry Evaluation Themes"

	def __str__(self) -> str:
		return f"{self.theme} · evaluator evaluation={self.entry_evaluation_id}"
	

# ENTRY EVALUATION ISSUE
# Issues assigned by a human evaluator to an individual entry evaluation.
class EntryEvaluationIssue(models.Model):

	entry_evaluation = models.ForeignKey(
		EntryEvaluation,
		on_delete=models.CASCADE,
		related_name="evaluator_issue_assignments",
	)

	issue = models.ForeignKey(
		CanonicalIssue,
		on_delete=models.PROTECT,
		related_name="entry_evaluation_assignments",
	)

	rationale = models.TextField(blank=True)

	assigned_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="assigned_evaluator_entry_evaluation_issues",
	)

	assigned_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ["issue__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["entry_evaluation", "issue"],
				name="unique_evaluator_issue_per_entry_evaluation",
			),
		]
		indexes = [
			models.Index(fields=["entry_evaluation"]),
			models.Index(fields=["issue"]),
		]
		verbose_name = "Entry Evaluation Issue"
		verbose_name_plural = "Entry Evaluation Issues"

	def __str__(self) -> str:
		return f"{self.issue} · evaluator evaluation={self.entry_evaluation_id}"
	

# STUDY EVALUATION THEME
# Themes found across all entry evaluations in a study.
class StudyEvaluationTheme(models.Model):

	study = models.ForeignKey(
		Study,
		on_delete=models.CASCADE,
		related_name="evaluation_theme_summaries",
	)

	theme = models.ForeignKey(
		CanonicalTheme,
		on_delete=models.PROTECT,
		related_name="study_evaluation_summaries",
	)

	entry_count = models.PositiveIntegerField(default=0)
	average_confidence_score = models.FloatField(null=True, blank=True)

	class Meta:
		ordering = ["-entry_count", "theme__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["study", "theme"],
				name="unique_theme_per_study",
			),
		]
		indexes = [
			models.Index(fields=["study"]),
			models.Index(fields=["theme"]),
			models.Index(fields=["study", "entry_count"]),
		]
		verbose_name = "Study Evaluation Theme"
		verbose_name_plural = "Study Evaluation Themes"

	def __str__(self) -> str:
		return f"{self.theme} · study={self.study_id} · entries={self.entry_count}"


# STUDY EVALUATION ISSUE
# Issues found across all entry evaluations in a study.
class StudyEvaluationIssue(models.Model):

	study = models.ForeignKey(
		Study,
		on_delete=models.CASCADE,
		related_name="evaluation_issue_summaries",
	)

	issue = models.ForeignKey(
		CanonicalIssue,
		on_delete=models.PROTECT,
		related_name="study_evaluation_summaries",
	)

	entry_count = models.PositiveIntegerField(default=0)
	average_confidence_score = models.FloatField(null=True, blank=True)

	class Meta:
		ordering = ["-entry_count", "issue__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["study", "issue"],
				name="unique_issue_per_study",
			),
		]
		indexes = [
			models.Index(fields=["study"]),
			models.Index(fields=["issue"]),
			models.Index(fields=["study", "entry_count"]),
		]
		verbose_name = "Study Evaluation Issue"
		verbose_name_plural = "Study Evaluation Issues"

	def __str__(self) -> str:
		return f"{self.issue} · study={self.study_id} · entries={self.entry_count}"
	

# STUDY ANALYSIS THEME
# Themes found across all entry analyses in a run.
# So I don't have to recompute every time from the entries.
class StudyAnalysisTheme(models.Model):

	run = models.ForeignKey(
		StudyAnalysis,
		on_delete=models.CASCADE,
		related_name="theme_summaries",
	)

	theme = models.ForeignKey(
		CanonicalTheme,
		on_delete=models.PROTECT,
		related_name="run_summaries",
	)

	entry_count = models.PositiveIntegerField(default=0)
	average_confidence_score = models.FloatField(null=True, blank=True)
	average_sentiment_score = models.FloatField(null=True, blank=True)

	class Meta:
		ordering = ["-entry_count", "theme__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["run", "theme"],
				name="unique_theme_per_run",
			),
		]
		indexes = [
			models.Index(fields=["run"]),
			models.Index(fields=["theme"]),
			models.Index(fields=["run", "entry_count"]),
		]
		verbose_name = "Study Analysis Theme"
		verbose_name_plural = "Study Analysis Themes"

	def __str__(self) -> str:
		return f"{self.theme} · run={self.run_id} · entries={self.entry_count}"


# STUDY ANALYSIS ISSUE
# Issues found across all entry analyses in a run.
# So I don't have to recompute every time from the entries.
class StudyAnalysisIssue(models.Model):

	run = models.ForeignKey(
		StudyAnalysis,
		on_delete=models.CASCADE,
		related_name="issue_summaries",
	)

	issue = models.ForeignKey(
		CanonicalIssue,
		on_delete=models.PROTECT,
		related_name="run_summaries",
	)

	entry_count = models.PositiveIntegerField(default=0)
	average_confidence_score = models.FloatField(null=True, blank=True)
	average_sentiment_score = models.FloatField(null=True, blank=True)

	class Meta:
		ordering = ["-entry_count", "issue__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["run", "issue"],
				name="unique_issue_per_run",
			),
		]
		indexes = [
			models.Index(fields=["run"]),
			models.Index(fields=["issue"]),
			models.Index(fields=["run", "entry_count"]),
		]
		verbose_name = "Study Analysis Issue"
		verbose_name_plural = "Study Analysis Issues"

	def __str__(self) -> str:
		return f"{self.issue} · run={self.run_id} · entries={self.entry_count}"
	

# UX FRAMEWORK
# Catalogue of usability frameworks, such as Nielsen, ISO, WCAG, or project-specific frameworks.
class UXFramework(models.Model):

	name = models.CharField(max_length=255, unique=True)
	description = models.TextField(blank=True)
	framework_type = models.CharField(
		max_length=20,
		choices=UXFrameworkType.choices,
		default=UXFrameworkType.CUSTOM,
	)
	source = models.CharField(
		max_length=255,
		blank=True,
		help_text="Source, publication, standard, or reference used for this framework.",
	)
	version = models.CharField(max_length=80, blank=True)
	is_active = models.BooleanField(default=True)

	created_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="created_ux_frameworks",
	)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ["name"]
		indexes = [
			models.Index(fields=["is_active"]),
			models.Index(fields=["framework_type"]),
			models.Index(fields=["name"]),
		]
		verbose_name = "UX Framework"
		verbose_name_plural = "UX Frameworks"

	def clean(self):
		if self.name:
			self.name = self.name.strip()

		if not self.name:
			raise ValidationError({"name": "UX framework name is required."})

	def __str__(self) -> str:
		return self.name


# UX FRAMEWORK CRITERION
# Individual criterion, heuristic, principle, dimension, or guideline inside a framework.
class UXFrameworkCriterion(models.Model):

	framework = models.ForeignKey(
		UXFramework,
		on_delete=models.CASCADE,
		related_name="criteria",
	)
	
	code = models.CharField(
		max_length=80,
		blank=True,
		help_text="Optional framework code, e.g. N1, ISO-9241-EFFICIENCY.",
	)

	name = models.CharField(
		max_length=255,
		help_text="Name of the framework criterion, heuristic, principle, dimension, or guideline, e.g. Visibility of System Status or Efficiency.",
	)

	description = models.TextField(
		blank=True,
		help_text="Explain what this criterion means and what kind of usability problem it helps identify.",
	)

	aliases = models.JSONField(
		default=list,
		blank=True,
		help_text="Alternative terms, labels, or phrases that may refer to this criterion. Used to support matching and search.",
	)

	examples = models.TextField(
		blank=True,
		help_text="Optional examples of user comments, diary evidence, or UX situations that would fit this criterion.",
	)

	recommendation_guidance = models.TextField(
		blank=True,
		help_text="Reusable recommendation guidance associated with this framework criterion.",
	)
	evaluation_questions = models.JSONField(
		default=list,
		blank=True,
		help_text="Optional evaluator-facing questions that help interpret this criterion.",
	)
	is_active = models.BooleanField(default=True)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ["framework__name", "code", "name"]
		constraints = [
			models.UniqueConstraint(
				fields=["framework", "name"],
				name="unique_ux_framework_criterion_name",
			),
		]
		indexes = [
			models.Index(fields=["framework"]),
			models.Index(fields=["is_active"]),
			models.Index(fields=["name"]),
		]
		verbose_name = "UX Framework Criterion"
		verbose_name_plural = "UX Framework Criteria"

	def clean(self):
		if self.name:
			self.name = self.name.strip()

		if self.code:
			self.code = self.code.strip()

		if not self.name:
			raise ValidationError({"name": "UX framework criterion name is required."})

	def __str__(self) -> str:
		return f"{self.framework.name} · {self.name}"


# CANONICAL ISSUE TO FRAMEWORK MAPPING
# Maps canonical issues to framework criteria, either manually or through a framework matching engine.
class CanonicalIssueToFrameworkMapping(models.Model):

	issue = models.ForeignKey(
		CanonicalIssue,
		on_delete=models.CASCADE,
		related_name="framework_mappings",
	)
	criterion = models.ForeignKey(
		UXFrameworkCriterion,
		on_delete=models.CASCADE,
		related_name="issue_mappings",
	)

	method = models.CharField(
		max_length=20,
		choices=FrameworkMappingMethod.choices,
		default=FrameworkMappingMethod.MANUAL,
	)
	status = models.CharField(
		max_length=20,
		choices=FrameworkMappingStatus.choices,
		default=FrameworkMappingStatus.SUGGESTED,
	)
	score = models.FloatField(
		null=True,
		blank=True,
		help_text="Similarity or confidence score produced by the matching method.",
	)
	rationale = models.TextField(blank=True)

	created_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="created_issue_framework_mappings",
	)
	approved_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="approved_issue_framework_mappings",
	)
	approved_at = models.DateTimeField(null=True, blank=True)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ["issue__name", "criterion__framework__name", "criterion__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["issue", "criterion"],
				name="unique_issue_to_framework_criterion",
			),
		]
		indexes = [
			models.Index(fields=["issue"]),
			models.Index(fields=["criterion"]),
			models.Index(fields=["method"]),
			models.Index(fields=["status"]),
		]
		verbose_name = "Canonical Issue to Framework Mapping"
		verbose_name_plural = "Canonical Issue to Framework Mappings"

	def clean(self):
		if self.status == FrameworkMappingStatus.APPROVED and not self.approved_at:
			self.approved_at = timezone.now()

	def __str__(self) -> str:
		return f"{self.issue} → {self.criterion}"


# CANONICAL THEME TO FRAMEWORK MAPPING
# Maps canonical themes to framework criteria, either manually or through a framework matching engine.
class CanonicalThemeToFrameworkMapping(models.Model):

	theme = models.ForeignKey(
		CanonicalTheme,
		on_delete=models.CASCADE,
		related_name="framework_mappings",
	)
	criterion = models.ForeignKey(
		UXFrameworkCriterion,
		on_delete=models.CASCADE,
		related_name="theme_mappings",
	)

	method = models.CharField(
		max_length=20,
		choices=FrameworkMappingMethod.choices,
		default=FrameworkMappingMethod.MANUAL,
	)
	status = models.CharField(
		max_length=20,
		choices=FrameworkMappingStatus.choices,
		default=FrameworkMappingStatus.SUGGESTED,
	)
	score = models.FloatField(
		null=True,
		blank=True,
		help_text="Similarity or confidence score produced by the matching method.",
	)
	rationale = models.TextField(blank=True)

	created_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="created_theme_framework_mappings",
	)
	approved_by = models.ForeignKey(
		User,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="approved_theme_framework_mappings",
	)
	approved_at = models.DateTimeField(null=True, blank=True)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ["theme__name", "criterion__framework__name", "criterion__name"]
		constraints = [
			models.UniqueConstraint(
				fields=["theme", "criterion"],
				name="unique_theme_to_framework_criterion",
			),
		]
		indexes = [
			models.Index(fields=["theme"]),
			models.Index(fields=["criterion"]),
			models.Index(fields=["method"]),
			models.Index(fields=["status"]),
		]
		verbose_name = "Canonical Theme to Framework Mapping"
		verbose_name_plural = "Canonical Theme to Framework Mappings"

	def clean(self):
		if self.status == FrameworkMappingStatus.APPROVED and not self.approved_at:
			self.approved_at = timezone.now()

	def __str__(self) -> str:
		return f"{self.theme} → {self.criterion}"