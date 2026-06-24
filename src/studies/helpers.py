from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db import transaction
from django.utils.dateparse import parse_datetime


from .models import (
	CanonicalTheme,
	CanonicalIssue,
	DiaryEntry,
	DiaryEntrySource,
	MembershipRole,
	SentimentCategory,
	StudyMembership,
	ThemeAndIssueSource,
	ThemeAndIssueStatus,
)

User = get_user_model()
VALID_SENTIMENTS = {choice[0] for choice in SentimentCategory.choices}

# ----------------------- HELPERS ----------------------- #

# USER CAN EVALUATE STUDY?
def user_can_evaluate_study(user, study):
	"""
	Returns True if the user can view/evaluate study entries and analysis.

	Allowed:
	- Study owner
	- Study members with evaluator role

	Not allowed:
	- Participants
	- Non authenticated users (obvs)
	"""

	if not user or not user.is_authenticated:
		return False

	if study.owner_id == user.id:
		return True

	return StudyMembership.objects.filter(
		study=study,
		user=user,
		role=MembershipRole.EVALUATOR,
	).exists()

# IMPORT ENTRIES: IMPORT ROWS INTO STUDY
@transaction.atomic
def import_rows_into_study (study, rows):
	created_count = 0
	skipped_owner = 0
	skipped_evaluator = 0

	for index, row in enumerate(rows, start=1):
		try:
			# Ignore the rows if they were authored by the study owner or an evaluator
			# B/c study owners and evaluators shall never write diary entries
			if is_owner_row(study, row):
				skipped_owner+=1
				continue

			if is_evaluator_row(study, row):
				skipped_evaluator+=1
				continue

			create_diary_entry_from_row(study, row)
			created_count+=1

		except Exception as e:
			raise ValueError(f"Row {index}: {e}")
	
	return {
		"created": created_count,
		"skipped_owner": skipped_owner,
		"skipped_evaluator": skipped_evaluator,
	}

# IMPORT ENTRIES: CREATE DIARY ENTRY FROM DATA ROW
def create_diary_entry_from_row(study, row):
	participant = resolve_participant_for_study(study, row)

	content = row.get("content")
	if not content:
		raise ValueError ("Field `content` is required in a diary entry")
	
	sentiment_self_report = row.get("sentiment_self_report")
	if not sentiment_self_report:
		raise ValueError ("Field `sentiment_self_report` is required in a diary entry")
	
	participant_display_name = row.get("participant_display_name")
	if participant is None and not participant_display_name:
		raise ValueError("Each entry must have either a resolvable ChronicleUX participant or a `participant_display_name`")
	
	created_at = parse_imported_datetime (row.get("created_at"))

	entry = DiaryEntry(
		study=study,
		participant=participant,
		participant_display_name=participant_display_name or "",
		participant_external_id=row.get("participant_external_id") or "",
		participant_email=row.get("participant_email") or "",
		content=content,
		sentiment_self_report=sentiment_self_report,
		issue_encountered=row.get("issue_encountered"),
		created_at=created_at,
		source=DiaryEntrySource.EXTERNAL,
	)

	entry.full_clean()
	entry.save()

	return entry

# ENTRY IMPORT HELPER: IS OWNER ROW
def is_owner_row(study, row):
	participant_email = row.get("participant_email")
	participant_external_id = row.get("participant_external_id")

	if participant_email and study.owner.email and participant_email.lower() == study.owner.email.lower():
		return True
	
	if participant_external_id and participant_external_id == study.owner.username:
		return True
	
	return False

# ENTRY IMPORT HELPER: IS EVALUATOR ROW
def is_evaluator_row(study, row):
	participant_email = row.get("participant_email")
	participant_external_id = row.get("participant_external_id")

	evaluator_user = None

	if participant_email:
		evaluator_user = User.objects.filter(email__iexact=participant_email).first()

	if evaluator_user is None and participant_external_id:
		evaluator_user = User.objects.filter(username=participant_external_id).first()

	if evaluator_user is None:
		return False

	return StudyMembership.objects.filter(
		study=study,
		user=evaluator_user,
		role=MembershipRole.EVALUATOR,
	).exists()


# ENTRY IMPORT HELPER: RESOLVE PARTICIPANT FOR STUDY
def resolve_participant_for_study(study, row):
	participant_email = row.get("participant_email")
	participant_external_id = row.get("participant_external_id")

	participant_user = None

	# If the entry has a participant e-mail or participant_external_id, see if they resolve to a user
	if participant_email:
		participant_user = User.objects.filter(email__iexact=participant_email).first()
	
	if participant_user is None and participant_external_id:
		participant_user = User.objects.filter(username=participant_external_id).first()
		
	# If a user was found, validate if the user is actually a study participant
	if participant_user:
		is_participant_in_study = StudyMembership.objects.filter(
			study=study,
			user=participant_user,
			role=MembershipRole.PARTICIPANT,
		).exists()

		# If the user is not a participant in the study, make them a participant
		if not is_participant_in_study:
			StudyMembership.objects.create(
				study=study,
				user=participant_user,
				role=MembershipRole.PARTICIPANT
			)

		return participant_user
	
	return None

# ENTRY IMPORT HELPER: PARSE IMPORTED DATETIME
def parse_imported_datetime(value):
	value = normalize_str(value)
	if value is None:
		return None

	dt = parse_datetime(value)
	if dt is None:
		raise ValueError(
			"Invalid created_at value. Use ISO 8601 format, for example "
			"'2026-03-29T14:30:00Z'."
		)

	return dt

# ENTRY IMPORT HELPER: PARSE UPLOADED FILE
def parse_uploaded_file(uploaded_file):
	filename = uploaded_file.name.lower()

	if filename.endswith(".csv"):
		return parse_csv(uploaded_file.file)

	if filename.endswith(".json"):
		return parse_json(uploaded_file.file)

	raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")

# ENTRY IMPORT HELPER: NORMALISE SENTIMENT
def normalize_sentiment(value):
	value = normalize_str(value)
	if value is None:
		raise ValueError("sentiment_self_report is required.")
	if value not in VALID_SENTIMENTS:
		raise ValueError(
			f"Invalid sentiment_self_report: {value}. "
			f"Allowed values: {', '.join(VALID_SENTIMENTS)}"
		)
	return value

# ENTRY IMPORT HELPER: NORMALIZE STRING
def normalize_str(value):
	if value is None:
		return None
	value = str(value).strip()
	return value if value else None

# ENTRY IMPORT HELPER: NORMALIZE BOOLEAN
def normalize_bool(value):
	if isinstance(value, bool):
		return value

	value = normalize_str(value)
	if value is None:
		raise ValueError("issue_encountered is required.")

	value = value.lower()
	if value in {"true", "1", "yes", "y"}:
		return True
	if value in {"false", "0", "no", "n"}:
		return False

	raise ValueError(f"Invalid boolean value: {value}")

# ENTRY IMPORT HELPER: NORMALIZE ROW
def normalize_row(row):
	return {
		"participant_external_id": normalize_str(row.get("participant_external_id")),
		"participant_display_name": normalize_str(row.get("participant_display_name")),
		"participant_email": normalize_str(row.get("participant_email")),
		"sentiment_self_report": normalize_sentiment(row.get("sentiment_self_report")),
		"issue_encountered": normalize_bool(row.get("issue_encountered")),
		"content": normalize_str(row.get("content")),
		"created_at": normalize_str(row.get("created_at")),
	}

# ENTRY IMPORT HELPER: PARSE CSV
def parse_csv(file):
	import csv
	from io import TextIOWrapper

	text_file = TextIOWrapper(file, encoding="utf-8", newline="")
	reader = csv.DictReader(text_file)

	rows = []
	for row in reader:
		rows.append(normalize_row(row))

	return rows

# ENTRY IMPORT HELPER: PARSE JSON
def parse_json(file):
	import json

	data = json.load(file)

	if not isinstance(data, list):
		raise ValueError("JSON must be a list of entries.")

	rows = []
	for item in data:
		if not isinstance(item, dict):
			raise ValueError("Each JSON entry must be an object.")
		rows.append(normalize_row(item))

	return rows

# CATALOGUE PERMISSION HELPER
# Only users who are STAFF can manage Theme and Issue Catalogue
def require_catalogue_manager(user):
    if not user.is_staff:
        raise PermissionDenied("You do not have permission to manage canonical catalogues.")
	

# CATALOG IMPORT HELPERS ---

VALID_THEME_AND_ISSUE_SOURCES = {choice[0] for choice in ThemeAndIssueSource.choices}
VALID_THEME_AND_ISSUE_STATUSES = {choice[0] for choice in ThemeAndIssueStatus.choices}

# CATALOG IMPORT HELPER: IMPORT CANONICAL THEMES
@transaction.atomic
def import_canonical_themes(rows, created_by_user):
	created_count = 0
	updated_count = 0
	skipped_count = 0

	for index, row in enumerate(rows, start=1):
		try:
			if not row.get("name"):
				skipped_count += 1
				continue

			theme, created = create_canonical_theme_from_row(row, created_by_user)

			if created:
				created_count += 1
			else:
				updated_count += 1

		except Exception as e:
			raise ValueError(f"Row {index}: {e}")

	return {
		"created": created_count,
		"updated": updated_count,
		"skipped": skipped_count,
	}

# CATALOG IMPORT HELPER: CREATE CANONICAL THEME FROM DATA ROW
def create_canonical_theme_from_row(row, created_by_user):
	name = row.get("name")
	if not name:
		raise ValueError("Field `name` is required in a canonical theme.")

	theme, created = CanonicalTheme.objects.update_or_create(
		name=name,
		defaults={
			"description": row.get("description") or "",
			"aliases": row.get("aliases") or [],
			"examples": row.get("examples") or "",
			"source": row.get("source") or ThemeAndIssueSource.EVALUATOR,
			"status": row.get("status") or ThemeAndIssueStatus.APPROVED,
			"is_active": row.get("is_active"),
			"created_by": created_by_user,
		},
	)

	return theme, created

# CATALOG IMPORT HELPER: PARSE UPLOADED CANONICAL THEME FILE
def parse_uploaded_canonical_theme_file(uploaded_file):
	if uploaded_file is None:
		raise ValueError("Please choose a CSV file to import.")

	filename = uploaded_file.name.lower()

	if filename.endswith(".csv"):
		return parse_canonical_theme_csv(uploaded_file.file)

	raise ValueError("Unsupported file type. Please upload a CSV file.")

# CATALOG IMPORT HELPER: PARSE CANONICAL THEME CSV
def parse_canonical_theme_csv(file):
	import csv
	from io import TextIOWrapper

	text_file = TextIOWrapper(file, encoding="utf-8-sig", newline="")
	reader = csv.DictReader(text_file)

	if not reader.fieldnames:
		raise ValueError("The CSV file is empty or missing a header row.")

	required_columns = {"name"}
	available_columns = {column.strip() for column in reader.fieldnames if column}
	missing_columns = required_columns - available_columns

	if missing_columns:
		raise ValueError(
			f"Missing required column(s): {', '.join(sorted(missing_columns))}."
		)

	rows = []
	for row in reader:
		rows.append(normalize_canonical_theme_row(row))

	return rows

# CATALOG IMPORT HELPER: IMPORT CANONICAL ISSUES
@transaction.atomic
def import_canonical_issues(rows, created_by_user):
	created_count = 0
	updated_count = 0
	skipped_count = 0

	for index, row in enumerate(rows, start=1):
		try:
			if not row.get("name"):
				skipped_count += 1
				continue

			issue, created = create_canonical_issue_from_row(row, created_by_user)

			if created:
				created_count += 1
			else:
				updated_count += 1

		except Exception as e:
			raise ValueError(f"Row {index}: {e}")

	return {
		"created": created_count,
		"updated": updated_count,
		"skipped": skipped_count,
	}

# CATALOG IMPORT HELPER: CREATE CANONICAL ISSUE FROM DATA ROW
def create_canonical_issue_from_row(row, created_by_user):
	name = row.get("name")
	if not name:
		raise ValueError("Field `name` is required in a canonical issue.")

	issue, created = CanonicalIssue.objects.update_or_create(
		name=name,
		defaults={
			"description": row.get("description") or "",
			"aliases": row.get("aliases") or [],
			"examples": row.get("examples") or "",
			"source": row.get("source") or ThemeAndIssueSource.EVALUATOR,
			"status": row.get("status") or ThemeAndIssueStatus.APPROVED,
			"is_active": row.get("is_active"),
			"created_by": created_by_user,
		},
	)

	return issue, created

# CATALOG IMPORT HELPER: PARSE UPLOADED CANONICAL ISSUE FILE
def parse_uploaded_canonical_issue_file(uploaded_file):
	if uploaded_file is None:
		raise ValueError("Please choose a CSV file to import.")

	filename = uploaded_file.name.lower()

	if filename.endswith(".csv"):
		return parse_canonical_issue_csv(uploaded_file.file)

	raise ValueError("Unsupported file type. Please upload a CSV file.")

# CATALOG IMPORT HELPER: PARSE CANONICAL ISSUE CSV
def parse_canonical_issue_csv(file):
	import csv
	from io import TextIOWrapper

	text_file = TextIOWrapper(file, encoding="utf-8-sig", newline="")
	reader = csv.DictReader(text_file)

	if not reader.fieldnames:
		raise ValueError("The CSV file is empty or missing a header row.")

	required_columns = {"name"}
	available_columns = {column.strip() for column in reader.fieldnames if column}
	missing_columns = required_columns - available_columns

	if missing_columns:
		raise ValueError(
			f"Missing required column(s): {', '.join(sorted(missing_columns))}."
		)

	rows = []
	for row in reader:
		rows.append(normalize_canonical_issue_row(row))

	return rows

# CATALOG IMPORT HELPER: NORMALIZE CANONICAL THEME ROW
def normalize_canonical_theme_row(row):
	return {
		"name": normalize_str(row.get("name")),
		"description": normalize_str(row.get("description")) or "",
		"aliases": normalize_aliases(row.get("aliases")),
		"examples": normalize_str(row.get("examples")) or "",
		"source": normalize_theme_and_issue_source(row.get("source")),
		"status": normalize_theme_and_issue_status(row.get("status")),
		"is_active": normalize_optional_bool(row.get("is_active"), default=True),
	}


# CATALOG IMPORT HELPER: NORMALIZE CANONICAL ISSUE ROW
def normalize_canonical_issue_row(row):
	return {
		"name": normalize_str(row.get("name")),
		"description": normalize_str(row.get("description")) or "",
		"aliases": normalize_aliases(row.get("aliases")),
		"examples": normalize_str(row.get("examples")) or "",
		"source": normalize_theme_and_issue_source(row.get("source")),
		"status": normalize_theme_and_issue_status(row.get("status")),
		"is_active": normalize_optional_bool(row.get("is_active"), default=True),
	}


# CATALOG IMPORT HELPER: NORMALIZE ALIASES
def normalize_aliases(value):
	value = normalize_str(value)
	if value is None:
		return []

	aliases = []

	for alias in value.replace("\n", "|").split("|"):
		clean_alias = alias.strip()

		if clean_alias and clean_alias not in aliases:
			aliases.append(clean_alias)

	return aliases


# CATALOG IMPORT HELPER: NORMALIZE THEME / ISSUE SOURCE
def normalize_theme_and_issue_source(value):
	value = normalize_str(value)

	if value is None:
		return ThemeAndIssueSource.EVALUATOR

	value = value.upper()

	if value not in VALID_THEME_AND_ISSUE_SOURCES:
		raise ValueError(
			f"Invalid source: {value}. "
			f"Allowed values: {', '.join(sorted(VALID_THEME_AND_ISSUE_SOURCES))}"
		)

	return value


# CATALOG IMPORT HELPER: NORMALIZE THEME / ISSUE STATUS
def normalize_theme_and_issue_status(value):
	value = normalize_str(value)

	if value is None:
		return ThemeAndIssueStatus.APPROVED

	value = value.upper()

	if value not in VALID_THEME_AND_ISSUE_STATUSES:
		raise ValueError(
			f"Invalid status: {value}. "
			f"Allowed values: {', '.join(sorted(VALID_THEME_AND_ISSUE_STATUSES))}"
		)

	return value


# CATALOG IMPORT HELPER: NORMALIZE OPTIONAL BOOLEAN
def normalize_optional_bool(value, default=True):
	value = normalize_str(value)

	if value is None:
		return default

	value = value.lower()

	if value in {"true", "1", "yes", "y"}:
		return True

	if value in {"false", "0", "no", "n"}:
		return False

	raise ValueError(f"Invalid boolean value: {value}")