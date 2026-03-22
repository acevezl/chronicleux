from datetime import date, timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from studies.models import (
    Study,
    StudyMembership,
    DiaryEntry,
    StudyStatus,
    EntryFrequency,
    MembershipRole,
)

User = get_user_model()


class Command(BaseCommand):
    help = "Seeds the database with reusable ChronicleUX sample data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--wipe",
            action="store_true",
            help="Delete existing seeded sample data before recreating it",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        if options["wipe"]:
            self.stdout.write(self.style.WARNING("Removing old seeded sample data..."))
            self._wipe_seed_data()

        self.stdout.write("Creating users...")
        users = self._create_users()

        self.stdout.write("Creating studies...")
        studies = self._create_studies(users)

        self.stdout.write("Creating memberships...")
        self._create_memberships(users, studies)

        self.stdout.write("Creating diary entries...")
        self._create_diary_entries(users, studies)

        self.stdout.write(self.style.SUCCESS("Seed completed successfully."))

    def _wipe_seed_data(self):
        sample_usernames = [
            "luigi",
            "eva_researcher",
            "participant_anna",
            "participant_mateo",
            "participant_lucia",
        ]

        sample_study_titles = [
            "Mobile Fitness App Motivation Study",
            "Indie Game Player Experience Diary Study",
            "Food Delivery App Friction Study",
        ]

        DiaryEntry.objects.filter(study__title__in=sample_study_titles).delete()
        StudyMembership.objects.filter(study__title__in=sample_study_titles).delete()
        Study.objects.filter(title__in=sample_study_titles).delete()
        User.objects.filter(username__in=sample_usernames).delete()

    def _create_users(self):
        users = {}

        users["luigi"], _ = User.objects.get_or_create(
            username="luigi",
            defaults={
                "email": "luigi@example.com",
                "first_name": "Luigi",
                "last_name": "Arnoldi",
                "is_staff": True,
                "is_superuser": True,
            },
        )
        users["luigi"].set_password("changeme123")
        users["luigi"].save()

        users["eva_researcher"], _ = User.objects.get_or_create(
            username="eva_researcher",
            defaults={
                "email": "eva@example.com",
                "first_name": "Eva",
                "last_name": "Martin",
                "is_staff": True,
            },
        )
        users["eva_researcher"].set_password("changeme123")
        users["eva_researcher"].save()

        users["participant_anna"], _ = User.objects.get_or_create(
            username="participant_anna",
            defaults={
                "email": "anna@example.com",
                "first_name": "Anna",
                "last_name": "Lopez",
            },
        )
        users["participant_anna"].set_password("changeme123")
        users["participant_anna"].save()

        users["participant_mateo"], _ = User.objects.get_or_create(
            username="participant_mateo",
            defaults={
                "email": "mateo@example.com",
                "first_name": "Mateo",
                "last_name": "Lane",
            },
        )
        users["participant_mateo"].set_password("changeme123")
        users["participant_mateo"].save()

        users["participant_lucia"], _ = User.objects.get_or_create(
            username="participant_lucia",
            defaults={
                "email": "lucia@example.com",
                "first_name": "Lucia",
                "last_name": "Santos",
            },
        )
        users["participant_lucia"].set_password("changeme123")
        users["participant_lucia"].save()

        return users

    def _create_studies(self, users):
        studies = {}

        studies["fitness"], _ = Study.objects.get_or_create(
            title="Mobile Fitness App Motivation Study",
            defaults={
                "description": (
                    "A 14-day diary study exploring motivation, adherence, "
                    "and friction in a mobile fitness application."
                ),
                "participant_instructions": (
                    "Use the fitness app as part of your normal routine for 14 days. "
                    "Submit one diary entry per day describing what you tried to do, "
                    "how the experience felt, any friction you encountered, and anything "
                    "that motivated or discouraged you."
                ),
                "goal": (
                    "Understand how a mobile fitness app influences participant motivation, "
                    "routine adherence, and perceived friction over a 14-day period."
                ),
                "context": (
                    "This study focuses on a consumer mobile fitness application used in "
                    "everyday life settings such as home workouts, lunch breaks, and evening exercise."
                ),
                "hypotheses": (
                    "1. Daily reminders and streak mechanics will increase short-term motivation.\n"
                    "2. Participants will report frustration when notifications feel excessive or repetitive.\n"
                    "3. Clear progress indicators will be associated with more positive diary entries."
                ),
                "tags": "fitness, mobile-app, motivation, habit-formation, notifications, longitudinal-ux",
                "owner": users["luigi"],
                "status": StudyStatus.COLLECTING,
                "entry_frequency": EntryFrequency.DAILY,
                "data_collection_start": timezone.now() - timedelta(days=10),
                "data_collection_end": timezone.now() + timedelta(days=4),
            },
        )

        studies["game"], _ = Study.objects.get_or_create(
            title="Indie Game Player Experience Diary Study",
            defaults={
                "description": (
                    "Diary study examining enjoyment, frustration, and progression "
                    "in an indie game experience over time."
                ),
                "participant_instructions": (
                    "Play the assigned indie game regularly during the study period. "
                    "After each play session, record a diary entry describing what happened, "
                    "how you felt, whether anything frustrated or delighted you, and whether "
                    "you would want to keep playing."
                ),
                "goal": (
                    "Explore how players experience enjoyment, frustration, challenge, "
                    "and narrative engagement while playing an indie game over time."
                ),
                "context": (
                    "This study examines the player experience of a narrative-driven indie game, "
                    "with attention to onboarding, mechanics, pacing, challenge, and emotional response."
                ),
                "hypotheses": (
                    "1. Players will report strong initial engagement driven by aesthetics and novelty.\n"
                    "2. Usability issues such as inventory management and backtracking will reduce satisfaction.\n"
                    "3. Narrative developments and fair challenge will improve emotional engagement over time."
                ),
                "tags": "gaming, player-experience, indie-game, engagement, frustration, ux-research",
                "owner": users["luigi"],
                "status": StudyStatus.ANALYZING,
                "entry_frequency": EntryFrequency.DAILY,
                "data_collection_start": timezone.now() - timedelta(days=20),
                "data_collection_end": timezone.now() - timedelta(days=6),
            },
        )

        studies["food"], _ = Study.objects.get_or_create(
            title="Food Delivery App Friction Study",
            defaults={
                "description": (
                    "A short diary study focused on delivery tracking, order confidence, "
                    "and usability friction in a food delivery app."
                ),
                "participant_instructions": (
                    "Whenever you place a food delivery order during the study period, "
                    "submit a diary entry after the experience. Describe what you were trying to do, "
                    "whether anything felt confusing or reassuring, and whether the app helped or hindered the process."
                ),
                "goal": (
                    "Identify moments of uncertainty, trust, and friction in the end-to-end "
                    "food delivery experience, from browsing to order tracking and receipt."
                ),
                "context": (
                    "This study focuses on real-world food ordering moments, including "
                    "meal selection, checkout, delivery waiting time, and handoff."
                ),
                "hypotheses": (
                    "1. Participants will experience the most friction during checkout and delivery tracking.\n"
                    "2. Unclear order status messages will reduce confidence in the service.\n"
                    "3. Fast reorder flows and transparent ETAs will contribute to a more positive experience."
                ),
                "tags": "food-delivery, mobile-app, checkout, tracking, trust, service-ux",
                "owner": users["eva_researcher"],
                "status": StudyStatus.PLANNING,
                "entry_frequency": EntryFrequency.EVENT_BASED,
                "data_collection_start": timezone.now() + timedelta(days=7),
                "data_collection_end": timezone.now() + timedelta(days=21),
            },
        )

        return studies

    def _create_memberships(self, users, studies):
        memberships = [
            (studies["fitness"], users["luigi"], MembershipRole.EVALUATOR),
            (studies["fitness"], users["eva_researcher"], MembershipRole.EVALUATOR),
            (studies["fitness"], users["participant_anna"], MembershipRole.PARTICIPANT),
            (studies["fitness"], users["participant_mateo"], MembershipRole.PARTICIPANT),
            (studies["game"], users["luigi"], MembershipRole.EVALUATOR),
            (studies["game"], users["participant_anna"], MembershipRole.PARTICIPANT),
            (studies["game"], users["participant_lucia"], MembershipRole.PARTICIPANT),
            (studies["food"], users["eva_researcher"], MembershipRole.EVALUATOR),
        ]

        for study, user, role in memberships:
            StudyMembership.objects.get_or_create(
                study=study,
                user=user,
                defaults={"role": role},
            )

    def _create_diary_entries(self, users, studies):
        entry_data = [
            {
                "study": studies["fitness"],
                "user": users["participant_anna"],
                "entries": [
                    (
                        "I felt motivated to start today because the workout plan looked manageable. The onboarding was clear and I liked the tone of the reminders.",
                        "POSITIVE",
                        False,
                        9,
                    ),
                    (
                        "I almost ignored the app because I received too many reminders in one day. It started feeling pushy instead of encouraging.",
                        "NEUTRAL",
                        True,
                        7,
                    ),
                    (
                        "Completing the session and seeing progress stats made me feel accomplished. The streak visualization helped a lot.",
                        "VERY_POSITIVE",
                        False,
                        5,
                    ),
                ],
            },
            {
                "study": studies["fitness"],
                "user": users["participant_mateo"],
                "entries": [
                    (
                        "Some exercise names were unclear and I had to look them up elsewhere. That interrupted the flow.",
                        "NEGATIVE",
                        True,
                        8,
                    ),
                    (
                        "The short routine fit well into my lunch break. I appreciated how fast I could get started.",
                        "POSITIVE",
                        False,
                        6,
                    ),
                    (
                        "The app logged me out unexpectedly and I did not feel like signing back in and setting everything up again.",
                        "VERY_NEGATIVE",
                        True,
                        4,
                    ),
                ],
            },
            {
                "study": studies["game"],
                "user": users["participant_anna"],
                "entries": [
                    (
                        "The art style and soundtrack pulled me in immediately. I was curious to keep exploring.",
                        "VERY_POSITIVE",
                        False,
                        14,
                    ),
                    (
                        "I spent too much time moving items around and could not easily tell what I should keep.",
                        "NEGATIVE",
                        True,
                        12,
                    ),
                    (
                        "The new story reveal made me want to continue. I felt more emotionally invested than yesterday.",
                        "POSITIVE",
                        False,
                        10,
                    ),
                ],
            },
            {
                "study": studies["game"],
                "user": users["participant_lucia"],
                "entries": [
                    (
                        "The tutorial explained everything but dragged on. I wanted more control earlier.",
                        "NEUTRAL",
                        True,
                        15,
                    ),
                    (
                        "I failed twice but the challenge felt fair. Finally beating the boss was satisfying.",
                        "POSITIVE",
                        False,
                        11,
                    ),
                    (
                        "Too much repeated walking between areas made the session feel padded.",
                        "NEGATIVE",
                        True,
                        9,
                    ),
                ],
            },
        ]

        for block in entry_data:
            study = block["study"]
            user = block["user"]

            for content, sentiment, issue_encountered, days_ago in block["entries"]:
                entry_date = timezone.now() - timedelta(days=days_ago)

                entry, created = DiaryEntry.objects.get_or_create(
                    study=study,
                    participant=user,
                    content=content,
                    defaults={
                        "sentiment_self_report": sentiment,
                        "issue_encountered": issue_encountered,
                    },
                )

                if created:
                    entry.created_at = entry_date
                    entry.save(update_fields=["created_at"])