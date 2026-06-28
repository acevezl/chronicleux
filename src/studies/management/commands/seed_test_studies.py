from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.utils import timezone

from studies.models import EntryFrequency, MembershipRole, Study, StudyMembership, StudyStatus


class Command(BaseCommand):
	help = "Create test studies for ChronicleUX."

	def handle(self, *args, **options):
		User = get_user_model()

		owner, created = User.objects.get_or_create(
			username="luigi",
			defaults={
				"first_name": "Luigi",
				"last_name": "Arnoldi",
				"email": "luigi@chronicleux.test",
			},
		)

		if created:
			owner.set_password("test1234")
			owner.save()

		now = timezone.now()

		studies = [
			{
				"title": "WhatsApp Messaging Experience Diary Study",
				"description": "A diary study exploring how participants use WhatsApp for daily communication, group coordination, media sharing, and managing notifications.",
				"goal": "Identify usability issues, emotional responses, and recurring friction points in everyday WhatsApp use.",
				"context": "Participants document their WhatsApp interactions over time, including personal chats, group conversations, voice notes, media sharing, and notification handling.",
				"hypotheses": "Participants may experience friction around notification overload, search, media organization, group management, and privacy expectations.",
				"participant_instructions": "After using WhatsApp, write a short diary entry describing what you tried to do, what worked well, what felt frustrating, and how the experience made you feel.",
				"tags": "messaging, mobile app, communication, notifications, privacy",
				"entry_frequency": EntryFrequency.DAILY,
			},
			{
				"title": "Hogwarts Legacy Gameplay Experience Diary Study",
				"description": "A diary study examining player experiences with Hogwarts Legacy, including exploration, combat, quests, menus, progression systems, and accessibility.",
				"goal": "Understand how players experience usability, immersion, frustration, and satisfaction while playing Hogwarts Legacy.",
				"context": "Participants record gameplay sessions and reflect on navigation, quest clarity, combat, inventory management, map use, and overall enjoyment.",
				"hypotheses": "Players may encounter friction around quest tracking, map navigation, inventory management, combat learning curves, and accessibility settings.",
				"participant_instructions": "After each gameplay session, describe what you did, where you felt engaged or frustrated, and whether anything disrupted your sense of flow or immersion.",
				"tags": "gaming, console, PC, open world, accessibility, player experience",
				"entry_frequency": EntryFrequency.EVENT_BASED,
			},
			{
				"title": "Netflix Content Discovery Diary Study",
				"description": "A diary study investigating how participants browse, search, choose, and watch content on Netflix across different contexts.",
				"goal": "Identify usability and satisfaction patterns in Netflix content discovery, recommendation, search, playback, and profile management.",
				"context": "Participants document moments where they open Netflix, browse recommendations, search for specific content, start or abandon playback, and manage profiles or watchlists.",
				"hypotheses": "Participants may experience decision fatigue, recommendation mismatch, search friction, difficulty resuming content, and frustration with content organization.",
				"participant_instructions": "Whenever you use Netflix, describe what you wanted to watch, how you searched or browsed, what helped or blocked you, and how satisfied you felt with the experience.",
				"tags": "streaming, entertainment, recommendations, search, content discovery",
				"entry_frequency": EntryFrequency.FREEFORM,
			},
		]

		for study_data in studies:
			study, created = Study.objects.get_or_create(
				title=study_data["title"],
				defaults={
					**study_data,
					"owner": owner,
					"status": StudyStatus.PLANNING,
					"data_collection_start": now,
					"data_collection_end": now + timedelta(days=14),
				},
			)

			StudyMembership.objects.get_or_create(
				study=study,
				user=owner,
				defaults={"role": MembershipRole.EVALUATOR},
			)

			action = "Created" if created else "Already exists"
			self.stdout.write(f"{action}: {study.title}")

		self.stdout.write(self.style.SUCCESS("Seeded test studies."))