from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from studies.models import MembershipRole, Study, StudyMembership


class Command(BaseCommand):
	help = "Create test evaluators and participants."

	def add_arguments(self, parser):
		parser.add_argument(
			"--study-id",
			type=int,
			help="Optional study ID. If provided, users are added as study members.",
		)

	def handle(self, *args, **options):
		User = get_user_model()
		default_password = "test123#"

		evaluators = [
			("luigi", "Luigi", "Arnoldi"),
			("maya", "Maya", "Rudolph"),
			("tina", "Tina", "Fey"),
			("amy", "Amy", "Poehler"),
			("seth", "Seth", "Meyers"),
		]

		participants = [
            ("p_andy_samberg", "Andy", "Samberg"),
            ("p_cecily_strong", "Cecily", "Strong"),
            ("p_will_ferrell", "Will", "Ferrell"),
            ("p_adam_sandler", "Adam", "Sandler"),
            ("p_eddie_murphy", "Eddie", "Murphy"),
            ("p_mike_myers", "Mike", "Myers"),
            ("p_rachel_dratch", "Rachel", "Dratch"),
            ("p_marcelo_hernandez", "Marcelo", "Hernandez"),
            ("p_kristen_wiig", "Kristen", "Wiig"),
            ("p_kate_mckinnon", "Kate", "McKinnon"),
            ("p_mikey_day", "Mikey", "Day"),
            ("p_dan_aykroyd", "Dan", "Aykroyd"),
            ("p_jason_sudekis", "Jason", "Sudekis"),
            ("p_jane_curtin", "Jane", "Curtin"),
            ("p_gilda_radner", "Gilda", "Radner"),
            ("p_molly_shannon", "Molly", "Shannon"),
            ("p_dana_carvey", "Dana", "Carvey"),
            ("p_phil_hartman", "Phil", "Hartman"),
            ("p_jimmy_fallon", "Jimmy", "Fallon"),
            ("p_bill_hader", "Bill", "Hader"),
        ]

		study = None
		if options["study_id"]:
			study = Study.objects.get(pk=options["study_id"])

		created_users = []

		for username, first_name, last_name in evaluators:
			user, created = User.objects.get_or_create(
				username=username,
				defaults={
					"first_name": first_name,
					"last_name": last_name,
					"email": f"{username}@chronicleux.test",
					"is_staff": True,
				},
			)

			if created:
				user.set_password(default_password)
				user.save()
			elif not user.is_staff:
				user.is_staff = True
				user.save(update_fields=["is_staff"])

			created_users.append((user, MembershipRole.EVALUATOR))

		for username, first_name, last_name in participants:
			user, created = User.objects.get_or_create(
				username=username,
				defaults={
					"first_name": first_name,
					"last_name": last_name,
					"email": f"{username}@chronicleux.test",
				},
			)

			if created:
				user.set_password(default_password)
				user.save()

			created_users.append((user, MembershipRole.PARTICIPANT))

		if study:
			for user, role in created_users:
				StudyMembership.objects.get_or_create(
					study=study,
					user=user,
					defaults={"role": role},
				)

		self.stdout.write(self.style.SUCCESS("Seeded test users."))
		self.stdout.write(f"Default password: {default_password}")

		if study:
			self.stdout.write(
				self.style.SUCCESS(f"Added users to study: {study}")
			)