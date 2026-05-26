from django.contrib.auth import get_user_model

User = get_user_model()

user = User.objects.get(username="participant_mateo")
user.set_password("participant123#")
user.save()

print("Password updated")