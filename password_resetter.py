from django.contrib.auth import get_user_model

User = get_user_model()

user = User.objects.get(username="admin")
user.set_password("admin123#")
user.save()

print("Password updated")