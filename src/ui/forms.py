from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import get_user_model

User = get_user_model()


class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=False)

    class Meta:
        model = User
        fields = ("username", "email")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base = "text-gray-600 mt-1 px-3 py-2 block w-full rounded-sm border border-gray-300 focus:ring-gray-700 dark:text-gray-200"
        self.fields["username"].widget.attrs.update({"class": base})
        self.fields["email"].widget.attrs.update({"class": base})
        self.fields["password1"].widget.attrs.update({"class": base})
        self.fields["password2"].widget.attrs.update({"class": base})