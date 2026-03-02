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
        base = "mt-1 w-full rounded-lg border px-3 py-2"
        self.fields["username"].widget.attrs.update({"class": base})
        self.fields["email"].widget.attrs.update({"class": base})
        self.fields["password1"].widget.attrs.update({"class": base})
        self.fields["password2"].widget.attrs.update({"class": base})