from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import login

from .forms import SignUpForm

from studies.models import MembershipRole, StudyMembership

# HOME
@login_required
def home(request):
	return render(request, "/")

@login_required
def evaluator_dashboard(request):
	return render(request, "ui/evaluator_dashboard.html")

@login_required
def participant_dashboard(request):
	return render(request, "ui/participant_dashboard.html")

def signup(request):
    if request.user.is_authenticated:
        return redirect("evaluator_dashboard")

    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("evaluator_dashboard")
    else:
        form = SignUpForm()

    return render(request, "registration/signup.html", {"form": form})

# --------------------
# HELPERS
# --------------------
def get_user_initials(user):
    first = (user.first_name or "").strip()
    last = (user.last_name or "").strip()
    username = (user.username or "").strip()

    if first and last:
        return f"{first[0]}{last[0]}".upper()

    if username:
        return username[:2].upper()

    return "??"