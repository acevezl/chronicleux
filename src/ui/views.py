from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import login

from .forms import SignUpForm
from studies.models import Study
from django.db.models import Q

@login_required
def dashboard(request):

    studies = Study.objects.all().order_by("-created_at")

    search = request.GET.get("q")
    status = request.GET.get("status")
    owner = request.GET.get("owner")

    if search:
        studies = studies.filter(
            Q(title__icontains=search) |
            Q(description__icontains=search)
        )

    if status:
        studies = studies.filter(status=status)

    if owner:
        studies = studies.filter(owner__username__icontains=owner)

    return render(request, "dashboard.html", {
        "studies": studies
    })

def signup(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("dashboard")
    else:
        form = SignUpForm()

    return render(request, "registration/signup.html", {"form": form})