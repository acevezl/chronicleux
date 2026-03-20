from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import login

from .forms import SignUpForm
from studies.models import Study
from django.db.models import Q

from django.core.paginator import Paginator

@login_required
def evaluator_dashboard(request):

    studies = Study.objects.all()

    # Filtering
    q = request.GET.get("q")
    owner = request.GET.get("owner")
    status = request.GET.get("status")

    if q:
        studies = studies.filter(title__icontains=q)

    if owner:
        studies = studies.filter(owner__username__icontains=owner)

    if status:
        studies = studies.filter(status=status)

    # Sorting
    sort = request.GET.get("sort", "-created_at")

    allowed_sort_fields = [
        "title",
        "created_at",
        "status",
        "owner__username",
    ]

    if sort.lstrip("-") in allowed_sort_fields:
        studies = studies.order_by(sort)

    # Pagination
    paginator = Paginator(studies, 10)  # 10 per page
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    sort_params = request.GET.copy()
    sort_params.pop("sort", None)
    sort_params.pop("page", None)

    page_params = request.GET.copy()
    page_params.pop("page", None)

    return render(request, "evaluator_dashboard.html", {
        "studies": page_obj,
        "page_obj": page_obj,
        "sort": sort,
        "sort_params": sort_params,
        "page_params": page_params,
    })

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


def get_user_initials(user):
    first = (user.first_name or "").strip()
    last = (user.last_name or "").strip()
    username = (user.username or "").strip()

    if first and last:
        return f"{first[0]}{last[0]}".upper()

    if username:
        return username[:2].upper()

    return "??"