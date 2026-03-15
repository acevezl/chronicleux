from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages

from .models import Study
from .forms import StudyForm
from .models import StudyMembership

@login_required
def create_study(request):
    if request.method == "POST":
        form = StudyForm(request.POST)
        if form.is_valid():
            study = form.save(commit=False)
            study.owner = request.user
            study.save()

            # Makes study creator the owner by default
            StudyMembership.objects.get_or_create(
                study = study,
                user = request.user,
                defaults={"role":"owner"}
            )

            messages.success(request, f"Study '{study.title}' created successfully by {study.owner}.")
            return redirect("study_detail", pk=study.pk)
    else:
        form = StudyForm()
    return render(request, "studies/create_study.html", {"form": form})

from django.shortcuts import get_object_or_404

@login_required
def study_detail(request, pk):
    study = get_object_or_404(Study, pk=pk)
    return render(request, "studies/study_detail.html", {"study": study})