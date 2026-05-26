from django.urls import path
from . import views

urlpatterns = [
    path("", views.studies, name="studies"),
    path("new/", views.create_study, name="create_study"),
    path("<int:pk>/", views.study_detail, name="study_detail"),
    path("<int:pk>/edit/", views.edit_study, name="edit_study"),
    path("studies/<int:pk>/import/", views.import_entries, name="import_entries"),
    path("<int:pk>/entries/", views.study_entries, name="study_entries"),
    path("<int:pk>/analysis/", views.study_analysis, name="study_analysis"),
    path("entries/<int:pk>/", views.entry_detail, name="entry_detail"),
    path("<int:pk>/run-machine-analysis/", views.run_machine_analysis, name="run_machine_analysis"),
    path("<int:pk>/analysis/", views.study_analysis, name="study_analysis"),
]