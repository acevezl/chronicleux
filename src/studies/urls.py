from django.urls import path
from . import views

urlpatterns = [
    path("", views.studies, name="studies"),
    path("new/", views.create_study, name="create_study"),
    path("<int:pk>/", views.study_detail, name="study_detail"),
    path("<int:pk>/edit/", views.edit_study, name="edit_study"),
    path("studies/<int:pk>/import/", views.import_entries, name="import_entries"),
    path("<int:pk>/entries/", views.study_entries, name="study_entries"),
    path("studies/<int:study_pk>/entries/<int:entry_pk>/", views.diary_entry_detail, name="diary_entry_detail"),
    path("<int:pk>/run-machine-analysis/", views.run_machine_analysis, name="run_machine_analysis"),
    path("studies/<int:study_pk>/machine_analysis/<int:run_pk>/", views.machine_analysis_details, name="machine_analysis_details"),

    path("<int:pk>/participants/",views.manage_participants, name="manage_participants"),
    path("<int:pk>/evaluators/",views.manage_evaluators, name="manage_evaluators")
]