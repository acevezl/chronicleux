from django.urls import path
from . import views

urlpatterns = [

    # Studies
    path("", views.studies, name="studies"),
    path("new/", views.create_diary_study, name="create_diary_study"),
    path("<int:pk>/", views.diary_study_detail, name="diary_study_detail"),
    path("<int:pk>/edit/", views.edit_diary_study, name="edit_diary_study"),

    # Entries
    path("<int:pk>/entries/", views.entries, name="entries"),
    path("<int:study_pk>/entries/partial/", views.study_entries_partial, name="diary_entries_partial"),
    path("<int:study_pk>/entries/<int:entry_pk>/", views.diary_entry_detail, name="diary_entry_detail"),
    path("<int:pk>/import/", views.import_entries, name="import_entries"),
    path("<int:pk>/new-entry/", views.create_diary_entry, name="create_diary_entry"),

    # Machine Analysis
    path("<int:pk>/select-analysis-methods/", views.select_analysis_methods, name="select_analysis_methods"),
    path("<int:pk>/run-machine-analysis/", views.run_machine_analysis, name="run_machine_analysis"),
    path("<int:study_pk>/machine-analysis/<int:run_pk>/", views.machine_analysis_details, name="machine_analysis_details"),
    path("<int:study_pk>/machine-analysis/<int:run_pk>/entries/partial/", views.machine_analysis_entries_partial, name="machine_analysis_entries_partial"),

    # Manual Evaluation
    path("<int:study_pk>/machine-analysis/<int:run_pk>/human-evaluation/", views.human_evaluation_queue, name="human_evaluation_queue"),
    path("<int:study_pk>/machine-analysis/<int:run_pk>/human-evaluation/partial/", views.human_evaluation_queue_partial, name="human_evaluation_queue_partial"),
    path("<int:study_pk>/machine-analysis/<int:run_pk>/human-evaluation/<int:analysis_pk>/evaluate", views.evaluate_entry_analysis, name="evaluate_entry_analysis"),
    
    # Update Metrics
    path("studies/<int:study_pk>/machine-analysis/<int:run_pk>/refresh-metrics/", views.refresh_analysis_run_metrics, name="refresh_analysis_run_metrics"),

    # Evaluators / Participant Management
    path("<int:pk>/participants/",views.manage_participants, name="manage_participants"),
    path("<int:pk>/evaluators/",views.manage_evaluators, name="manage_evaluators"),

    # Canonical Themes
    path("catalogues/canonical-themes/", views.canonical_theme_catalogue_list, name="canonical_theme_catalogue_list"),
    path("catalogues/canonical-themes/create/", views.canonical_theme_create, name="canonical_theme_create"),
    path("catalogues/canonical-themes/<int:theme_pk>/edit/", views.canonical_theme_update, name="canonical_theme_update"),
    path("catalogues/canonical-themes/<int:theme_pk>/delete/", views.canonical_theme_delete, name="canonical_theme_delete"),
    path("catalogues/canonical-themes/import/", views.canonical_theme_import, name="canonical_theme_import"),
    path("catalogues/canonical-themes/partial/", views.canonical_theme_catalogue_partial, name="canonical_theme_catalogue_partial"),

    # Canonical Issues
    path("catalogues/canonical-issues/", views.canonical_issue_catalogue_list, name="canonical_issue_catalogue_list"),
    path("catalogues/canonical-issues/create/", views.canonical_issue_create, name="canonical_issue_create"),
    path("catalogues/canonical-issues/<int:issue_pk>/edit/", views.canonical_issue_update, name="canonical_issue_update"),
    path("catalogues/canonical-issues/<int:issue_pk>/delete/", views.canonical_issue_delete, name="canonical_issue_delete"),
    path("catalogues/canonical-issues/import/", views.canonical_issue_import, name="canonical_issue_import"),
    path("catalogues/canonical-issues/partial/", views.canonical_issue_catalogue_partial, name="canonical_issue_catalogue_partial"),
    
]