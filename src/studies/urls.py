from django.urls import path
from . import views

urlpatterns = [
    path("new/", views.create_study, name="create_study"),
    path("<int:pk>/", views.study_detail, name="study_detail"),
    path("<int:pk>/edit/", views.edit_study, name="edit_study"),
    path("studies/<int:pk>/import/", views.import_entries, name="import_entries"),
]