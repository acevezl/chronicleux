"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.contrib.auth.views import LogoutView
from ui.views import home, evaluator_dashboard, participant_dashboard, signup


urlpatterns = [
    # Home
    path("", home, name="home"),
    path("dashboard/evaluator/", evaluator_dashboard, name="evaluator_dashboard"),
    path("dashboard/participant/", participant_dashboard, name="participant_dashboard"),

    # Admin
    path("admin/", admin.site.urls),
    path("accounts/", include("django.contrib.auth.urls")),
    path("accounts/signup/", signup, name="signup"),
    path("logout/", LogoutView.as_view(), name="logout"),
    
    # Studies
    path("studies/", include("studies.urls")),
]