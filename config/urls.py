"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from analyzer import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('analyzer/', include('analyzer.urls')),
    path('home/', views.index, name='home'),
    path("get-xgb-predictions/", views.get_xgb_predictions_api, name="get_xgb_predictions_api"),
    path("get-likes/", views.get_likes, name="get-likes"),
    path("add-like/", views.add_like, name="add-like"),
    path("get-visitors/", views.get_visitors, name="get-visitors"),
    path("support/", views.support_page, name="support"),
]
