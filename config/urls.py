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
from analyzer import views, home_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('analyzer/', include('analyzer.urls')),
    path('BTC/', views.index, name='home'),
    path("get-likes/", home_views.get_likes, name="get-likes"),
    path("add-like/", home_views.add_like, name="add-like"),
    path("get-visitors/", home_views.get_visitors, name="get-visitors"),
    path("support/", views.support_page, name="support"),
    path("get-btc-analysis/", views.get_btc_analysis, name="get-btc-analysis"),
    path("get-btc-technical-analysis/", views.get_btc_technical_analysis, name="get-btc-technical-analysis"),
    path("main-page/", home_views.home_page, name="home-page"),
    path("main-page/data/", home_views.home_page_summary_data, name="home-page-data"),
]
