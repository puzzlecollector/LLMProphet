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
from django.views.generic import RedirectView
from analyzer import views, home_views, eth_views, sol_views, xrp_views, ada_views, xlm_views, sui_views, link_views, ondo_views, high_risk_crypto_views

urlpatterns = [
    path('', RedirectView.as_view(url='/main-page/', permanent=True)),
    path('admin/', admin.site.urls),
    path('analyzer/', include('analyzer.urls')),
    path('high-risk/', high_risk_crypto_views.high_risk_crypto_view, name='high-risk-page'),
    path('BTC/', views.index, name='home'),
    path('ETH/', eth_views.index, name="eth"),
    path('SOL/', sol_views.index, name="sol"),
    path('XRP/', xrp_views.index, name="xrp"),
    path('ADA/', ada_views.index, name="ada"),
    path('XLM/', xlm_views.index, name="xlm"),
    path('SUI/', sui_views.index, name="sui"),
    path('LINK/', link_views.index, name="link"),
    path('ONDO/', ondo_views.index, name="ondo"),
    path("get-likes/", home_views.get_likes, name="get-likes"),
    path("add-like/", home_views.add_like, name="add-like"),
    path("get-visitors/", home_views.get_visitors, name="get-visitors"),
    path("support/", views.support_page, name="support"),
    path("get-btc-analysis/", views.get_btc_analysis, name="get-btc-analysis"),
    path("get-btc-technical-analysis/", views.get_btc_technical_analysis, name="get-btc-technical-analysis"),
    path("main-page/", home_views.home_page, name="home-page"),
    path("main-page/data/", home_views.home_page_summary_data, name="home-page-data"),
    path("get-eth-analysis/", eth_views.get_eth_analysis, name="get-eth-analysis"),
    path("get-eth-technical-analysis/", eth_views.get_eth_technical_analysis, name="get-eth-technical-analysis"),
    path("get-sol-analysis/", sol_views.get_sol_analysis, name="get-sol-analysis"),
    path("get-sol-technical-analysis/", sol_views.get_sol_technical_analysis, name="get-sol-technical-analysis"),
    path("get-xrp-analysis/", xrp_views.get_xrp_analysis, name="get-xrp-analysis"),
    path("get-xrp-technical-analysis/", xrp_views.get_xrp_technical_analysis, name="get-xrp-technical-analysis"),
    path("get-ada-analysis/", ada_views.get_ada_analysis, name="get-ada-analysis"),
    path("get-ada-technical-analysis/", ada_views.get_ada_technical_analysis, name="get-ada-technical-analysis"),
    path("get-xlm-analysis/", xlm_views.get_xlm_analysis, name="get-xlm-analysis"),
    path("get-xlm-technical-analysis/", xlm_views.get_xlm_technical_analysis, name="get-xlm-technical-analysis"),
    path("get-sui-analysis/", sui_views.get_sui_analysis, name="get-sui-analysis"),
    path("get-sui-technical-analysis/", sui_views.get_sui_technical_analysis, name="get-sui-technical-analysis"),
    path("get-link-analysis/", link_views.get_link_analysis, name="get-link-analysis"),
    path("get-link-technical-analysis/", link_views.get_link_technical_analysis, name="get-link-technical-analysis"),
    path("get-ondo-analysis/", ondo_views.get_ondo_analysis, name="get-ondo-analysis"),
    path("get-ondo-technical-analysis/", ondo_views.get_ondo_technical_analysis, name="get-ondo-technical-analysis"),
]
