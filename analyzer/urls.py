from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path("get-xgb-predictions/", views.get_xgb_predictions_api, name="get_xgb_predictions_api"),
]
