# views.py

from django.shortcuts import render


def naive_model(request):
    return render(request, "analyzer/naive_model.html")
