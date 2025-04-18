# views.py

from django.shortcuts import render


def basic_terms(request):
    return render(request, "analyzer/basic_terms.html")

def naive_model(request):
    return render(request, "analyzer/naive_model.html")
