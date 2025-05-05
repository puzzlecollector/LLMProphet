# views.py

from django.shortcuts import render


def basic_terms(request):
    return render(request, "analyzer/basic_terms.html")

def naive_model(request):
    return render(request, "analyzer/naive_model.html")

def classical_decomposition(request):
    return render(request, "analyzer/classical_decomposition.html")

def exponential_smoothing(request):
    return render(request, "analyzer/exponential_smoothing.html")

def holts_linear(request):
    return render(request, "analyzer/holts_linear.html")

def holt_winters(request):
    return render(request, "analyzer/holt_winters.html")

def ets(request):
    return render(request, "analyzer/ets.html")