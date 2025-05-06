# views.py

from django.shortcuts import render

def portfolio1(request):
    return render(request, "analyzer/portfolio1.html")

def portfolio2(request):
    return render(request, "analyzer/portfolio2.html")
