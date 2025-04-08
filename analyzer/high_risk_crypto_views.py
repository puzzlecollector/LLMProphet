# views.py
from django.shortcuts import render

def high_risk_crypto_view(request):
    return render(request, 'analyzer/high_risk_crypto.html')
