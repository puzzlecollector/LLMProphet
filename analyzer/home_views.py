# views.py
import requests
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
import openai
import os
from django.conf import settings
from django.core.cache import cache
from datetime import timedelta
from django.utils.timezone import now
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_GET
from .models import Like, Visitor

@require_GET
def get_visitors(request):
    visitor_obj, created = Visitor.objects.get_or_create(id=1)
    return JsonResponse({"visitors": visitor_obj.count})

@csrf_protect
def add_like(request):
    if request.method == "POST":
        like_obj, created = Like.objects.get_or_create(id=1)
        like_obj.count += 1
        like_obj.save()
        cache.set("likes_count", like_obj.count, timeout=3600)
        return JsonResponse({"likes": like_obj.count})
    return JsonResponse({"error": "Invalid request"}, status=400)

@require_GET
def get_likes(request):
    likes = cache.get("likes_count")
    if likes is None:
        like_obj, created = Like.objects.get_or_create(id=1)
        likes = like_obj.count
        cache.set("likes_count", likes, timeout=3600)
    return JsonResponse({"likes": likes})



openai.api_key = settings.OPENAI_API_KEY
client = openai.OpenAI()

def fetch_with_retry(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def scrape_coinness_xhr():
    url = 'https://api.coinness.com/feed/v1/news'
    headers = {'User-Agent': 'Mozilla/5.0'}
    titles, contents, datetimes_arr = [], [], []

    try:
        news_data = fetch_with_retry(url, headers)
        for news_item in news_data:
            titles.append(news_item.get('title'))
            contents.append(news_item.get('content'))
            datetimes_arr.append(news_item.get('publishAt'))
    except requests.exceptions.RequestException as e:
        print("Error fetching data:", e)
        return pd.DataFrame()

    return pd.DataFrame({'titles': titles, 'contents': contents, 'datetimes': datetimes_arr})

def generate_summary_with_retry(information, max_retries=10):
    for _ in range(max_retries):
        try:
            prompt = f"""
            다음은 최근 암호화폐 관련 뉴스 속보 모음입니다. 전체 내용을 바탕으로 시장에 영향을 줄 수 있는 핵심 이슈나 트렌드를 정리해 주세요. 
            중요한 키워드, 정책, 기업 동향, 시장 심리에 영향을 줄 요소들을 중심으로 간결하고 분석적으로 요약해 주세요.
            {information}
            """
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 암호화폐 및 금융 시장 분석 전문가입니다. 투자자가 참고할 수 있도록 분석적으로 요약해 주세요."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print("OpenAI error:", e)
    return "요약을 가져오지 못했습니다."

# Page view
def home_page(request):
    # Increment visitor count
    visitor_obj, created = Visitor.objects.get_or_create(id=1)
    visitor_obj.count += 1
    visitor_obj.save()
    return render(request, "analyzer/home_page_summary.html", {"visitor_count": visitor_obj.count})

# Async API view
def home_page_summary_data(request):
    cached_summary = cache.get("coinness_summary")
    cached_summary_time = cache.get("coinness_summary_time")
    if cached_summary:
        return JsonResponse({"summary": cached_summary, "last_updated": cached_summary_time})

    df = scrape_coinness_xhr()
    if df.empty:
        return JsonResponse({"summary": "데이터를 불러오지 못했습니다."})

    news_data = "## 최근 속보 ##\n"
    for i, (t, c) in enumerate(zip(df["titles"], df["contents"])):
        news_data += f"\n{i+1}. {t}: {c}\n"

    summary = generate_summary_with_retry(news_data)

    # Set KST time
    last_updated_kst = (now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST")

    # Cache both summary and time
    cache.set("coinness_summary", summary, timeout=60 * 15)
    cache.set("coinness_summary_time", last_updated_kst, timeout=60 * 15)

    return JsonResponse({
        "summary": summary,
        "last_updated": last_updated_kst
    })
