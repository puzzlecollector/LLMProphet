from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from django.core.cache import cache
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator
from django.utils.timezone import now
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import requests
from tqdm import tqdm
import base64
from io import BytesIO
from .forms import ImageUploadForm
from .models import Like, Visitor
from background_task import background
from background_task.models import Task
import ccxt
import openai
import os
import pandas as pd
import numpy as np
import re
import logging
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import tiktoken
from datetime import datetime, timedelta, timezone
import pickle
import xgboost as xgb
from django.http import JsonResponse
from django.conf import settings

OKX_API_KEY = settings.OKX_API_KEY
OKX_API_SECRET_KEY = settings.OKX_API_SECRET_KEY
OKX_API_PASSPHRASE = settings.OKX_API_PASSPHRASE
OPENAI_API_KEY = settings.OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
client = openai.OpenAI()

okx = ccxt.okx({
    "apiKey": OKX_API_KEY,
    "secret": OKX_API_SECRET_KEY,
    "password": OKX_API_PASSPHRASE,
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
})

model_paths = {
    "1m": "analyzer/xgb_1m.pkl",
    "5m": "analyzer/xgb_5m.pkl",
    "30m": "analyzer/xgb_30m.pkl",
    "1h": "analyzer/xgb_1h.pkl",
    "4h": "analyzer/xgb_4h.pkl",
}
models = {key: pickle.load(open(path, "rb")) for key, path in model_paths.items()}

# Function to fetch Bitcoin chart data
def get_historical_ohlcv(symbol, interval, lookback):
    try:
        okx = ccxt.okx({
            "apiKey": os.getenv("OKX_API_KEY"),
            "secret": os.getenv("OKX_API_SECRET_KEY"),
            "password": os.getenv("OKX_API_PASSPHRASE"),
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        ohlcv = okx.fetch_ohlcv(symbol, interval, limit=lookback)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["date", "open", "high", "low", "close", "volume"]]
        return df
    except Exception as e:
        logging.error(f"❌ Error fetching OHLCV data: {e}")
        return None

# ✅ Preprocess Data for Model
def preprocess_data_for_xgb(df):
    """Prepares data for model inference by calculating ratios."""
    for column in ["open", "high", "low", "close", "volume"]:
        df[f"rolling_mean_7_{column}"] = df[column].rolling(window=7).mean()
        df[f"ratio_{column}"] = df[column] / df[f"rolling_mean_7_{column}"]
    features = ["ratio_low", "ratio_high", "ratio_open", "ratio_close", "ratio_volume"]
    df = df[features].dropna()
    return df.iloc[-1].values.reshape((-1, 5))

# ✅ Cache Timeout Configuration
TIMEOUTS = {
    "1m": 60,       # 1 minute
    "5m": 300,      # 5 minutes
    "30m": 1800,    # 30 minutes
    "1h": 3600,     # 1 hour
    "4h": 14400,    # 4 hours
}

def calculate_next_candle_timeout(interval):
    """Calculate the remaining time until the next expected candle update."""
    now = datetime.now(timezone.utc) + timedelta(hours=9)  # Convert to KST

    if interval == "1m":
        next_time = now + timedelta(minutes=1 - (now.minute % 1))
    elif interval == "5m":
        next_time = now + timedelta(minutes=5 - (now.minute % 5))
    elif interval == "30m":
        next_time = now + timedelta(minutes=30 - (now.minute % 30))
    elif interval == "1h":
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    elif interval == "4h":
        next_time = now.replace(minute=0, second=0, microsecond=0)
        while next_time.hour % 4 != 1:  # Ensure it aligns with 01:00, 05:00, ..., 21:00
            next_time += timedelta(hours=1)
    else:
        return TIMEOUTS[interval]  # Default to existing timeouts

    timeout_seconds = (next_time - now).total_seconds()
    return max(1, int(timeout_seconds))  # Ensure timeout is positive

# for calculating bithumb prediction
def calculate_probabilities_bithumb_shorter_candles(df):
    if df is None or len(df) < 2:
        logging.error("Insufficient data for probability calculations.")
        return None

    # Calculate breakout occurrences
    high_breakouts = (df["high"].shift(-1) > df["high"]).sum()  # Next high > Current high
    low_breakouts = (df["low"].shift(-1) < df["low"]).sum()  # Next low < Current low

    total_candles = len(df) - 1  # We compare each candle to the next one

    upward_prob = (high_breakouts / total_candles) * 100
    downward_prob = (low_breakouts / total_candles) * 100

    total_prob = upward_prob + downward_prob
    if total_prob > 0:
        upward_prob = round((upward_prob / total_prob) * 100.0, 2)
        downward_prob = round((downward_prob / total_prob) * 100.0, 2)

    return upward_prob, downward_prob


# ✅ Fetch XGB predictions
def get_xgb_predictions(interval):
    """Fetch predictions and store in cache aligned with exchange-generated candle times."""
    symbol = "BTC-USDT-SWAP"
    lookback = 300

    data = get_historical_ohlcv(symbol, interval, lookback)
    if data is None or data.empty:
        logging.warning(f"⚠️ No data fetched for {interval}, skipping prediction.")
        return None

    last_updated_kst = (data["date"].iloc[-1] + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST")
    test_input = preprocess_data_for_xgb(data)

    if interval in models:
        pred = models[interval].predict_proba(test_input)[0]
        probabilities = {
            f"prob_{interval}_long": round(pred[0] * 100, 1),
            f"prob_{interval}_short": round(pred[1] * 100, 1),
        }

        bithumb_up, bithumb_down = calculate_probabilities_bithumb_shorter_candles(data)
        bithumb_probabilities = {
            f"bithumb_{interval}_up": bithumb_up,
            f"bithumb_{interval}_down": bithumb_down
        }

        # ✅ Align Cache Expiry with Exchange Candle Timing
        timeout = calculate_next_candle_timeout(interval)

        cache.set(f"btc_xgb_{interval}", probabilities, timeout=timeout)
        cache.set(f"btc_xgb_{interval}_timestamp", last_updated_kst, timeout=timeout)

        cache.set(f"btc_bithumb_{interval}", bithumb_probabilities, timeout=timeout)
        cache.set(f"btc_bithumb_{interval}_timestamp", last_updated_kst, timeout=timeout)

        logging.info(f"✅ Cached predictions for {interval}: {probabilities} | {bithumb_probabilities}")
        return probabilities, bithumb_probabilities
    return None


# ✅ Update predictions only if cache is expired
def update_xgb_predictions(interval):
    """Ensure updates occur exactly when a new candle is generated by the exchange."""
    now_kst = datetime.now(timezone.utc) + timedelta(hours=9)  # Get current time in KST
    now_minute = now_kst.minute
    now_hour = now_kst.hour

    # ✅ Check if cache exists; if missing, perform an immediate update
    cached_data = cache.get(f"btc_xgb_{interval}")
    last_update = cache.get(f"btc_xgb_{interval}_timestamp")

    if not cached_data or not last_update:
        print(f"⚠️ No cache found for {interval} - Performing initial update.")
        get_xgb_predictions(interval)
        return  # Prevents unnecessary scheduling check

    # ✅ Scheduled update time logic
    should_update = False

    if interval == "1m":
        should_update = True  # Every minute
    elif interval == "5m":
        should_update = now_minute % 5 == 0  # Every 5 minutes
    elif interval == "30m":
        should_update = now_minute in [0, 30]  # 00 and 30 minutes past the hour
    elif interval == "1h":
        should_update = now_minute == 0  # Exactly on the hour
    elif interval == "4h":
        should_update = now_hour % 4 == 1 and now_minute == 0  # Exactly at 01:00, 05:00, ..., 21:00

    # ✅ Proceed with update only if it's the right time
    if should_update:
        print(f"✅ Updating {interval} predictions at {now_kst.strftime('%Y-%m-%d %H:%M:%S KST')}")
        get_xgb_predictions(interval)
    else:
        print(f"⏳ Skipping {interval} update, not the right time ({now_kst.strftime('%Y-%m-%d %H:%M:%S KST')})")

# ✅ Fetch Cached Predictions
# ✅ Fetch Cached Predictions with Auto-Update if Missing
def fetch_predictions():
    """Retrieve cached predictions and ensure missing ones are updated immediately."""
    predictions = {}
    timestamps = {}

    for timeframe in ["1m", "5m", "30m", "1h", "4h"]:
        cached_xgb = cache.get(f"btc_xgb_{timeframe}", {})
        last_updated_xgb = cache.get(f"btc_xgb_{timeframe}_timestamp", None)  # Change "N/A" to None for checks

        cached_bithumb = cache.get(f"btc_bithumb_{timeframe}", {})
        last_updated_bithumb = cache.get(f"btc_bithumb_{timeframe}_timestamp", None)

        # ✅ If predictions are missing, trigger an update
        if not cached_xgb or not last_updated_xgb:
            logging.warning(f"⚠️ Cache expired or missing for {timeframe}, fetching new predictions.")
            update_xgb_predictions(timeframe)  # Force update
            cached_xgb = cache.get(f"btc_xgb_{timeframe}", {})  # Fetch again after update
            last_updated_xgb = cache.get(f"btc_xgb_{timeframe}_timestamp", "N/A")  # Ensure it's not None

        if not cached_bithumb or not last_updated_bithumb:
            logging.warning(f"⚠️ Cache expired or missing for Bithumb {timeframe}, fetching new predictions.")
            update_xgb_predictions(timeframe)  # Update Bithumb as well
            cached_bithumb = cache.get(f"btc_bithumb_{timeframe}", {})
            last_updated_bithumb = cache.get(f"btc_bithumb_{timeframe}_timestamp", "N/A")

        logging.info(f"📊 Fetching {timeframe} from cache:")
        logging.info(f"XGB: {cached_xgb}, Last Updated: {last_updated_xgb}")
        logging.info(f"Bithumb: {cached_bithumb}, Last Updated: {last_updated_bithumb}")

        # ✅ Ensure all values are stored as float
        predictions.update({key: float(value) for key, value in cached_xgb.items()})
        predictions.update({key: float(value) for key, value in cached_bithumb.items()})

        timestamps[f"last_updated_xgb_{timeframe}"] = last_updated_xgb
        timestamps[f"last_updated_bithumb_{timeframe}"] = last_updated_bithumb

    return predictions, timestamps


# ✅ API Endpoint for Predictions
def get_xgb_predictions_api(request):
    """API to fetch latest XGB predictions."""
    predictions, timestamps = fetch_predictions()
    return JsonResponse({"predictions": predictions, "timestamps": timestamps})

# ✅ Background task to update all intervals
# ✅ Background Task to Update All Predictions
@background(schedule=60)
def update_all_xgb_predictions():
    """Scheduled task to update all XGB predictions."""
    print(f"🕒 Task started at {datetime.now(timezone.utc) + timedelta(hours=9)} KST")
    for interval in ["1m", "5m", "30m", "1h", "4h"]:
        print(f"🔄 Attempting to update {interval} at {datetime.now(timezone.utc) + timedelta(hours=9)} KST...")
        update_xgb_predictions(interval)
    print(f"✅ Finished updating all predictions at {datetime.now(timezone.utc) + timedelta(hours=9)} KST")


# ✅ Schedule background task at server startup
def schedule_background_tasks():
    """Schedule XGB updates to run every 60 seconds."""
    update_all_xgb_predictions(now=True)  # 🔹 Ensure it runs immediately at startup
    update_all_xgb_predictions(repeat=60, schedule=60)  # 🔹 Schedule periodic updates

# Function to fetch latest news using Selenium
# Set up the webdriver options for headless Chrome
def create_webdriver():
    try:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        logging.error(f"Failed to create WebDriver: {e}")
        return None

def can_click_more(driver):
    try:
        more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="root"]/div/div[2]/div[3]/button')))
        more_button.click()
        return True
    except (Exception):
        return False

def collect_latest_news(driver, limit=30):
    news_data = []
    try:
        while len(news_data) < limit:
            WebDriverWait(driver, 10).until(
                EC.visibility_of_all_elements_located((By.XPATH, '//div[contains(@class, "NewsWrap-sc")]')))
            news_items = driver.find_elements(By.XPATH, '//div[contains(@class, "NewsWrap-sc")]')
            for item in news_items:
                datetime_str = item.find_element(By.XPATH, './/div[contains(@class, "TimeDisplay-sc")]').text.replace('\n', ' ')
                header = item.find_element(By.XPATH, './/h3[contains(@class, "Header-sc")]').text
                content = item.find_element(By.XPATH, './/div[contains(@class, "ContentsWrap-sc")]').text
                news_data.append({'datetime': datetime_str, 'header': header, 'content': content})
                if len(news_data) >= limit:
                    break
            if not can_click_more(driver):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

    aggregated_text = ""
    for news in news_data:
        aggregated_text += f"Title: {news['header']}\nContent: {news['content']}\nDatetime: {news['datetime']}\n\n"

    return aggregated_text

def truncate_text(text, max_tokens, tokenizer):
    """
    Truncates text to fit within the specified max_tokens limit.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]  # Truncate to max_tokens
    return tokenizer.decode(tokens)

def extract_decision(response_text):
    # Extract the analysis part
    analysis_match = re.search(r"분석:\s*(.+)", response_text)
    analysis = analysis_match.group(1).strip() if analysis_match else None

    # Extract the decision part
    decision_match = re.search(r"결정:\s*(0|1)", response_text)
    decision = int(decision_match.group(1)) if decision_match else None

    # Ensure both analysis and decision exist
    if analysis and decision is not None:
        return analysis, decision
    else:
        raise ValueError("Response is not in the required format.")

# Function to generate a response with retry mechanism
def generate_response_with_retry(background_information, max_retries=10):
    for attempt in range(max_retries):
        try:
            # Construct the prompt
            prompt = f"""
            📌 **분석 요청:** 
            아래에 제공된 시장 정보(배경 정보)를 바탕으로 **비트코인 가격이 내일 상승(Long, 0)할지 하락(Short, 1)할지** 분석하고 결정해 주세요.  
            **차트 데이터(시가, 고가, 저가, 종가, 거래량)와 최신 뉴스, 시장 심리 및 거시경제적 요인을 종합적으로 고려하여** 결정을 내려야 하며, 단순 예측이 아닌 **구체적인 논리적 근거**를 포함해야 합니다. 

            📝 **출력 형식 (반드시 이 형식을 따를 것):**  
            분석: [제공된 배경 정보를 바탕으로 BTC 가격 움직임에 대한 구체적 분석]  
            결정: [Long(0) 또는 Short(1)]  

            ✅ **예시 출력:**  
            분석: 트럼프 행정부의 비트코인 전략적 비축 자산 발언과 최근 비트코인의 차트 움직임을 분석했을 때 내일 종가가 오늘 종가에 비해서 높을 확률이 더 크다고 생각합니다.  
            결정: 0  

            🔍 **배경 정보:**  
            {background_information}
            """

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        당신은 비트코인 시장 분석 전문가입니다.  
                        주어진 정보를 종합적으로 분석하여 **BTC 가격이 내일 상승할지(Long, 0) 하락할지(Short, 1) 논리적으로 결정**하세요.  
                        반드시 지정된 형식으로만 답변해야 합니다.
                        """
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the response content
            response_text = response.choices[0].message.content

            # Try to extract the analysis and decision
            analysis, decision = extract_decision(response_text)
            return analysis, decision

        except ValueError as e:
            # Handle parsing errors and retry
            print(f"Attempt {attempt + 1}/{max_retries}: Response format invalid. Retrying...")
            time.sleep(1)  # Optional delay between retries

    # Raise an error if all retries fail
    raise RuntimeError("Failed to get a valid response after multiple retries.")

# ✅ Background Task: Update BTC Analysis at 00:00 UTC
@background(schedule=0)
def update_btc_analysis():
    print("🔄 Running scheduled BTC analysis update...")
    driver = create_webdriver()
    driver.get('https://coinness.com/search?q=BTC&category=news')
    latest_btc_news = collect_latest_news(driver, limit=30)

    btc_chart_data = get_historical_ohlcv("BTC-USDT-SWAP", "1d", 30)
    btc_chart_data = btc_chart_data.to_string(index=False)
    background = f"\n### Bitcoin Price History (Last 30 Days) ###\n{btc_chart_data}\n\n### Latest BTC News ###\n{latest_btc_news}"
    analysis, decision = generate_response_with_retry(background)

    now = datetime.now(timezone.utc)
    now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
    cache_timeout = int((datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1) - now).total_seconds())
    cache.set("btc_analysis_decision",
              {"analysis": analysis,
               "decision": decision,
               "last_updated": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
              },
              timeout=cache_timeout)
    print(f"✅ Updated BTC Analysis at {now} UTC / {now_kst} KST")

## for likes modeling
def get_likes(request):
    """Fetch the current like count."""
    likes = cache.get("likes_count")
    if likes is None:
        like_obj, created = Like.objects.get_or_create(id=1)
        likes = like_obj.count
        cache.set("likes_count", likes, timeout=3600)  # Cache for 1 hour
    return JsonResponse({"likes": likes})

@csrf_protect
def add_like(request):
    """Increase the like count when button is pressed."""
    if request.method == "POST":
        like_obj, created = Like.objects.get_or_create(id=1)
        like_obj.count += 1
        like_obj.save()
        cache.set("likes_count", like_obj.count, timeout=3600)  # Update cache
        return JsonResponse({"likes": like_obj.count})
    return JsonResponse({"error": "Invalid request"}, status=400)

def get_visitors(request):
    """Fetch the cumulative visitor count directly from the database (not cache)."""
    visitor_obj, created = Visitor.objects.get_or_create(id=1)
    return JsonResponse({"visitors": visitor_obj.count})


def index(request):
    """
    Fetches and displays Bitcoin analysis. If cached GPT analysis exists, it is used;
    otherwise, a new analysis is generated. XGBoost price movement predictions are also fetched.
    """
    #########
    for intervals in ["1m","5m","30m","1h","4h"]:
        get_xgb_predictions(intervals)

    schedule_background_tasks()
    # ✅ Fetch cached XGBoost predictions for each timeframe or trigger updates if missing
    cached_predictions, cached_timestamps = fetch_predictions()

    # ✅ Increment visitor count
    visitor_obj, created = Visitor.objects.get_or_create(id=1)
    visitor_obj.count += 1
    visitor_obj.save()

    # ✅ Fetch cached GPT analysis (decision & explanation)
    cached_analysis = cache.get("btc_analysis_decision") or {}

    print("📌 Cached Analysis:", cached_analysis)  # Debugging
    print("📌 Cached Predictions:", cached_predictions)  # Debugging

    # ✅ If GPT analysis exists, use cached data
    if cached_analysis and "analysis" in cached_analysis and "decision" in cached_analysis:
        print("✅ Cache Hit: Using cached GPT analysis and decision.")
        return render(request, "analyzer/main_landing.html", {
            "analysis": cached_analysis.get("analysis", ""),
            "decision": cached_analysis.get("decision", ""),
            "last_updated": cached_analysis.get("last_updated", ""),
            "visitor_count": visitor_obj.count,  # ✅ Ensuring visitor count is passed
            **cached_predictions,
            **cached_timestamps,
        })

    print("⚡ Cache Miss: Generating new GPT analysis immediately.")

    # ✅ Step 2: Generate new analysis **immediately**
    driver = create_webdriver()
    driver.get('https://coinness.com/search?q=BTC&category=news')
    latest_btc_news = collect_latest_news(driver, limit=30)

    btc_chart_data = get_historical_ohlcv("BTC-USDT-SWAP", "1d", 30)
    btc_chart_data = btc_chart_data.to_string(index=False)

    background = f"\n### Bitcoin Price History (Last 30 Days) ###\n{btc_chart_data}\n\n### Latest BTC News ###\n{latest_btc_news}"
    analysis, decision = generate_response_with_retry(background)

    # ✅ Step 3: Cache result with expiration at 00:00 UTC
    now = datetime.now(timezone.utc)
    now_kst = now + timedelta(hours=9)
    cache_timeout = int((datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=1) - now).total_seconds())

    cache.set("btc_analysis_decision", {
        "analysis": analysis,
        "decision": decision,
        "last_updated": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
    }, timeout=cache_timeout)

    # ✅ Step 4: Ensure `update_btc_analysis` runs at 00:00 UTC daily
    midnight_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if not Task.objects.filter(task_name="update_btc_analysis").exists():
        update_btc_analysis(schedule=midnight_utc)
        print("🕒 Scheduled BTC Analysis update for 00:00 UTC daily.")

    print("📌 Final Analysis:", analysis)  # Debugging
    print("📌 Final Decision:", decision)  # Debugging
    print("📌 Cached Predictions (After update):", cached_predictions)  # Debugging

    return render(request, "analyzer/main_landing.html", {
        "analysis": analysis,
        "decision": decision,
        "last_updated": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
        "visitor_count": visitor_obj.count,  # ✅ Added visitor count here
        **cached_predictions,
        **cached_timestamps,
    })

def support_page(request):
    return render(request, "analyzer/support_page.html", {})