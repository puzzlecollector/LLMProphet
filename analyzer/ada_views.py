from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from django.core.cache import cache
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator
from django.utils.timezone import now
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.decorators.http import require_GET
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
    # Extract 시황 요약
    summary_match = re.search(r"시황 요약:\s*(.+?)\s*심리 판단:", response_text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else None

    # Extract 심리 판단
    sentiment_match = re.search(r"심리 판단:\s*(.+?)\s*결정:", response_text, re.DOTALL)
    sentiment = sentiment_match.group(1).strip() if sentiment_match else None

    # Extract 결정
    decision_match = re.search(r"결정:\s*(0|1)", response_text)
    decision = int(decision_match.group(1)) if decision_match else None

    # Ensure all parts exist
    if summary and sentiment and decision is not None:
        # analysis_combined = f"시황 요약: {summary}. 심리 판단: {sentiment}"
        return summary, sentiment, decision
    else:
        raise ValueError("Response is not in the required format.")

# Function to generate a response with retry mechanism
def generate_response_with_retry(background_information, max_retries=10):
    for attempt in range(max_retries):
        try:
            # Construct the prompt
            prompt = f"""
            📌 **분석 요청:**  
            아래에 제공된 에이다(ADA) 관련 뉴스를 바탕으로 **중요한 시황을 핵심 이슈 중심으로 자세히 정리**해 주세요.  
            그 후, 전체적인 분위기에서 **상승(Long) 심리가 강한지, 하락(Short) 심리가 강한지 판단**하고  
            마지막에 **ADA 가격이 내일 상승(Long, 0)할지 하락(Short, 1)할지를 결정**해 주세요.

            ❗️**주의:**  
            - 정보가 많더라도 **핵심적인 시황을 충분히 설명하면서 요약**해 주세요. (불필요한 잡음은 제거)  
            - 시황 요약은 **단순 나열이 아니라, 논리적으로 흐름 있게 요약**해 주세요.  
            - 단순 예측이 아닌, **생태계 개발, 프로젝트 확장성, 파트너십, 매크로 트렌드 등 다면적 요소를 기반으로** 분석해야 합니다.  
            - 마지막에 반드시 **Long(0) 또는 Short(1)** 형식으로만 결정 결과를 적어주세요.

            📝 **출력 형식 (반드시 이 형식을 따를 것):**  
            시황 요약: [시장 전반의 주요 뉴스 및 요소들을 디테일하게 요약한 문단]  
            심리 판단: [Long 심리가 우세한지, Short 심리가 우세한지, 그리고 그 이유]  
            결정: [Long(0) 또는 Short(1)]

            ✅ **예시 출력:**  
            시황 요약: 카르다노 생태계가 최근 DeFi 및 스마트 컨트랙트 관련 프로젝트 확장을 이어가고 있으며, 특히 Lace 지갑 출시와 함께 사용성 증가가 관찰되고 있습니다. 또한 아프리카 기반 프로젝트들과의 협업이 주목받고 있으며, 온체인 거래량과 TVL 수치도 상승세를 보이고 있습니다.  
            심리 판단: 기술적 진보와 실사용 확대로 인해 투자자 심리가 점진적으로 개선되고 있으며, 장기적 전망에 대한 기대감이 형성되고 있습니다. 롱 심리가 우세합니다.  
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
                        당신은 카르다노(에이다) 전문가입니다.  
                        주어진 정보에서 **핵심 시황을 요약**하고, 전체적인 시장 심리 분석 후 ADA가 내일 상승할지(Long, 0) 하락할지(Short, 1) 결정하세요.  
                        반드시 지정된 형식만 사용하여 출력해야 합니다.
                        """
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the response content
            response_text = response.choices[0].message.content

            # Try to extract the analysis and decision
            summary, sentiment, decision = extract_decision(response_text)
            return summary, sentiment, decision

        except ValueError as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Response format invalid. Retrying...")
            time.sleep(1)

    raise RuntimeError("Failed to get a valid response after multiple retries.")

def update_ada_analysis():
    print("🔄 Running scheduled ADA analysis update...")
    driver = create_webdriver()
    driver.get('https://coinness.com/search?q=ADA&category=news')
    latest_ada_news = collect_latest_news(driver, limit=30)

    background = f"### Latest ADA News ###\n{latest_ada_news}"
    summary, sentiment, decision = generate_response_with_retry(background)

    now = datetime.now(timezone.utc)
    now_kst = datetime.now(timezone.utc) + timedelta(hours=9)
    cache.set("ada_analysis_decision",
              {"summary": summary,
               "sentiment": sentiment,
               "decision": decision,
               "last_updated": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
              },
              timeout=3600)
    print(f"✅ Updated ADA Analysis at {now} UTC / {now_kst} KST")

@require_GET
def get_ada_analysis(request):
    cached = cache.get("ada_analysis_decision")
    if cached:
        return JsonResponse({
            "summary": cached.get("summary"),
            "sentiment": cached.get("sentiment"),
            "decision": cached.get("decision"),
            "last_updated": cached.get("last_updated")
        })
    update_ada_analysis()
    return JsonResponse({
        "loading":True,
        "message": "분석 로딩 중입니다. 잠시 후 다시 시도해 주세요."
    })


def calculate_indicators(df):
    def compute_rsi(series, period):
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, min_periods=1).mean()
        ema_slow = series.ewm(span=slow, min_periods=1).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, min_periods=1).mean()
        return macd, signal_line

    def compute_adx(df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = high.diff()
        minus_dm = low.diff()
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        return adx

    def compute_cci(df, period=14):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma) / (0.015 * mad)
        return cci

    def compute_atr(df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    # 실제 지표 계산
    df["rsi"] = compute_rsi(df["close"], 14)
    df["macd"], df["macd_signal"] = compute_macd(df["close"])
    df["adx"] = compute_adx(df)
    df["cci"] = compute_cci(df, 14)
    df["roc"] = df["close"].pct_change(periods=12) * 100
    df["willr"] = (df["high"].rolling(14).max() - df["close"]) / (
        df["high"].rolling(14).max() - df["low"].rolling(14).min()) * -100
    df["atr"] = compute_atr(df, 14)

    df.dropna(inplace=True)
    return df

@require_GET
def get_ada_technical_analysis(request):
    cached = cache.get("ada_technical_analysis")
    if cached:
        return JsonResponse(cached)
    df = get_historical_ohlcv("ADA/USDT", "1h", 300)
    if df is None:
        return JsonResponse({"error": "데이터를 불러오는 데 실패했습니다."}, status=500)

    df = calculate_indicators(df)

    # 지표 요약
    summary_data = []
    ma_summary_data = []

    # 기술적 지표 해석
    latest = df.iloc[-1]

    def decision(val, low, high):
        if val < low:
            return "Sell"
        elif val > high:
            return "Buy"
        else:
            return "Neutral"

    summary_data.append(
        {"name": "RSI(14)", "value": round(latest["rsi"], 2), "action": decision(latest["rsi"], 30, 70)})
    macd_val = round(latest["macd"], 2)
    macd_signal = round(latest["macd_signal"], 2)
    summary_data.append(
        {"name": "MACD(12,26)", "value": macd_val, "action": "Buy" if macd_val > macd_signal else "Sell"})
    summary_data.append(
        {"name": "ADX(14)", "value": round(latest["adx"], 2), "action": "Sell" if latest["adx"] > 25 else "Neutral"})
    summary_data.append(
        {"name": "CCI(14)", "value": round(latest["cci"], 2), "action": decision(latest["cci"], -100, 100)})
    summary_data.append(
        {"name": "ROC", "value": round(latest["roc"], 2), "action": "Buy" if latest["roc"] > 0 else "Sell"})
    summary_data.append({"name": "Williams %R", "value": round(latest["willr"], 2),
                         "action": "Oversold" if latest["willr"] < -80 else "Neutral"})
    summary_data.append({"name": "ATR(14)", "value": round(latest["atr"], 2),
                         "action": "High Volatility" if latest["atr"] > 10 else "Less Volatility"})

    buy_count = sum(1 for i in summary_data if i["action"] == "Buy")
    sell_count = sum(1 for i in summary_data if i["action"] == "Sell")
    if sell_count >= 5:
        indicator_summary = "Strong Sell"
    elif buy_count >= 5:
        indicator_summary = "Strong Buy"
    elif buy_count > sell_count:
        indicator_summary = "Buy"
    elif sell_count > buy_count:
        indicator_summary = "Sell"
    else:
        indicator_summary = "Neutral"

    # 이동 평균 해석
    close_price = latest["close"]
    ma_periods = [5, 10, 20, 50, 100, 200]
    buy_count = 0
    sell_count = 0
    for period in ma_periods:
        sma = df["close"].rolling(window=period).mean().iloc[-1]
        ema = df["close"].ewm(span=period, adjust=False).mean().iloc[-1]
        sma_action = "Buy" if close_price > sma else "Sell"
        ema_action = "Buy" if close_price > ema else "Sell"
        if sma_action == "Buy":
            buy_count += 1
        else:
            sell_count += 1
        if ema_action == "Buy":
            buy_count += 1
        else:
            sell_count += 1

        ma_summary_data.append({
            "name": f"MA{period}",
            "sma": round(sma, 2),
            "sma_action": sma_action,
            "ema": round(ema, 2),
            "ema_action": ema_action,
        })

    if sell_count >= 10:
        ma_summary = "Strong Sell"
    elif buy_count >= 10:
        ma_summary = "Strong Buy"
    elif buy_count > sell_count:
        ma_summary = "Buy"
    elif sell_count > buy_count:
        ma_summary = "Sell"
    else:
        ma_summary = "Neutral"

    result = {
        "last_updated": (now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST"),
        "indicator_summary": indicator_summary,
        "indicators": summary_data,
        "ma_summary": ma_summary,
        "moving_averages": ma_summary_data
    }

    # 캐시 저장
    cache.set("ada_technical_analysis", result, timeout=3600)
    return JsonResponse(result)


def index(request):
    return render(request, "analyzer/ada_page.html", {})

def support_page(request):
    return render(request, "analyzer/support_page.html", {})