# pages/stock_search.py

import os
import random
import re
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import platform
from loguru import logger
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# .env에서 API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

# ── 한글 폰트 자동 설정 ──
def set_korean_font():
    plt.rcParams['axes.unicode_minus'] = False
    sys = platform.system().lower()
    if sys == 'darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif sys == 'windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        if os.path.exists(path):
            from matplotlib import font_manager as fm
            plt.rcParams['font.family'] = fm.FontProperties(fname=path).get_name()
        else:
            plt.rcParams['font.family'] = 'sans-serif'
set_korean_font()
plt.style.use('ggplot')

# 대표 종목 매핑 및 키워드
KR_STOCK_MAP = {
    '삼성전자':   '005930.KS',
    'SK하이닉스': '000660.KS',
    '네이버':     '035420.KS',
    '카카오':     '035720.KS',
    'LG화학':     '051910.KS',
    '현대차':     '005380.KS',
}
STOCK_KEYWORDS = ['주가','가격','시세','티커','주식','종목','차트','정보','뉴스']
TICKER_REGEX   = re.compile(r'^[A-Za-z0-9\.\-]+$')

def is_valid_ticker(sym: str) -> bool:
    try:
        info = yf.Ticker(sym).info
        return bool(info.get('symbol'))
    except:
        return False

def extract_ticker_and_name(query: str):
    q = query.strip()
    # 1) 티커 형식이면 yfinance에서 회사명 추출
    if TICKER_REGEX.match(q) and is_valid_ticker(q):
        ticker = q.upper()
        info   = yf.Ticker(ticker).info
        name   = info.get('longName') or info.get('shortName') or ticker
        return ticker, name
    # 2) 한국 대표 종목명 매핑
    for name, tk in KR_STOCK_MAP.items():
        if name in query:
            return tk, name
    # 3) yfinance 검색
    try:
        info   = yf.Ticker(query).info
        ticker = info.get('symbol')
        name   = info.get('longName') or info.get('shortName') or query
        return ticker, name
    except:
        return None, query

def plot_stock(ticker: str, name: str, period: str='6mo'):
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        st.error(f"⚠️ '{name}'({ticker}) 데이터가 없습니다.")
        return None, None

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df.index, df['Close'], label='종가', color='tab:blue')
    ax.plot(df['Close'].rolling(20).mean(), '--', label='20일 이동평균')
    ax.set_title(name, pad=10)
    ax.set_ylabel("가격")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    latest = df['Close'].iloc[-1]
    prev   = df['Close'].iloc[-2] if len(df)>1 else latest
    change = latest - prev
    pct    = (change/prev*100) if prev else 0
    vol    = yf.Ticker(ticker).history(period='1d')['Volume'].iloc[-1]
    metrics = {
        "현재가":       f"{latest:,.2f}",
        "전일 대비":    f"{change:+,.2f} ({pct:+.2f}%)",
        "거래량":       f"{int(vol):,}주",
        "52주 최고가": f"{df['High'].max():,.2f}",
        "52주 최저가": f"{df['Low'].min():,.2f}",
    }
    return fig, metrics

def get_company_summary(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get('longBusinessSummary') or ""
    except:
        return ""

def get_llm_response(query: str) -> str:
    mv = st.session_state.model_version
    if mv == "GEMINI":
        llm = ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEY,
            model=st.session_state.gemini_model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            max_output_tokens=st.session_state.max_tokens,
            system_instruction="You are a helpful financial assistant."
        )
        return llm.predict(query)
    else:
        model = "gpt-4" if mv=="GPT-4" else "gpt-3.5-turbo"
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            frequency_penalty=st.session_state.frequency_penalty,
            presence_penalty=st.session_state.presence_penalty,
            max_tokens=st.session_state.max_tokens
        )
        # simple chat
        return llm.predict(query)

def render_stock_search():
    st.subheader("📈 주식 정보")

    st.session_state.setdefault("stock_msgs", [
        {"role":"assistant",
         "content":"회사명·티커 입력 시 차트·지표·요약을, 일반 질문은 AI로 답변합니다."}
    ])
    st.session_state.setdefault("init_shown", False)

    # 초기 랜덤 차트
    if not st.session_state.init_shown:
        name, ticker = random.choice(list(KR_STOCK_MAP.items()))
        fig, metrics = plot_stock(ticker, name)
        summary = get_company_summary(ticker)
        if fig:
            st.pyplot(fig)
            cols = st.columns(len(metrics))
            for col,(lbl,val) in zip(cols, metrics.items()):
                col.metric(lbl, val)
            if summary:
                st.markdown("**기업 요약**")
                st.write(summary)
        st.session_state.init_shown = True

    # 이전 메시지
    for msg in st.session_state.stock_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 사용자 입력
    query = st.chat_input("회사명/티커/일반 질문 입력")
    if not query:
        return

    st.session_state.stock_msgs.append({"role":"user","content":query})
    with st.chat_message("assistant"):
        ticker, name = extract_ticker_and_name(query)
        if ticker:
            fig, metrics = plot_stock(ticker, name)
            summary = get_company_summary(ticker)
            if fig:
                st.pyplot(fig)
                cols = st.columns(len(metrics))
                for col,(lbl,val) in zip(cols, metrics.items()):
                    col.metric(lbl, val)
                if summary:
                    st.markdown("**기업 요약**")
                    st.write(summary)
            response = f"'{name}'({ticker}) 정보를 표시했습니다."
        else:
            try:
                response = get_llm_response(query)
            except Exception as e:
                response = f"AI 응답 오류: {e}"
        st.markdown(response)
        st.session_state.stock_msgs.append({"role":"assistant","content":response})
