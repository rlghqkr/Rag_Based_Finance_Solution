import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.chains import RetrievalQA
import yfinance as yf
import tempfile, uuid, shutil, time
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import re
import json

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 설문 UI ---
def portfolio_survey():
    st.markdown("### 📝 내 주식 현황 설문")
    sector = st.selectbox("선호 업종", ["IT/테크", "헬스케어", "금융", "에너지", "소비재", "기타"])
    risk = st.radio("투자 성향", ["안정형", "중립형", "공격형"])
    period = st.selectbox("예상 투자 기간", ["1년 미만", "1~3년", "3~5년", "5년 이상"])
    region = st.multiselect("관심 국가", ["한국", "미국", "중국", "일본", "유럽", "기타"], default=["한국","미국"])
    tickers = st.text_input("주요 투자 종목(티커, 콤마로 구분)", placeholder="예: AAPL, TSLA, 005930.KS")
    amount = st.slider("총 투자금(만원)", 100, 10000, 1000, 100)
    return {
        "sector": sector,
        "risk": risk,
        "period": period,
        "region": region,
        "tickers": tickers,
        "amount": amount
    }

def get_portfolio_description(survey):
    desc = (
        f"선호 업종: {survey['sector']}\n"
        f"투자 성향: {survey['risk']}\n"
        f"투자 기간: {survey['period']}\n"
        f"관심 국가: {', '.join(survey['region'])}\n"
        f"주요 투자 종목(티커): {survey['tickers']}\n"
        f"총 투자금: {survey['amount']}만원"
    )
    return desc

# --- yfinance 차트/정보 Tool ---
def get_stock_info_and_plot(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        name = info.get('longName', ticker)
        df = yf.Ticker(ticker).history(period='6mo')
        if df.empty:
            return f"'{ticker}' 데이터가 없습니다."
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(df.index, df['Close'], label='종가')
        ax.set_title(name)
        ax.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        latest = df['Close'].iloc[-1]
        summary = info.get('longBusinessSummary', '')
        return f"![{name}](data:image/png;base64,{img_base64})\n\n**현재가:** {latest:,.2f}\n\n{summary}"
    except Exception as e:
        return f"{ticker} 데이터 조회 오류: {e}"

# --- 문서 업로드 및 벡터스토어 ---
def load_documents(files):
    temp_dir = tempfile.mkdtemp(prefix="st_upload_")
    docs = []
    try:
        for f in files:
            path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{f.name}")
            with open(path, "wb") as fp:
                fp.write(f.getvalue())
            low = path.lower()
            if low.endswith(".pdf"):
                docs += PyPDFLoader(path).load_and_split()
            elif low.endswith(".docx"):
                docs += Docx2txtLoader(path).load()
            elif low.endswith(".pptx"):
                docs += UnstructuredPowerPointLoader(path).load()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return docs

@st.cache_resource(ttl="1h")
def get_vectorstore(files):
    if not files:
        return None
    docs = load_documents(files)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def rag_search(query: str, vectorstore, llm) -> str:
    if not vectorstore:
        return "참고 문서가 없습니다."
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    return chain.run(query)

# --- 포트폴리오 분석 Tool ---
def analyze_portfolio(survey, llm) -> str:
    desc = get_portfolio_description(survey)
    prompt = f"""아래는 사용자의 주식 포트폴리오 설문 결과입니다.
{desc}
이 조건에 맞는 최적의 포트폴리오(종목, 비중, 국가, 업종 등)를 추천하고,
추천 이유, 리스크 요인, 분산 효과, 업종별 전망도 자세히 설명해줘.
포트폴리오는 마크다운 표로, 설명은 자연어로 출력해줘. JSON, 리스트, 코드블록 등은 출력하지 마."""
    return llm.predict(prompt)

# --- 마크다운 표 파싱 및 파이차트 시각화 ---
def extract_markdown_table(answer):
    # 마크다운 표 추출 (|로 시작하는 줄이 있으면)
    lines = answer.splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        if "|" in line:
            table_lines.append(line)
            in_table = True
        elif in_table and line.strip() == "":
            break
    if not table_lines:
        return None, answer
    table_md = "\n".join(table_lines)
    pre = answer.split(table_md)[0].strip()
    post = answer.split(table_md)[1].strip() if table_md in answer else ""
    return table_md, pre + "\n\n" + post

def plot_portfolio_pie_from_md(table_md):
    import pandas as pd
    from io import StringIO

    try:
        # 마크다운 표에서 구분선(---) 행을 제거
        lines = table_md.strip().splitlines()
        # 첫 줄: 헤더, 두 번째 줄: 구분선(---), 나머지: 데이터
        if len(lines) >= 2 and set(lines[1].replace('|', '').strip()) <= {'-'}:
            lines = [lines[0]] + lines[2:]
        # 혹시 중간에 또 ---가 있으면 모두 제거
        clean_lines = [line for line in lines if not (set(line.replace('|', '').strip()) <= {'-'})]
        clean_table_md = "\n".join(clean_lines)

        # DataFrame 변환
        df = pd.read_csv(StringIO(clean_table_md), sep="|", engine="python")
        # 첫 번째 빈 컬럼 자동 제거 (마크다운 표 특성)
        df = df.loc[:, ~df.columns.str.strip().eq("")]
        df = df.reset_index(drop=True)

        # weight 또는 비중 컬럼 찾기
        weight_col = None
        for c in df.columns:
            if "weight" in c.lower() or "비중" in c:
                weight_col = c
        if not weight_col:
            st.warning("비중(weight) 컬럼을 찾을 수 없습니다.")
            return

        labels = df[df.columns[0]].astype(str)
        sizes = df[weight_col].astype(float)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    except Exception as e:
        st.warning("포트폴리오 파싱/시각화 실패: " + str(e))

def render_agentic_rag_tab():
    st.header("🤖 Agentic RAG: 개인화 포트폴리오 추천")
    survey = portfolio_survey()

    # 문서 업로드
    uploaded_docs = st.file_uploader(
        "시장분석 PDF/DOCX/PPTX 업로드 (선택)", type=["pdf","docx","pptx"], accept_multiple_files=True
    )
    vectorstore = get_vectorstore(uploaded_docs) if uploaded_docs else None

    # GPT-4.0 LLM 고정
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0.3)

    # Tool 정의 (영문명 필수)
    tools = [
        Tool(
            name="portfolio_analysis",
            func=lambda x: analyze_portfolio(survey, llm),
            description="Analyze the user's stock survey and recommend an optimal portfolio."
        ),
        Tool(
            name="stock_chart",
            func=lambda ticker: get_stock_info_and_plot(ticker),
            description="Get stock chart and info by ticker symbol (e.g., 'AAPL', '005930.KS')."
        )
    ]
    if vectorstore:
        tools.append(
            Tool(
                name="market_report_search",
                func=lambda q: rag_search(q, vectorstore, llm),
                description="Search uploaded market analysis documents."
            )
        )

    # Agent 초기화
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, memory=memory
    )

    # 포트폴리오 추천 버튼 & 결과 시각화
    st.markdown("#### 📊 내게 맞는 포트폴리오 추천")
    if st.button("포트폴리오 추천받기"):
        with st.spinner("AI가 개인화 포트폴리오를 추천 중..."):
            try:
                answer = agent.tools[0].func("")  # portfolio_analysis tool 직접 호출
                table_md, explanation = extract_markdown_table(answer)
                st.success("추천 포트폴리오 결과:")
                if explanation:
                    st.markdown(explanation)
                if table_md:
                    st.markdown(table_md)
                    plot_portfolio_pie_from_md(table_md)
            except Exception as e:
                st.error(f"추천 오류: {e}")

    # 채팅 UI
    st.markdown("#### 💬 포트폴리오/시장/주가에 대해 자유롭게 질문하세요!")
    if "agentic_msgs" not in st.session_state:
        st.session_state.agentic_msgs = [
            {"role": "assistant", "content": "설문 작성 후, 포트폴리오 추천을 받거나 자유롭게 질문해보세요."}
        ]

    for msg in st.session_state.agentic_msgs:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_q = st.chat_input("예: 삼성전자 차트, 시장 전망, 내 포트폴리오 요약 등")
    if user_q:
        st.session_state.agentic_msgs.append({"role":"user","content":user_q})
        with st.chat_message("assistant"):
            context = f"설문 결과:\n{get_portfolio_description(survey)}\n질문:\n{user_q}"
            with st.spinner("에이전트가 답변 중..."):
                try:
                    answer = agent.run(context)
                except Exception as e:
                    answer = f"오류: {e}"
            st.write(answer)
        st.session_state.agentic_msgs.append({"role":"assistant","content":answer})
