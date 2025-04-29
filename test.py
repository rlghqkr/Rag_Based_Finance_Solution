# main.py
import streamlit as st
import requests
import os
from googleapiclient.discovery import build
from PyPDF2 import PdfReader
import docx
from io import BytesIO

# --------------------------
# 테마 설정 및 관리
# --------------------------
    
def setup_sidebar():
    st.sidebar.image('./36logo.png',
                     use_container_width=True)
    
    st.sidebar.markdown("""
                        씨이이이이이발 진짜 ㅈ같노
                        """)

    st.sidebar.markdown("""
                        Still workin on it""")
    
    st.sidebar.divider()
    
    with st.sidebar.expander("팀원", expanded=False):
        st.markdown("""
                    🫡박기호의 팀:\n
                        팀원 1: 🐙김태윤
                        팀원 2: 👨‍🦲박현신
                    """)

# --------------------------
# 기능 모듈
# --------------------------
def process_file(uploaded_file):
    """PDF/DOCX/TXT 파일에서 텍스트 추출"""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    return ""

def google_search(query, api_key, cse_id, num=5):
    """Google Custom Search API 호출"""
    service = build("customsearch", "v1", developerKey=api_key)
    return service.cse().list(q=query, cx=cse_id, num=num).execute().get("items", [])

def generate_answer(messages, openai_key, model="gpt-3.5-turbo"):
    """OpenAI Chat Completion 호출"""
    headers = {"Authorization": f"Bearer {openai_key}"}
    payload = {"model": model, "messages": messages}
    resp = requests.post("https://api.openai.com/v1/chat/completions",
                         headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# --------------------------
# 레이아웃 모듈
# --------------------------
def render_initial_view():
    """Perplexity 초기 화면 (중앙 입력창)"""
    st.write("")  # 상단 여백
    st.write("")
    st.write("")
    st.image("36logo.png", width=100)
    st.markdown("## 원하는 것을 말하시오")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        q = st.text_input("", placeholder="여기에 질문을 입력하세요", key="init_q")
        if q:
            st.session_state.first_question = q
            st.session_state.messages.append({"role": "user", "content": q})
            st.experimental_rerun()

def setup_sidebar():
    """채팅 인터페이스용 사이드바"""
    st.sidebar.image("36logo.png", use_container_width=True)
    st.sidebar.header("⚙️ 설정")
    mode = st.sidebar.radio("모드 선택", ["검색 모드", "직접 답변"])
    openai_key = st.sidebar.text_input("OpenAI API 키", type="password")
    google_key = google_cse = None
    if mode == "검색 모드":
        google_key = st.sidebar.text_input("Google API 키", type="password")
        google_cse = st.sidebar.text_input("Google CSE ID", type="password")
    file_uploader = st.sidebar.file_uploader("파일 업로드", type=["pdf", "docx", "txt"])
    if file_uploader:
        st.session_state.file_content = process_file(file_uploader)
        st.sidebar.success("파일 분석 완료!")
    return mode, openai_key, google_key, google_cse

def render_chat_interface():
    """질문/답변을 표시하고 추가 입력을 처리"""
    # 사이드바 표시
    mode, openai_key, google_key, google_cse = setup_sidebar()

    st.title("🔍 AI 검색 엔진")
    # 초기 어시스턴트 메시지
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "안녕하세요! 무엇이든 물어보세요."
        })

    # 채팅 기록 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 새 입력 처리
    if prompt := st.chat_input("추가 질문을 입력하세요..."):
        # 필수 키 검증
        if not openai_key or (mode == "검색 모드" and (not google_key or not google_cse)):
            st.sidebar.error("필수 API 키를 모두 입력해주세요")
            return

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            # 컨텍스트 구성
            context = ""
            if st.session_state.get("file_content"):
                context += f"파일 내용:\n{st.session_state.file_content}\n\n"
            sources = []
            if mode == "검색 모드":
                results = google_search(prompt, google_key, google_cse, num=3)
                for i, item in enumerate(results, 1):
                    context += f"[{i}] {item['title']}\n{item.get('snippet','')}\n\n"
                    sources.append((item['title'], item['link']))

            # OpenAI 호출
            messages = [
                {"role": "system", "content": "Perplexity 스타일로 답변하세요."},
                {"role": "user", "content": f"{prompt}\n\n{context}"}
            ]
            answer = generate_answer(messages, openai_key)
            st.markdown(answer)

            # 소스 표시
            if sources:
                st.markdown("**참고 자료**")
                for title, link in sources:
                    st.markdown(f"- [{title}]({link})")

            # 세션에 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

# --------------------------
# 앱 실행
# --------------------------
def main():
    st.set_page_config(
        page_title="Perplexity Clone",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_content" not in st.session_state:
        st.session_state.file_content = ""
    if "first_question" not in st.session_state:
        render_initial_view()
    else:
        render_chat_interface()

if __name__ == "__main__":
    main()
