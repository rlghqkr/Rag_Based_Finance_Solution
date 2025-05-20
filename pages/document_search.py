import os
import streamlit as st
import tiktoken
import tempfile, uuid, shutil, time
from loguru import logger
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.document_loaders import Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def tiktoken_len(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))

def load_documents(files):
    temp_dir = tempfile.mkdtemp(prefix="st_upload_")
    docs = []
    try:
        for f in files:
            path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{f.name}")
            with open(path, "wb") as fp:
                fp.write(f.getvalue()); time.sleep(0.01)
            low = path.lower()
            try:
                if low.endswith(".pdf"):
                    loader = PyMuPDFLoader(path)
                    loaded = loader.load()
                    if not loaded or not any(doc.page_content.strip() for doc in loaded):
                        st.error(f"'{f.name}': 텍스트를 추출할 수 없습니다 (이미지/스캔 PDF일 수 있음)")
                        continue
                elif low.endswith(".docx"):
                    loader = Docx2txtLoader(path)
                    loaded = loader.load()
                elif low.endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(path)
                    loaded = loader.load()
                else:
                    st.error(f"지원하지 않는 파일 형식: {f.name}")
                    continue
                if not loaded:
                    st.error(f"빈 문서입니다: {f.name}")
                    continue
                if not all(hasattr(doc, 'page_content') for doc in loaded):
                    st.error(f"잘못된 문서 구조: {f.name}")
                    continue
                docs.extend(loaded)
            except Exception as e:
                logger.error(f"문서 로딩 실패: {f.name} - {str(e)}")
                st.error(f"'{f.name}' 처리 실패: {str(e)}")
                continue
    except Exception as e:
        logger.error("load_documents error: %s", e)
        st.error("파일 처리 중 오류 발생")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return docs

def chunk_documents(docs):
    if not docs:
        st.error("분석 가능한 문서가 없습니다.")
        return []
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,           # 더 작은 청크
            chunk_overlap=100,        # 더 넓은 중첩
            length_function=tiktoken_len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_documents(docs)
    except Exception as e:
        logger.error(f"청킹 실패: {str(e)}")
        st.error("텍스트 분할 실패")
        return []


def make_vectorstore(chunks):
    if not chunks:
        st.error("분할된 텍스트 청크가 없습니다.")
        return None
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vs = FAISS.from_documents(chunks, embeddings)
        if vs is None:
            st.error("벡터 저장소 생성 실패")
        return vs
    except Exception as e:
        logger.error(f"벡터 저장소 생성 실패: {str(e)}")
        st.error("벡터 저장소 생성 실패")
        return None

def make_document_chain(vs):
    if vs is None:
        return None
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True,output_key="answer")
    try:
        retr   = vs.as_retriever(search_kwargs={"k":3})
    except Exception as e:
        logger.error(f"as_retriever 실패: {str(e)}")
        st.error("벡터 저장소에서 검색기를 생성할 수 없습니다.")
        return None
    mv     = st.session_state.model_version
    if mv=="GEMINI":
        llm = ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEY, model=st.session_state.gemini_model,
            temperature=st.session_state.temperature, top_p=st.session_state.top_p,
            max_output_tokens=st.session_state.max_tokens,
            system_instruction="You are a helpful financial assistant."
        )
    else:
        model_name = "gpt-4" if mv=="GPT-4" else "gpt-3.5-turbo"
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model_name=model_name,
            temperature=st.session_state.temperature, top_p=st.session_state.top_p,
            frequency_penalty=st.session_state.frequency_penalty,
            presence_penalty=st.session_state.presence_penalty,
            max_tokens=st.session_state.max_tokens
        )
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retr, memory=memory,
            return_source_documents=True, verbose=False
        )
        return chain
    except Exception as e:
        logger.error(f"대화 체인 생성 실패: {str(e)}")
        st.error("AI 체인 생성 실패")
        return None

def render_document_search():
    st.header("📄 금융 문서 분석")
    st.session_state.setdefault("doc_chain", None)
    st.session_state.setdefault("doc_msgs", [{"role":"assistant","content":"문서를 업로드・처리해주세요."}])
    st.session_state.setdefault("doc_ready", False)

    uploaded = st.file_uploader(
        "PDF/DOCX/PPTX 업로드", type=["pdf","docx","pptx"],
        accept_multiple_files=True, key="doc_upload"
    )

    if st.button("문서 처리", disabled=not uploaded):
        with st.spinner("문서 처리 중..."):
            docs  = load_documents(uploaded)
            if not docs:
                return
            chunks = chunk_documents(docs)
            if not chunks:
                return
            vs    = make_vectorstore(chunks)
            if vs is None:
                return
            chain = make_document_chain(vs)
            if chain is None:
                return
            st.session_state.doc_chain  = chain
            st.session_state.doc_ready  = True
            st.success(f"{len(docs)}개 문서, {len(chunks)}개 청크 준비 완료.")

    # 대화 UI
    for m in st.session_state.doc_msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if st.session_state.doc_ready:
        q = st.chat_input("문서에 대해 질문하세요…")
        if q:
            st.session_state.doc_msgs.append({"role":"user","content":q})
            with st.chat_message("assistant"):
                try:
                    res = st.session_state.doc_chain({"question":q})
                    st.write(res["answer"])
                    for i,doc in enumerate(res.get("source_documents", []),1):
                        src  = doc.metadata.get("source","unknown")
                        page = doc.metadata.get("page","?")
                        st.markdown(f"> 출처{i}: (page{page})")
                    st.session_state.doc_msgs.append({"role":"assistant","content":res["answer"]})
                except Exception as e:
                    st.error(f"답변 생성 오류: {e}")
