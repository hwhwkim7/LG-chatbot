import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings

# Streamlit 페이지 설정
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 세탁기 인사이트 도출 chatbot")

# 사이드바 설명
st.sidebar.title("🔹 About")
st.sidebar.markdown(
    """
    - **Llama-3.1-8B-Instruct 기반 RAG Chatbot**
    - FAISS 벡터 검색을 활용한 질문 응답 시스템
    - 자료 출처 : [EPREL](https://eprel.ec.europa.eu/screen/home)
    """
)

import functions
import torch

# 🔥 강제적으로 모든 GPU 캐시 삭제
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# 🔥 전체 GPU 메모리 할당 해제
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
print("✅ CUDA 캐시 메모리 해제 완료!")
print(f"✅ 현재 사용 중인 GPU ID: {torch.cuda.current_device()}")
print(f"✅ 현재 사용 중인 GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"✅ 현재 GPU에서 예약된 메모리: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# load token
LLM_token, SERPER_token, _, _ = functions.load_token()

# vectorDB load 여부 확인
if "vector_db" not in st.session_state:
    # CSV 데이터 불러오기
    st.sidebar.write("📑 Processing CSV file...")
    df, des_dict = functions.load_data()

    # 임베딩 모델 load
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name='intfloat/e5-large-v2',
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 벡터 저장소 로드
    st.session_state.vector_db = functions.set_vectorDB(df, st.session_state.embeddings)
    st.session_state.df = df
    st.session_state.des_dict = des_dict
    st.sidebar.success("✅ FAISS 벡터 저장소 로드 완료!")

# LLM 모델 load 여부 확인
if "model" not in st.session_state:
    st.sidebar.write("⏳ 모델 로딩 중...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    generator= functions.load_model(
        LLM_token,
        model_name = model_name
    )
    st.session_state.generator = generator
    st.sidebar.success("✅ 모델 로드 완료!")

# chat 기록 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 웹 페이지에 chat 기록 보여주기
for role, text in st.session_state["messages"]:
    st.chat_message(role).write(text)

# 사용자 입력칸
query = st.chat_input("📝 질문을 입력하세요")

# 질문을 입력하면 응답 생성
if query:
    # 메시지 저장
    st.session_state["messages"].append(("user", query))
    st.chat_message("user").write(query)

    with st.spinner("🔍 검색 중..."):
        # 응답 생성
        response = functions.generate_response(query, st.session_state.generator, st.session_state.des_dict, st.session_state.df, st.session_state.embeddings)

    st.session_state["messages"].append(("assistant", response))
    st.chat_message("assistant").write(response)
