import streamlit as st

# Streamlit 페이지 설정
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 세탁기 인사이트 도출 chatbot")

# 사이드바 설명
st.sidebar.title("🔹 About")
st.sidebar.markdown(
    """
    - **Llama-3.2-11B-Vision-Instruct 기반 RAG Chatbot**
    - FAISS 벡터 검색을 활용한 질문 응답 시스템
    - EPREL: https://eprel.ec.europa.eu/screen/home
    """
)

import functions

functions.load_token()

# 데이터 로드 및 벡터 저장소 설정
# CSV 데이터 불러오기
st.sidebar.write("📑 Processing CSV file...")
df, des_dict = functions.load_data()

# 데이터 전처리
functions.pre_process(df)

# 벡터 저장소 생성
functions.set_vectorDB(df)

st.sidebar.success("✅ FAISS 벡터 저장소 생성 완료!")

# Llama 모델 로드
st.sidebar.write("⏳ 모델 로딩 중...")
model, tokenizer = functions.load_model()
st.sidebar.success("✅ 모델 로드 완료!")

# 사용자 입력
query = st.text_input("📝 질문을 입력하세요:", "")

# 질문을 입력하면 응답 생성
if query:
    with st.spinner("🔍 검색 중..."):
        response = functions.generate_response(query, model, tokenizer, des_dict)
        st.success("✅ 응답 생성 완료!")
        st.write("🤖 **챗봇 응답:**")
        st.write(response)