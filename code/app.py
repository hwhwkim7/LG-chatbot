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
    - 자료 출처 : [EPREL](https://eprel.ec.europa.eu/screen/home)
    """
)

import functions

if "vector_db" not in st.session_state:
    # 데이터 로드 및 벡터 저장소 설정
    # CSV 데이터 불러오기
    st.sidebar.write("📑 Processing CSV file...")
    df, des_dict = functions.load_data()

    # 벡터 저장소 로드 또는 업데이트
    functions.set_vectorDB(df)
    st.session_state.vector_db = functions.load_vectorDB()

    st.session_state.des_dict = des_dict

    st.sidebar.success("✅ FAISS 벡터 저장소 로드 완료!")

# ✅ 모델이 없으면 한 번만 로드
if "model" not in st.session_state:
    st.sidebar.write("⏳ 모델 로딩 중...")
    model, tokenizer = functions.load_model()
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.sidebar.success("✅ 모델 로드 완료!")

# Store chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for role, text in st.session_state["messages"]:
    st.chat_message(role).write(text)

# 사용자 입력
query = st.chat_input("📝 질문을 입력하세요:")

# 질문을 입력하면 응답 생성
if query:
    # Store user message
    st.session_state["messages"].append(("user", query))
    st.chat_message("user").write(query)

    with st.spinner("🔍 검색 중..."):
        response = functions.generate_response(query, st.session_state.model, st.session_state.tokenizer)
        # st.success("✅ 응답 생성 완료!")
        # st.write("🤖 **챗봇 응답:**")
        # st.write(response)

    st.session_state["messages"].append(("assistant", response))
    st.chat_message("assistant").write(response)
