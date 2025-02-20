import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv


# streamlit
def add_vertical_space(spaces=1):
    for _ in range(spaces):
        st.sidebar.markdown("---")


# 데이터 로드
def load_data(file_name='../output/pre_data_IF.csv', des_file_name="../data/description.csv"):
    des_df = pd.read_csv(des_file_name)
    des_df.columns = ['col_name', 'description']
    des_dict = dict(zip(des_df['col_name'], des_df['description']))
    data_df = pd.read_csv(file_name)

    return data_df, des_dict

def set_vectorDB(df, db_faiss_path="vectorstore/db_faiss"):
    # Product Name: Apple | Price: 100 | Rating: 4 | Stock: 50
    df['text'] = df.apply(lambda row: " | ".join(f"{col}: {val}" for col, val in row.items()), axis=1)

    # csv 데이터를 list로
    documents = df['text'].tolist()

    # 문서를 작은 청크로 나누기 (필요에 따라 크기 조정 가능)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.create_documents(documents)

    # 임베딩 모델 적용 (HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # FAISS 벡터 저장소 생성
    docsearch = FAISS.from_documents(text_chunks, embeddings)

    #  로컬 저장소에 저장
    os.makedirs(os.path.dirname(db_faiss_path), exist_ok=True)
    docsearch.save_local(db_faiss_path)

    # 저장 후 존재 여부 확인
    if os.path.exists(db_faiss_path):
        print(f"✅ FAISS 저장소가 성공적으로 생성됨: {db_faiss_path}")
    else:
        print(f"❌ FAISS 저장소 생성 실패: {db_faiss_path}")

    return docsearch


# 벡터 DB 생성 또는 로드
def load_vectorDB(db_faiss_path="vectorstore/db_faiss"):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    if os.path.exists(db_faiss_path):
        docsearch = FAISS.load_local(db_faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"❌ 벡터 저장소를 찾을 수 없습니다: {db_faiss_path}")
    return docsearch


# 모델 로드
def load_model(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    access_token = load_token()

    """4-bit 양자화 모델을 로드하고 파이프라인을 생성하는 함수"""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        token=access_token
    )  # GPU 자동 설정

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

    return model, tokenizer


# 벡터 검색 수행
def search_DB(query, k=20, db_faiss_path="vectorstore/db_faiss"):
    docsearch = load_vectorDB(db_faiss_path)
    results = docsearch.similarity_search(query, k=k)

    formatted_results = []
    extracted_models = set()

    for res in results:
        text = res.page_content
        formatted_results.append(text)

        for line in text.split("|"):
            if "modelIdentifier" in line.lower():
                model_name = line.split(":")[-1].strip()
                extracted_models.add(model_name)

    return formatted_results, extracted_models


# Llama 모델 응답 생성
def generate_response(query, model, tokenizer, k=20, db_faiss_path="vectorstore/db_faiss"):
    # Step 1: 벡터 DB 검색
    search_results, _ = search_DB(query, k, db_faiss_path)

    # Step 2: 검색 결과 포맷팅
    context = "\n".join(search_results) if search_results else "No relevant information found in the database."

    # Step 3: Llama 모델 프롬프트 생성
    system_prompt = """
        You are a friendly chatbot that provides insights on washing machine performance based on your knowledge database.
        All user inputs will always be in Korean, and you must always respond in Korean.
        When a question is asked, analyze the column information in your vector database to determine what should be checked, and provide an appropriate response.
    """

    prompt = f"""
        [System prompt]
        {system_prompt}

        [Context - Internal DB Search]
        {context}

        [User question]
        {query}

        [Answer]
    """

    # Step 4: 디바이스 확인 및 모델 입력
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Step 5: 모델 응답 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1
    )

    # Step 6: 응답 디코딩 및 후처리
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔹 `[Answer]` 이후만 추출하여 최종 응답 생성
    if "[Answer]" in response:
        response = response.split("[Answer]")[-1].strip()

    return response


# Hugging Face 토큰 로드
def load_token():
    load_dotenv()
    access_token = os.getenv("HUGGINGFACE_TOKEN")
    return access_token
