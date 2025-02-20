import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
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

# data
def load_data(file_name='../data/data_original.csv', des_file_name="../data/description.csv"):
    des_df = pd.read_csv(des_file_name)
    des_df.columns = ['col_name', 'description']
    des_dict = dict(zip(des_df['col_name'], des_df['description']))
    data_df = pd.read_csv(file_name)

    return data_df, des_dict

def pre_process(df):
    df.fillna("", inplace=True)

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

def load_vectorDB(db_faiss_path="vectorstore/db_faiss"):
    # set_vectorDB()와 동일한 임베딩 모델 사용
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # 저장된 벡터 저장소가 있는지 확인 후 로드
    if os.path.exists(db_faiss_path):
        docsearch = FAISS.load_local(db_faiss_path, embeddings)
    else:
        raise FileNotFoundError(f"❌ 벡터 저장소를 찾을 수 없습니다: {db_faiss_path}")
    return docsearch

# model
def load_model(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"):
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
        device_map="auto") # gpu 자동 설정

    # tokenizer: 자연어를 모델이 이해할 수 있는 숫자로 변환하는 도구
    # "Hello, how are you?" → ["Hello", ",", "how", "are", "you", "?"] → [1234, 567, 98, 456, 789, 25]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

### 튜닝 ###
def search_DB(query, des_dict, k=3, db_faiss_path="vectorstore/db_faiss"):
    # 저장된 벡터 저장소 로드
    docsearch = load_vectorDB(db_faiss_path)

    # 유사한 문서 검색 (k: 검색할 개수)
    results = docsearch.similarity_search(query, k=k)

    # 검색된 결과를 컬럼 설명과 함께 반환
    formatted_results = []
    for res in results:
        text = res.page_content
        formatted_results.append(text)

    # 컬럼 설명을 상단에 추가하여 LLM이 검색된 데이터를 이해할 수 있도록 함
    column_info = "\n".join([f"- {col}: {desc}" for col, desc in des_dict.items()])
    final_text = f"### Column Descriptions:\n{column_info}\n\n### Retrieved Data:\n" + "\n".join(formatted_results)

    # 검색된 결과 출력
    return final_text

### 튜닝 ###
def generate_response(query, model, tokenizer, des_dict, k=3, db_faiss_path="vectorstore/db_faiss"):
    # DB에서 검색
    search_results = search_DB(query, des_dict, k, db_faiss_path)

    # 검색된 문서들을 하나의 컨텍스트로 결합
    context = "\n".join(search_results)

    # Llama 모델에 전달할 프롬프트 생성
    prompt = f"""
        [Context]
        {context}
        
        [User question]
        {query}
        
        [Answer]
    """

    # 프롬프트를 토큰화하여 모델 입력
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    # 모델이 답변 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1
    )

    # 생성된 토큰을 다시 텍스트로 변환
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def load_token():
    # `.env` 파일에서 Hugging Face 토큰 로드
    load_dotenv()
    access_token = os.getenv("HUGGINGFACE_TOKEN")
