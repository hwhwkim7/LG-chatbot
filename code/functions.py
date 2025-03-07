import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
import ast
from transformers import pipeline
from langchain.docstore.document import Document

# 데이터 추가 전처리
def preprocess_data(df, dict):
    df['onMarketStart_Year'] = None
    df['onMarketStart_Month'] = None
    df['onMarketStart_Day'] = None


    for index, value in df['onMarketStartDate'].items():
        if isinstance(value, str):  # 문자열인지 확인
            parsed_value = ast.literal_eval(value)  # 문자열을 리스트로 변환
            # 리스트 길이에 따라 값 할당 (부족한 값은 None)
            year = parsed_value[0] if len(parsed_value) > 0 else None
            month = parsed_value[1] if len(parsed_value) > 1 else None
            day = parsed_value[2] if len(parsed_value) > 2 else None

            # 변환된 값 저장
            df.at[index, 'onMarketStart_Year'] = year
            df.at[index, 'onMarketStart_Month'] = month
            df.at[index, 'onMarketStart_Day'] = day

    # 설명 추가
    dict.update({
    "onMarketStart_Year": "The year this model was released on the market (YYYY)",
    "onMarketStart_Month": "The month this model was released on the market (MM)",
    "onMarketStart_Day": "The day this model was released on the market (DD)"
    })

    return df, dict

# 데이터 로드
def load_data(file_name='../output/pre_data_IF.csv', des_file_name="../data/description.csv"):
    # 컬럼 설명
    des_df = pd.read_csv(des_file_name, header=None, names=['col_name', 'description'])
    des_dict = dict(zip(des_df['col_name'], des_df['description']))
    # main data load
    data_df = pd.read_csv(file_name)
    data_df, des_dict = preprocess_data(data_df, des_dict)

    return data_df, des_dict

# vectorDB 생성
def set_vectorDB(df, embeddings, db_path="vectorstore/db_faiss"):
    # 이미 vectorDB가 존재한다면 load
    if os.path.exists(db_path):
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ FAISS 저장소가 성공적으로 로드됨: {db_path}")
        return vector_store
    else:
        print(f"⚠️ FAISS 로드 실패. 다시 생성합니다.")

    documents = []
    col_info = [
    "ratedCapacity", "energyClass", "energyConsPer100Cycle", "spinClass", 
    "spinSpeedRated", "noiseClass", "noise", "rinsingEffectivenes", "waterCons", 
    "supplierOrTrademark", "onMarketStart_Year"
    ] # 벡터 임베딩할 주요 컬럼
    
    # df의 각 row를 순회
    for _, row in df.iterrows():
        # 여러 열을 하나로 합쳐서 한 행씩 임베딩
        combined_text = ", ".join([f"{col}: {row[col]}" for col in col_info if pd.notna(row[col])])
        # 메타데이터 저장 (modelIdentifier)
        metadata = {"modelIdentifier": row["modelIdentifier"]} if "modelIdentifier" in df.columns else {}
        # 벡터DB에 저장할 문서 생성
        doc = Document(page_content=combined_text, metadata=metadata)
        documents.append(doc) # 생성한 document를 list에 저장

    # FAISS 벡터 저장소 생성 (임베딩을 통해 vectorDB 생성)
    vector_store = FAISS.from_documents(documents, embeddings)

    # FAISS 저장 (디렉토리 생성 후 저장)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    vector_store.save_local(db_path)

    # 저장 확인
    if os.path.exists(db_path):
        print(f"✅ FAISS 저장소가 성공적으로 생성됨: {db_path}")
    else:
        print(f"❌ FAISS 저장소 생성 실패: {db_path}")

    return vector_store


# vectorDB 로드
def load_vectorDB(embeddings, db_faiss_path="vectorstore/db_faiss"):
    # 해당 path가 존재하면 vectorDB 로드
    if os.path.exists(db_faiss_path):
        docsearch = FAISS.load_local(db_faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"❌ 벡터 저장소를 찾을 수 없습니다: {db_faiss_path}")
    return docsearch


# 모델 로드
def load_model(access_token, model_name):
    torch.cuda.empty_cache()  # 캐시된 메모리 강제 해제

    # 4-bit 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    # LLM 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config, # 4bit 양자화 적용
        torch_dtype=torch.float16,
        device_map="auto",
        token=access_token
    )
    torch.cuda.empty_cache()  # 캐시된 메모리 강제 해제

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=access_token
    )

    # Llama 계열은 left-padding 필요
    tokenizer.padding_side = "left"

    # 평가 모드 적용
    model.eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    torch.cuda.empty_cache()  # 캐시된 메모리 강제 해제

    return generator

# 벡터 검색 수행
def search_DB(query, k, df, embeddings, db_faiss_path="vectorstore/db_faiss"):
    # DB 불러오기
    docsearch = load_vectorDB(embeddings, db_faiss_path)

    # 쿼리 임베딩
    query_vector = embeddings.embed_query(query)
    # 벡터DB에서 가장 유사한 k개의 data 검색
    results_with_scores = docsearch.similarity_search_with_score_by_vector(query_vector, k=k)

    # 검색된 모델 ID 목록 추출 (메타데이터 활용)
    model_ids = list(set(res.metadata["modelIdentifier"] for res, _ in results_with_scores))
    # 모델 ID로 원래 데이터프레임에서 모델 전체 정보 가져오기
    matched_rows = df[df["modelIdentifier"].isin(model_ids)].copy()
    # 검색된 정보 포맷팅하여 리스트로 저장
    formatted_results = [
        f"{i + 1}. " + " | ".join([f"{col}: {row[col]}" for col in matched_rows.columns])
        for i, (_, row) in enumerate(matched_rows.iterrows())  # 1부터 시작하도록 설정
    ]

    return formatted_results

# Llama 모델 응답 생성
def generate_response(query, generator, des_dict, df, embeddings, k=50, db_faiss_path="vectorstore/db_faiss"):
    # 쿼리가 비었다면 종류
    if len(query) == 0: return ""

    # 벡터 DB 검색 (검색 결과 없으면 종료)
    search_results = search_DB(query, k, df, embeddings, db_faiss_path)
    if len(search_results) == 0:
        return "검색 결과가 없습니다."
    # 검색 결과 포맷팅
    formatted_results = "\n\n".join([f"{res}" for i, res in enumerate(search_results)])

    # 컬럼 설명 추가
    column_info = "\n".join([f"{key}: {value}" for key, value in des_dict.items()])

    # Llama 모델 프롬프트 생성
    system_prompt = f"""
    You are a friendly chatbot that provides insights on washing machine performance using your knowledge database.
    All user inputs will be in English, and you must always respond in English.
    When asked a question, analyze the relevant columns in your vector database and provide a clear, concise response.
    
    If the user requests a summary or specifications, extract key details from the search results and present them in a structured bullet-point format:
    - Maximum Washing Capacity: Range (ratedCapacity)
    - Spin-Drying Performance: Rating (spinClass)
    - Noise Level: Rating (noiseClass)
    - Energy Efficiency: Rating (energyClass)
    - Rinsing Effectiveness: Score (rinsingEffectiveness)

    Format numerical ranges as "Min~Max" (e.g., "6~9kg"). Adjust specification names based on the column descriptions below instead of using raw column names.
    When answering, provide only the specifications. 
    Do not mention modelIdentifier or list model names.
    Provide a single, well-structured response without unnecessary repetition.

    [Column Descriptions]
    {column_info}
    """

    prompt = f"""
    [System prompt]
    {system_prompt}

    [DB Search]
    {formatted_results}
    
    [User question]
    {query}

    [Answer]
    """

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # 응답 생성 및 적절한 답변이 나오도록 전처리
    response = generator(prompt, max_new_tokens=128, temperature=0.1)
    response = response[0]["generated_text"]
    # 실제 응답만 추출하여 최종 응답 생성
    if "[Answer]" in response:
        response = response.split("[Answer]")[-1].strip()
    if "[System prompt]" in response:
        response = response.split("[System prompt]")[-1].strip()
    if "[Column Descriptions]" in response:
        response = response.split("[Column Descriptions]")[-1].strip()
    if "[DB Search]" in response:
        response = response.split("[DB Search]")[-1].strip()
    if "[User question]" in response:
        response = response.split("[User question]")[-1].strip()
    if not response:
        response = "적절한 답변을 생성할 수 없습니다."

    return response

# .env 파일에 본인의 token 및 api를 저장하고 불러옴
def load_token():
    load_dotenv()
    LLM_token = os.getenv("HUGGINGFACE_TOKEN")
    SERPER_token = os.getenv("SERPER_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CX = os.getenv("GOOGLE_CX")
    return LLM_token, SERPER_token, GOOGLE_API_KEY, GOOGLE_CX
