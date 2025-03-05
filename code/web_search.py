import requests
import re

import functions

# .env에서 token load
HUGGINGFACE_TOKEN, _, API_KEY, CX= functions.load_token()
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"

def google_search(query):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": {query}, # 우리가 검색하고자 하는 것
        "key": API_KEY,
        "cx": CX,  # Custom Search Engine ID
        "num": 5, # 검색 결과 5개
        "gl": "us", # us 기준 검색
        "hl": "en" # 언어는 English
    }
    # 검색 후 결과를 json 형태로 추출
    response = requests.get(search_url, params=params)
    results = response.json()

    # 검색 결과 요약 부분(snippet)만 가져오기
    results_list = []
    if "items" in results:
        for item in results["items"]:
            snippet = item.get("snippet", "")
            results_list.append(snippet)
    return results_list if results_list else ""


def call_LLM(query, search_results, instruction):
    # 검색 결과가 list기 때문에 LLM에 적합하게 text 형식으로 변환
    formatted_results = "\n\n".join([f"{i+1}: {res}" for i, res in enumerate(search_results)])

    prompt = f"""
    Below is a list of search results related to "{query}".
    {instruction}

    Search results:
    {formatted_results}

    Answer:
    """

    # 모델 로드
    generator = functions.load_model(HUGGINGFACE_TOKEN, LLAMA)
    # 답변 생성
    result = generator(prompt, max_new_tokens=20, do_sample=True)
    raw_output = result[0]["generated_text"].split("Answer:")[-1].strip()

    return raw_output

def clean_price_response(response):
    """LLM 응답에서 통화 기호 + 숫자 형식('$3000', '€2500')으로 변환"""
    match = re.search(r'(\$|€|£|¥|₩|元)\s?(\d{1,6}(?:\.\d{1,2})?)', response)
    if match:
        return f"{match.group(1)}{match.group(2)}"  # 통화 기호 + 숫자 반환
    return None  # 유효한 형식이 아닐 경우 None 반환

# 전처리된 dataset 로드
df, _ = functions.load_data()
# 현재 API 제한때문에 년도로 분할해서 아래와 같이 조금씩 검색중입니다.
# 최종 코드는 전체 df에 대해서 검색하는 것으로 수정할 예정입니다.
year = 2024
num = 1
df = df[df["onMarketStart_Year"].isin([year])].head(25*num)

# price 열 생성
df['price'] = None
for idx, row in df.iterrows():
    model_name = row.get('modelIdentifier')
    supplier_name = row.get('supplierOrTrademark')
    weight = row.get('ratedCapacity')

    # 1차 검색 (모델명 + price)
    search_results = google_search(f"{model_name} price")
    instruction = (
        "Extract and return the most relevant and latest price from the provided search results. "
        "You MUST return a price if at least one valid price exists in the results. "
        "If multiple prices are found, return the most recent one. "
        "Ensure the format is strictly '$3000' or '€2500'. "
        "Do not use words, commas, or additional symbols. "
        "If no price information is found in the search results, return 'NO_PRICE'. "
        "DO NOT generate a price if no valid price exists in the results."
    )
    # LLM에게 응답 요청
    raw_price = call_LLM(model_name, search_results, instruction)
    # LLM 응답 전처리
    price = clean_price_response(raw_price)

    # 2차 검색 (제조사명 + 무게 기반)
    if not price or price == 'NO_PRICE':
        search_query = f"{supplier_name} {weight}kg price"
        search_results = google_search(search_query)
        raw_price = call_LLM(f"{supplier_name} {weight}kg", search_results, instruction)
        price = clean_price_response(raw_price)

    # 3차 검색 (LLM 추론)
    if not price or price == 'NO_PRICE':
        instruction = (
            "There are no exact prices available for this model. "
            "Estimate the price based on similar models from the same supplier with similar weight. "
            "Ensure the format is strictly '$3000' or '€2500'. "
        )
        raw_price = call_LLM(f"{supplier_name} {weight}kg (similar products)", [], instruction)
        price = clean_price_response(raw_price)

    # 가격 정보 추라
    df.at[idx, "price"] = price
output_path = f'../output/search_data_{year}_{num}.csv'
df.to_csv(output_path, index=False) # 최종 df 저장
