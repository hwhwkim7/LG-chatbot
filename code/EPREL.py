import requests
import pandas as pd
import warnings

def get_EPREL_DB_v3():
    warnings.filterwarnings("ignore")
    page_num = 1 # 첫 번째 페이지부터 시작

    # 크롤링 하고자 하는 웹 페이지 주소
    url_WM_Base = 'https://eprel.ec.europa.eu/api/products/washingmachines2019?_page='
    # _limit=100: 한 번에 100개 제품을 불러옴
    # 첫 번째 sort: 최신 출시 제품부터 (sort0=onMarketStartDateTS&order0=DESC)
    # 두 번째 sort: 에너지 등급이 높은 제품부터
    url_WM_add = '&_limit=100&sort0=onMarketStartDateTS&order0=DESC&sort1=energyClass&order1=DESC'

    df_list = [] # 여러 page에서 얻은 dataframe을 저장하는 list
    while True:
        # EPREL 사이트에서 페이지를 하나씩 넘겨가면서 원하는 순서대로 정렬
        url = url_WM_Base + str(page_num) + url_WM_add
        header = {'x-api-key': '3PR31D3F4ULTU1K3Y2020'}
        r = requests.get(url, headers=header)
        # 해당 페이지에서 데이터를 가져와서 json 형태로 변환
        df = pd.read_json(r.text)
        page = df['size'][0] // 100 + 1
        df = pd.DataFrame(data=df.hits)
        df_list.append(df)

        if page == page_num: # 더이상 넘길 페이지가 없으면 종료
            break
        page_num += 1

    # json 형태로 구성된 상태
    df1 = pd.concat(df_list, axis=0).reset_index(drop=True)
    # json을 개별 컬럼 형태로 변환
    structured_data = [pd.DataFrame(data=row.values(), index=row.keys()).T for row in df1.iloc[:, 0]]
    df2 = pd.concat(structured_data, ignore_index=True)
    return df2