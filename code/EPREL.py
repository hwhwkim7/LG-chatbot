import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import warnings
import json


def get_EPREL_DB():
    warnings.filterwarnings("ignore")
    page_num = 1
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    print("data 취합 중....")
    while True:
        url = 'https://eprel.ec.europa.eu/api/products/washingmachines2019?'
        header = {
            # 'accept' : 'application/json, text/plain, */*',
            # 'accept-encoding' : 'gzip, deflate, br',
            # 'accept-language ': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            # 'cache-control' : 'No-Cache',
            # 'content-type' : 'application/json',
            # 'referer': 'https://eprel.ec.europa.eu/screen/product/washingmachines2019',
            # 'sec-ch-ua-mobile' : '?0',
            # 'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
            'x-api-key': '3PR31D3F4ULTU1K3Y2020',
            # 'x-requested-with' : 'XMLHttpRequest'
        }
        param = {'_page': str(page_num),
                 '_limit': 100,
                 'sort0': 'onMarketStartDateTS',
                 'order0': 'DESC',
                 'sort1': 'energyClass',
                 'order1': 'DESC'}
        r = requests.get(url, headers=header, params=param)
        bs = BeautifulSoup(r.text, features="lxml")
        df = pd.read_json(bs.text)
        page = df['size'][0] // 100 + 1
        df = pd.DataFrame(data=df.hits)
        df1 = pd.concat((df1, df), axis=0)
        # df1 = df1.append(df, ignore_index= True)
        print(page_num, ' / ', page)
        if page == page_num:
            break
        page_num += 1
        # time.sleep(0.5)
    print("항목 별 셀로 나누는 중")
    for i in range(len(df1)):
        df2[i] = pd.DataFrame(data=df1.iloc[i, 0].values(), index=df1.iloc[i, 0].keys())

    return df2

def get_EPREL_DB_v2(url_base,url_add):
    warnings.filterwarnings("ignore")
    page_num = 1
    url = url_base + str(page_num) + url_add
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    print("data 취합 중....")
    while True:
        header = {'x-api-key': '3PR31D3F4ULTU1K3Y2020'}
        r = requests.get(url, headers=header)
        bs = BeautifulSoup(r.text, features="lxml")
        df = pd.read_json(bs.text)
        page = df['size'][0] // 100 + 1
        df = pd.DataFrame(data=df.hits)
        df1 = pd.concat((df1, df), axis=0)
        # df1 = df1.append(df, ignore_index= True)
        print(page_num, ' / ', page)
        if page == page_num:
            break
        page_num += 1
        # time.sleep(0.5)
    print("항목 별 셀로 나누는 중")
    for i in range(len(df1)):
        df2[i] = pd.DataFrame(data=df1.iloc[i, 0].values(), index=df1.iloc[i, 0].keys())

    return df2

def get_EPREL_DB_v3():
    warnings.filterwarnings("ignore")
    page_num = 1
    url_WM_Base = 'https://eprel.ec.europa.eu/api/products/washingmachines2019?_page='
    url_WD_Base = 'https://eprel.ec.europa.eu/api/products/washerdriers2019?_page='
    url_TD_Base = 'https://eprel.ec.europa.eu/api/products/tumbledriers?_page='
    url_WM_add = '&_limit=100&sort0=onMarketStartDateTS&order0=DESC&sort1=energyClass&order1=DESC'
    url_WD_add = '&_limit=100&sort0=onMarketStartDateTS&order0=DESC&sort1=energyClassWash&order1=DESC'
    url_TD_add = '&_limit=100&sort0=onMarketStartDateTS&order0=DESC&sort1=energyClass&order1=DESC'

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    print("data 취합 중....")
    while True:
        url = url_WM_Base + str(page_num) + url_WM_add
        header = {'x-api-key': '3PR31D3F4ULTU1K3Y2020'}
        r = requests.get(url, headers=header)
        bs = BeautifulSoup(r.text, features="lxml")
        df = pd.read_json(bs.text)

        data = json.loads(bs.text)

        # 예쁘게 출력
        page = df['size'][0] // 100 + 1
        df = pd.DataFrame(data=df.hits)
        df1 = pd.concat((df1, df), axis=0)
        # df1 = df1.append(df, ignore_index= True)
        print(page_num, ' / ', page)
        if page == page_num:
            break
        page_num += 1
        # time.sleep(0.5)
    print("항목 별 셀로 나누는 중")
    for i in range(len(df1)):
        df2[i] = pd.DataFrame(data=df1.iloc[i, 0].values(), index=df1.iloc[i, 0].keys())

    return df2





