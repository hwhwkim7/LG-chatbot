from EPREL import *
import pre_functions
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--craw', default='all')
parser.add_argument('--method', default='filter')
args = parser.parse_args()

# 크롤링만 진행
if args.craw == 'T':
    # EPREL.py의 get_EPREL_DB_v3() 함수 호출하여 크롤링 진행
    df = get_EPREL_DB_v3()
    # 크롤링으로 얻은 dataframe csv로 저장
    df.T.to_csv('../data/data_original.csv', index=False)

# 전처리만 진행
elif args.craw == 'F':
    # 이미 저장했던 크롤링 csv data 가져오기
    df = pd.read_csv('../data/data_original.csv')

    # output 폴더가 있는지 확인하고 없으면 폴더를 생성
    # (전처리한 data가 output 폴더 내에 저장되는데, 해당 폴더가 없으면 오류나기 때문에 이런 과정을 진행)
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 경로 (data.py 파일의 위치 파악)
    parent_dir = os.path.dirname(current_dir) # 현재 파일의 부모 경로 (data.py 파일이 들어있는 code 폴더의 위치 파악)
    output_path = os.path.join(parent_dir, "output") # 부모 경로에서의 output 폴더 존재 확인 (code 폴더와 같은 위치에 output 폴더가 있는지 확인)
    if not os.path.exists(output_path): # output 폴더 없으면
        os.makedirs(output_path) # 폴더 생성

    # 전처리 진행
    pre_functions.pre_process(df, args.method)

# 크롤링 + 전처리
elif args.craw == 'all':
    # output 폴더가 있는지 확인하고 없으면 폴더를 생성
    # (전처리한 data가 output 폴더 내에 저장되는데, 해당 폴더가 없으면 오류나기 때문에 이런 과정을 진행)
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 경로 (data.py 파일의 위치 파악)
    parent_dir = os.path.dirname(current_dir) # 현재 파일의 부모 경로 (data.py 파일이 들어있는 code 폴더의 위치 파악)
    output_path = os.path.join(parent_dir, "output") # 부모 경로에서의 output 폴더 존재 확인 (code 폴더와 같은 위치에 output 폴더가 있는지 확인)
    if not os.path.exists(output_path): # output 폴더 없으면
        os.makedirs(output_path) # 폴더 생성

    # EPREL.py의 get_EPREL_DB_v3() 함수 호출하여 크롤링 진행
    df = get_EPREL_DB_v3()
    # 크롤링으로 얻은 dataframe csv로 저장
    df.T.to_csv('../data/data_original.csv', index=False)

    # 전처리 진행
    pre_functions.pre_process(df.T, 'IF', "../output/pre_data_")
