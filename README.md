# Project Folder Structure

- 📂 **code**
  - 📄 .env : LLaMA에 access 하기 위한 token 저장 (https://kimwoolina.tistory.com/107)
  - 📄 app.py : **chatbot 실행** 
  - 📄 data.py : **크롤링 및 전처리 실행**
  - 📄 EPREL.py : 크롤링 함수
  - 📄 functions.py : chatbot 관련 함수
     - set_vectorDB() --> csv 파일 가져와서 벡터 DB를 생성
     - load_model() --> LLM 모델을 로드
     - query 
     - generate_response()
       - search_DB() --> query 임베딩 후, 유사한 k개의 data 추출
       - 가져온 data LLM에 적절하게 포맷팅
       - 시스템 프롬프트 작성
       - 최종 프롬프트 <-- 시스템 프롬프트 + DB 검색 결과 + query 
       - model.generate <-- LLM이 최종 프롬프트 기반으로 응답을 생성
       - 응답을 디코딩 해서 후처리 + [Answer] 이후만 받을 수 있게끔 후처리
       - ==> 응답 생성 !-!
  - 📄 pre_functions.py : 크롤링 및 전처리 관련 함수
  - 📄 requriement.txt : 활용할 패키지 저장
  - 📄 web_search.py : **가격 검색 실행**
- 📂 **data**
  - 📄 data_origin.csv : 크롤링 결과
  - 📄 description.csv : 컬럼 설명 **(정확한 정보가 맞는지 확인 필요, 직접 넣어줘야 함)**
- 📂 **output**
  - 📄 confusion.csv : confusion matrix 결과
  - 📄 confusion_matrix.png : confusion matrix 시각화
  - 📄 evaluation.csv : 이상감지 모델 평가
  - 📄 pre_data_filter.csv : 수작업으로 진행한 기준치로 이상감지
  - 📄 pre_data_IF.csv : Isolation Forest로 이상감지
  - 📄 pre_data_IQR.csv : IQR로 이상감지
  - 📄 pre_data_LOF.csv : Local Outlier Factor로 이상감지
- 📄 README.md


### 1. 가상환경 설정
```
conda create -n <name> python=3.9
conda activate <name>
```
### 2. 필요한 패키지 다운로드 
```
pip install -r requirements.txt
```
- ModuleNotFoundError 가 발생한다면, 해당 module이 설치되어있지 않다는 것
```
pip install <module_name>
```
### 3. 데이터 크롤링
```
python data.py --craw T
```

#### 크롤링 및 전처리를 한번에 진행하는 코드
```
python data.py
```

### 4. 데이터 전처리
```
# 직접 정한 기준
python data.py --method filter
```
```
# Boxplot
python data.py --method IQR
```
```
# Local outlier factor
python data.py --method LOF
```
```
# Isolation forest
python data.py --method IF
```
- 위의 코드 실행 시 ../output/pre_data_[method].csv 파일 생성
```
# 세 가지 방법 비교
python data.py --method compare
```
- compare 실행 후 ../output/confusion.csv, confusion_matrix.png, evaluation.csv 생성
- png 파일로 시각화 확인 가능, evaluation.csv를 통해 가장 적절한 모델 평가 가능 --> Isolation forest 선정
```
# 가격 정보 추가
python web_search.py
```
### 5. chatbot 실행
```
streamlit run app.py
```

