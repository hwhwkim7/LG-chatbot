# Project Folder Structure

- 📂 **code**
  - 📄 .env : LLaMA에 access 하기 위한 token 저장 (https://kimwoolina.tistory.com/107)
  - 📄 app.py : **chatbot 실행** 
  - 📄 data.py : **크롤링 및 전처리 실행**
  - 📄 EPREL.py : 크롤링 함수
  - 📄 functions.py : chatbot 관련 함수
  - 📄 pre_functions.py : 크롤링 및 전처리 관련 함수
  - 📄 requriement.txt : 활용할 패키지 저장
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
### 3. 데이터 크롤링
```
python data.py --craw T
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
### 5. chatbot 실행
```
streamlit run app.py
```



```
pip list | grep -E 'streamlit|pandas|transformers|torch|langchain|faiss|dotenv'
```
```
ctransformers               0.2.27
faiss-gpu                   1.7.2
langchain                   0.3.19
langchain-community         0.3.18
langchain-core              0.3.37
langchain-experimental      0.3.4
langchain-text-splitters    0.3.6
pandas                      2.1.4
python-dotenv               1.0.1
sentence-transformers       3.4.1
streamlit                   1.42.1
streamlit-camera-input-live 0.2.0
streamlit-card              1.0.2
streamlit-chat              0.1.1
streamlit-embedcode         0.1.2
streamlit-extras            0.5.5
streamlit-faker             0.0.3
streamlit-image-coordinates 0.1.9
streamlit-keyup             0.3.0
streamlit-toggle-switch     1.0.2
streamlit-vertical-slider   2.5.5
torch                       2.3.1+cu118
torchaudio                  2.3.1+cu118
torchvision                 0.18.1+cu118
transformers                4.49.0

```

```
streamlit run app.py --server.enableWebsocketCompression=false
```

