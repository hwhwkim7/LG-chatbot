from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score, jaccard_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from fuzzywuzzy import fuzz, process
from urllib.parse import urlparse


def filter_standard(df):
    # 필터링 조건 적용 + 결측값이 있는 경우 삭제 안함
    if "spinSpeedRated" in df.columns:
        df = df.loc[df["spinSpeedRated"].between(200, 2000) | df["spinSpeedRated"].isna()]
    if "ratedCapacity" in df.columns:
        df = df.loc[((df["ratedCapacity"] % 0.5 == 0) & (df["ratedCapacity"] < 30)) | df["ratedCapacity"].isna()]
    if "powerNetworkStandby" in df.columns:
        df = df.loc[(df["powerNetworkStandby"] < 5) | df["powerNetworkStandby"].isna()]
    if "powerStandbyMode" in df.columns:
        df = df.loc[(df["powerStandbyMode"] < 3) | df["powerStandbyMode"].isna()]
    if "powerOffMode" in df.columns:
        df = df.loc[(df["powerOffMode"] < 3) | df["powerOffMode"].isna()]
    if "energyConsPerCycle" in df.columns:
        df = df.loc[(df["energyConsPerCycle"] < 50) | df["energyConsPerCycle"].isna()]
    if "waterCons" in df.columns:
        df = df.loc[(df["waterCons"] < 120) | df["waterCons"].isna()]
    return df

# Boxplot IQR
def IQR(df, numeric_columns):
    outlier_indices = set()
    for col in numeric_columns:
        if col in df.columns:
            col_data = df[col].dropna()  # IQR 계산 시 NaN 제외 (하지만 최종적으로 제거하지 않음)
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:  # IQR이 0이면 이상치 판별 불가 → 스킵
                continue
            # 이상치에 해당하는 인덱스 추가 (NaN이 있는 행은 포함하지 않음)
            outlier_indices |= set(df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].index)
    # 이상치가 있는 경우만 drop 실행 (NaN이 포함된 행은 삭제되지 않음)
    if outlier_indices:
        df = df.drop(index=list(outlier_indices))
    return df

# Local Outlier Factor
def LOF(df, numeric_columns, n_neighbors=2, contamination='auto'):
    # 결측치 처리
    ## 1000개 이상의 결측값을 가지고 있는 경우, 해당 열 제거
    df = df.loc[:, df.isna().sum() < 1000]
    ## 결측값이 있는 행 제거
    df = df.dropna()
    # 결측값 때문에 제거된 열일 가능성이 있기 때문에 갖고 있는 수치 값이 있는 열과의 교집합에 해당하는 열만 활용하고자 함
    numeric_columns = list(set(numeric_columns) & set(df.columns))

    # n_neighbors개의 주변 이웃 데이터를 보고 이상치를 판별
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    X = df[numeric_columns] # 수치 데이터에 대해서만 적용
    lof_scores = lof.fit_predict(X)  # -1: 이상값, 1: 정상값
    valid_indices = X.index[lof_scores == 1] # 정상값을 가지고 있는 행의 index 추출
    df = df.loc[valid_indices] # 추출한 index에 해당하는 data를 추출
    return df

# Isolation Forest
def IF(df, numeric_columns, contamination='auto', random_state=42):
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    X = df[numeric_columns] # 수치 데이터에 대해서만 적용
    anomaly_scores = iso_forest.fit_predict(X)  # -1: 이상값, 1: 정상값
    df = df[anomaly_scores == 1] # 정상값에 해당하는 data를 추출
    return df

# 평가 함수 정의
def evaluate_anomaly_detection(y_true, y_pred, method_name):
    """
    Confusion Matrix 및 주요 평가 지표를 계산하여 해석하는 함수
    y_true: GT 기준 필터링 결과 (0: 제거됨, 1: 유지됨)
    y_pred: 이상감지 모델이 필터링한 결과 (0: 제거됨, 1: 유지됨)
    method_name: 평가할 이상감지 방법 이름
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Precision, Recall, F1-score 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # AUC-ROC Score
    auc_score = roc_auc_score(y_true, y_pred)

    # Jaccard Similarity
    jaccard_sim = jaccard_score(y_true, y_pred)

    return {
        "Method": method_name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "AUC-ROC": auc_score,
        "Jaccard Similarity": jaccard_sim
    }

# 네가지 전처리 방식 비교
def compare(df, num_columns):
    # 전처리 데이터 추출
    df_gt = filter_standard(df)
    df_IQR = IQR(df, num_columns)
    df_LOF = LOF(df, num_columns)
    df_IF = IF(df, num_columns)

    # 전체 데이터에서 각 필터링 방법이 데이터를 유지했는지 여부 (1: 유지됨, 0: 제거됨)
    df['GT_survived'] = df.index.isin(df_gt.index).astype(int)  # GT 기준으로 살아남음
    df['IQR_survived'] = df.index.isin(df_IQR.index).astype(int)  # IQR 기준으로 살아남음
    df['IF_survived'] = df.index.isin(df_IF.index).astype(int)  # Isolation Forest 기준으로 살아남음
    df['LOF_survived'] = df.index.isin(df_LOF.index).astype(int)  # LOF 기준으로 살아남음

    # Confusion Matrix 시각화할 때 편의를 위한 label 설정 (X축: 이상감지 방법명, Y축: Ground Truth)
    method_labels = {
        "IQR_survived": "IQR",
        "IF_survived": "Isolation Forest",
        "LOF_survived": "Local Outlier Factor"
    }

    # Confusion Matrix 재계산 (올바른 키값 사용 확인)
    methods = list(method_labels.keys())
    # Confusion Matrix 데이터 저장
    df_confusion = df[['GT_survived', 'IQR_survived', 'IF_survived', 'LOF_survived']]
    confusion_csv_path = "../output/confusion.csv"
    df_confusion.to_csv(confusion_csv_path, index=False) # csv 파일로 저장

    # 평가 결과 저장
    evaluation_results = []
    for method in methods:
        # 수작업으로 진행한 결과에 비해 이상탐지 결과가 어땠는지를 수치적으로 판단하는 함수
        result = evaluate_anomaly_detection(df["GT_survived"], df[method], method_labels[method])
        evaluation_results.append(result)

    # 평가 결과 DataFrame 생성 및 저장
    results_df = pd.DataFrame(evaluation_results)
    evaluation_csv_path = "../output/evaluation.csv"
    results_df.to_csv(evaluation_csv_path, index=False)

    # Confusion Matrix 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, method in zip(axes, methods):
        cm = confusion_matrix(df["GT_survived"], df[method])
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=['Removed (0)', 'Survived (1)'],
                    yticklabels=['Removed (0)', 'Survived (1)'], ax=ax)
        ax.set_title(f'Confusion Matrix: {method_labels[method]} vs Ground Truth')
        ax.set_xlabel(method_labels[method])  # X축: 해당 이상감지 방법명
        ax.set_ylabel('Ground Truth')  # Y축: Ground Truth

    plt.tight_layout()
    plt.savefig("../output/confusion_matrix.png", dpi=300, bbox_inches='tight') # confusion matrix 저장

def changed_name_by_url(df, name_col, url_col, threshold=80):
    # 제조사명과 url 정보가 담긴 열로만 이루어진 df 추출
    df_ = df[[name_col, url_col]]
    # 제조사명 중복 제거
    names = df_[name_col].unique().tolist()

    # 유사한 이름을 가진 제조사들을 그룹화
    name_groups = {} # 제조사명 : 그룹 아이디
    group_dict = defaultdict(list) # 그룹 아이디 : [유사한 이름을 가진 제조사 list]
    group_id = 0
    # 모든 이름을 순회하며 유사한 제조사 이름끼리 그룹화
    for i, name in enumerate(names):
        if name in name_groups: continue
        # 새로운 그룹을 만들고 해당 이름을 추가
        name_groups[name] = group_id
        group_dict[group_id].append(name)
        # 뒷 순서에 있는 이름과 현재 이름을 비교
        for j in range(i+1, len(names)):
            other_name = names[j] # 뒷 순서의 이름
            if other_name in name_groups: continue
            # 유사도를 비교 (lower()로 둘 다 소문자로 만든 후 유사도 판별)
            similarity = fuzz.ratio(name.lower(), other_name.lower())
            if similarity >= threshold: # 기준을 넘으면 유사한 것으로 판별하고 현재 이름과 동일한 그룹에 추가
                name_groups[other_name] = group_id
                group_dict[group_id].append(other_name)
        group_id += 1

    # 각 그룹에서 대표 이름을 선택 (그룹 내에서 다른 이름들과 가장 높은 총 유사도를 가지는 이름)
    representative_names = {
        group_id: max(names, key=lambda x: sum(fuzz.ratio(x, y) for y in names))
        for group_id, names in group_dict.items()
    }

    # 원래 이름을 대표 이름으로 매핑
    name_change_mapping = {name: representative_names[group_id] for name, group_id in name_groups.items()}
    updated_names = {}
    # 원래 이름과 대표 이름을 비교하여 변경할지 여부를 판단
    for original_name, new_name in name_change_mapping.items():
        if original_name == new_name:
            continue  # 이름이 변하지 않으면 패스
        # 유사한 이름을 가진 데이터들의 url set 추출
        original_urls = set(df_[df_[name_col] == original_name][url_col])
        new_name_urls = set(df_[df_[name_col] == new_name][url_col])
        # 두 이름이 한번이라도 동일한 URL을 가지고 있으면 변경 대상에 포함
        if original_urls & new_name_urls:  # 공통 URL이 존재하면 변경
            updated_names[original_name] = new_name
    # 데이터프레임의 이름을 변경된 값으로 대체
    df[name_col] = df[name_col].replace(updated_names)

    return df

def pre_process(df, method, file_path="../output/pre_data_"):
    # 유효한 열
    expected_columns = [
        "modelIdentifier", "spinSpeedRated", "type", "onMarketStartDate", "spinSpeedHalf", "ratedCapacity", "spinClass", "maxTemperatureRated",
        "maxTemperatureQuarter", "dimensionDepth", "guaranteeDuration", "dimensionHeight", "powerNetworkStandby", "moisture", "spinSpeedQuarter",
        "energyEfficiencyIndex", "noiseClass", "powerStandbyMode", "powerOffMode", "webLinkSupplier", "programmeDurationHalf", "programmeDurationQuarter", "energyConsPerCycle",
        "noise", "waterCons", "energyClass", "programmeDurationRated", "dimensionWidth", "supplierOrTrademark", "energyConsPer100Cycle", "maxTemperatureHalf", "powerDelayStart", "rinsingEffectiveness"
    ]
    # 유효한 열만 추출하여 dataframe 형성
    df = df[expected_columns]

    # 수치 데이터로만 이루어진 열 판별 후 float으로 변환
    num_columns = df.columns[
        df.apply(lambda col: col.dropna().apply(lambda x: str(x).replace('.', '', 1).isdigit()).all())
    ].tolist()
    df[num_columns] = df[num_columns].astype(float)

    # 전처리 옵션 선택
    if method != "compare":
        if method == "filter": # 수작업으로 정한 기준에 맞는 행만 추출
            df = filter_standard(df)
        elif method == "IQR": # IQR 이상탐지에서 정상으로 판별된 행만 추출
            df = IQR(df, num_columns)
        elif method == "LOF": # Local Outlier Factor 이상탐지에서 정상으로 판별된 행만 추출
            df = LOF(df, num_columns)
        elif method == "IF": # Isolation Forest 이상탐지에서 정상으로 판별된 행만 추출
            df = IF(df, num_columns)

        # 이상탐지가 완료된 data에서 제조사명 전처리
        df = changed_name_by_url(df, 'supplierOrTrademark', 'webLinkSupplier')
        # 모든 전처리가 완료된 dataframe 저장
        df.to_csv(file_path + method + '.csv', index=False)
    else:
        # 네 가지 전처리 방식 비교
        compare(df, num_columns)
