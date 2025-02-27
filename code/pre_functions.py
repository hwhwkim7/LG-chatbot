from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix, roc_auc_score, jaccard_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tldextract
from collections import Counter, defaultdict
from fuzzywuzzy import fuzz, process
import numpy as np

def filter_standard(df):
    # 필터링 조건 적용
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

def LOF(df, numeric_columns, n_neighbors=2, contamination='auto'):
    # 결측치 처리
    ## 1000개 이상의 결측값을 가지고 있는 경우, 해당 열 제거
    df = df.loc[:, df.isna().sum() < 1000]
    ## 결측값이 있는 행 제거
    df = df.dropna()
    numeric_columns = list(set(numeric_columns) & set(df.columns))

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    X = df[numeric_columns]
    lof_scores = lof.fit_predict(X)  # -1: 이상값, 1: 정상값
    valid_indices = X.index[lof_scores == 1]
    df = df.loc[valid_indices]
    return df


def IF(df, numeric_columns, contamination='auto', random_state=42):
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    X = df[numeric_columns]
    anomaly_scores = iso_forest.fit_predict(X)  # -1: 이상값, 1: 정상값
    df = df[anomaly_scores == 1]
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

    # 결과 출력
    print(f"\n=== {method_name} Evaluation ===")
    print(f"Confusion Matrix:\n{cm}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"Jaccard Similarity: {jaccard_sim:.4f}\n")

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

def compare(df, num_columns):
    df_gt = filter_standard(df)
    df_IQR = IQR(df, num_columns)
    df_LOF = LOF(df, num_columns)
    df_IF = IF(df, num_columns)

    # 전체 데이터에서 각 필터링 방법이 데이터를 유지했는지 여부 (1: 유지됨, 0: 제거됨)
    df['GT_survived'] = df.index.isin(df_gt.index).astype(int)  # GT 기준으로 살아남음
    df['IQR_survived'] = df.index.isin(df_IQR.index).astype(int)  # IQR 기준으로 살아남음
    df['IF_survived'] = df.index.isin(df_IF.index).astype(int)  # Isolation Forest 기준으로 살아남음
    df['LOF_survived'] = df.index.isin(df_LOF.index).astype(int)  # LOF 기준으로 살아남음

    # Confusion Matrix 시각화 (X축: 이상감지 방법명, Y축: Ground Truth)
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
    df_confusion.to_csv(confusion_csv_path, index=False)
    print(f"Confusion Matrix data saved at: {confusion_csv_path}")

    # 평가 결과 저장
    evaluation_results = []
    for method in methods:
        result = evaluate_anomaly_detection(df["GT_survived"], df[method], method_labels[method])
        evaluation_results.append(result)

    # 평가 결과 DataFrame 생성 및 저장
    results_df = pd.DataFrame(evaluation_results)
    evaluation_csv_path = "../output/evaluation.csv"
    results_df.to_csv(evaluation_csv_path, index=False)
    print(f"Evaluation results saved at: {evaluation_csv_path}")

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
    plt.savefig("../output/confusion_matrix.png", dpi=300, bbox_inches='tight')

def changed_name_by_url(df, name_col, url_col, threshold=80):
    df_ = df[[name_col, url_col]]
    names = df_[name_col].unique().tolist()
    name_groups = {}
    group_dict = defaultdict(list)
    group_id = 0
    for i, name in enumerate(names):
        if name in name_groups: continue
        name_groups[name] = group_id
        group_dict[group_id].append(name)
        for j in range(i+1, len(names)):
            other_name = names[j]
            if other_name in name_groups: continue

            similarity = fuzz.ratio(name.lower(), other_name.lower())
            if similarity >= threshold:
                name_groups[other_name] = group_id
                group_dict[group_id].append(other_name)
        group_id += 1
    # 4. 대표 이름 선택 (그룹 내에서 가장 유사한 이름 선택)
    representative_names = {
        group_id: max(names, key=lambda x: sum(fuzz.ratio(x, y) for y in names))
        for group_id, names in group_dict.items()
    }

    # 5. 원래 이름 → 대표 이름 매핑
    name_change_mapping = {name: representative_names[group_id] for name, group_id in name_groups.items()}
    updated_names = {}
    for original_name, new_name in name_change_mapping.items():
        if original_name == new_name:
            continue  # 이름이 변하지 않으면 패스

        # 같은 이름을 가진 데이터들의 URL 비교
        original_urls = set(df_[df_[name_col] == original_name][url_col])
        new_name_urls = set(df_[df_[name_col] == new_name][url_col])

        if original_urls & new_name_urls:  # 공통 URL이 존재하면 변경
            updated_names[original_name] = new_name
            print(f"{original_name} -> {new_name}")  # 변경 로그 출력

    update_set = set()
    for k,v in updated_names.items():
        if k != v:
            print(k,'->',v)
            update_set.add(k)
    print(update_set, len(update_set))
    df[name_col] = df[name_col].replace(updated_names)

    return df


def pre_process(df, method, file_path="../output/pre_data_"):
    expected_columns = [
        "modelIdentifier", "spinSpeedRated", "type", "onMarketStartDate", "spinSpeedHalf", "ratedCapacity", "spinClass", "maxTemperatureRated",
        "maxTemperatureQuarter", "dimensionDepth", "guaranteeDuration", "dimensionHeight", "powerNetworkStandby", "moisture", "spinSpeedQuarter",
        "energyEfficiencyIndex", "noiseClass", "powerStandbyMode", "powerOffMode", "webLinkSupplier", "programmeDurationHalf", "programmeDurationQuarter", "energyConsPerCycle",
        "noise", "waterCons", "energyClass", "programmeDurationRated", "dimensionWidth", "supplierOrTrademark", "energyConsPer100Cycle", "maxTemperatureHalf", "powerDelayStart", "rinsingEffectiveness"
    ]

    df = df[expected_columns]
    print(len(df.columns))

    print('결측치 확인\n', df.isnull().sum())

    num_columns = df.columns[
        df.apply(lambda col: col.dropna().apply(lambda x: str(x).replace('.', '', 1).isdigit()).all())
    ].tolist()
    df[num_columns] = df[num_columns].astype(float)

    if method != "compare":
        if method == "filter":
            df = filter_standard(df)
        elif method == "IQR":
            df = IQR(df, num_columns)
        elif method == "LOF":
            df = LOF(df, num_columns)
        elif method == "IF":
            df = IF(df, num_columns)


        df = changed_name_by_url(df, 'supplierOrTrademark', 'webLinkSupplier')
        print(df)

        df.to_csv(file_path + method + '.csv', index=False)
    else:
        compare(df, num_columns)
