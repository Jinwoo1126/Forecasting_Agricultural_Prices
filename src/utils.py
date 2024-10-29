import os
import random
import torch

import numpy as np
import pandas as pd


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_data(df_1, df_2, target_col='평균가격(원)'):
    df_1.rename({"품목(품종)명": "품목명"}, axis='columns', inplace=True)
    df_1.rename({"등급(특 5% 상 35% 중 40% 하 20%)": "등급"}, axis='columns', inplace=True)
    df_1['품종명'] = None
    df_1.rename({"평년 평균가격(원) Common Year SOON":"평년 평균가격(원)"}, axis='columns', inplace=True)
    df_2['거래단위'] = df_2['유통단계별 단위 '].apply(lambda x: str(int(x))) + df_2['유통단계별 무게 '].apply(lambda x: str(x))
    df_2.rename({"등급명": "등급"}, axis='columns', inplace=True)
    df_2.rename({"평년 평균가격(원) Common Year SOON":"평년 평균가격(원)"}, axis='columns', inplace=True)
    use_col = ['YYYYMMSOON', '품목명', '품종명', '거래단위', '등급', "평년 평균가격(원)"] + [target_col]
    df = pd.concat([df_1[use_col], df_2[use_col]], axis=0).reset_index(drop=True)
    df['거래단위'] = df['거래단위'].replace(" ", "")
    df.rename({"YYYYMMSOON":"시점"}, axis='columns', inplace=True)

    return df


def sliding_window(data, look_back, n_steps):
    X, y = [], []
    for i in range(len(data) - look_back - n_steps + 1):
        X.append(data[i:(i + look_back)].values.reshape(look_back, 1))
        y.append(data[(i + look_back):(i + look_back + n_steps)])

    return np.array(X, dtype='float32'), np.array(y, dtype='float32')


def NMAE(y_pred, y_true):
    mae = torch.mean(torch.abs(y_pred - y_true))
    norm = torch.mean(torch.abs(y_true))
    return mae / norm


def nmae_(week_answer, week_submission):
    answer = week_answer
    target_idx = np.where(answer!=0)
    true = answer[target_idx]
    pred = week_submission[target_idx]
    score = np.mean(np.abs(true-pred)/true)

    return score


def nmae(pred, dataset):
    y_true = dataset.get_label()
    score = nmae_(y_true, pred)

    return 'score', score, False


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)