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


def extract_year_month(dataframe):
    # 시점 열에서 년도와 월을 추출하는 함수
    def parse_date(s):
        # 연도는 앞의 4글자
        year = s[:4]
        # 월은 그 다음 2글자
        month = s[4:6]

        # 상순, 중순, 하순에 따른 월 조정 (상순: 1, 중순: 2, 하순: 3)
        if "상순" in s:
            day_part = "1"
        elif "중순" in s:
            day_part = "2"
        elif "하순" in s:
            day_part = "3"
        else:
            day_part = "0"  # 알 수 없는 경우

        return year, month, day_part

    # 연도, 월, 상/중/하순을 새로운 컬럼으로 추가
    dataframe['년도'], dataframe['월'], dataframe['순'] = zip(*dataframe['시점'].apply(parse_date))
    dataframe['년도'] = dataframe['년도'].astype('int')
    dataframe['월'] = dataframe['월'].astype('int')
    dataframe['순'] = dataframe['순'].astype('int')

    return dataframe


def sliding_window(data, look_back, n_steps):
    X, y = [], []
    for i in range(len(data) - look_back - n_steps + 1):
        X.append(data[i:(i + look_back)].values.reshape(look_back, -1))
        y.append(data[(i + look_back):(i + look_back + n_steps)])

    return np.array(X, dtype='float32'), np.array(y, dtype='float32')


def add_fourier_features(df, period=12):
    df['month_sin'] = np.sin(2 * np.pi * df['월'].astype('int') / period)
    df['month_cos'] = np.cos(2 * np.pi * df['월'].astype('int') / period)
    return df


def price_log(df, col):
    df[f'log_{col}'] = df[col].apply(lambda x:np.log(x))
    return df


def price_agg(df, target_col='평균가격(원)'):
    agg_df = df.groupby(['월','순'])[target_col].agg(['mean', 'std']).reset_index()
    df = pd.merge(df, agg_df, on=['월','순'], how='left')
    return agg_df, df


def process_time_range(df):
    df['year'] = df['시점'].str[:4]
    df['month'] = df['시점'].str[4:6]
    df['month'] = df['month'].astype(int)

    observed_start = df['year'].iloc[0] + '_' + df['month'].iloc[0].astype(str).zfill(2)  # First observation
    observed_end = df['year'].iloc[-1] + '_' + df['month'].iloc[-1].astype(str).zfill(2)  # Last observation
    last_year = int(df['year'].iloc[-1])
    last_month = df['month'].iloc[-1]

    if last_month == 12:
        predict_year = last_year + 1
        predict_month = 1
    else:
        predict_year = last_year
        predict_month = last_month + 1

    predict_month_str = str(predict_month).zfill(2)

    return observed_start, observed_end, predict_year, predict_month_str


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