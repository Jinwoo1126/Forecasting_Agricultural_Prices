import numpy as np
import pandas as pd

from scipy.stats import poisson

from src.utils import extract_year_month


class Fluctuation_Probability:
    def __init__(self, df, config):
        self.df = df
        self.items = df['품목명'].unique()
        self.k = config['fluctuation_probability']['k']

    def get_fluctuation_probability(self):
        prob_dict = {}
        for item in self.items:
            prob_dict[item] = {}

            target = self.df[self.df['품목명'] == item]

            
            target['년도'] = target['시점'].str[:4]
            target['월'] = target['시점'].str[4:6]
            target['분기'] = target['월'].astype(int).apply(lambda x: (x - 1) // 3 + 1)

            target['이전_월_가격'] = target['평균가격(원)'].shift(3)  
            target['20%_이상_상승'] = ((target['평균가격(원)'] - target['이전_월_가격']) / target['이전_월_가격'] >= 0.2).astype(int)  

            monthly_counts = target.groupby('월')['20%_이상_상승'].sum()  
            monthly_counts_total = target.groupby('월').size()  
            monthly_lambda_risep = monthly_counts / monthly_counts_total  

            prob_dict[item]['risep'] = monthly_lambda_risep

            target['이전_월_가격'] = target['평균가격(원)'].shift(3)  
            target['20%_이상_하락'] = ((target['평균가격(원)'] - target['이전_월_가격']) / target['이전_월_가격'] <= -0.2).astype(int)  

            monthly_counts = target.groupby('월')['20%_이상_하락'].sum()  
            monthly_counts_total = target.groupby('월').size()  
            monthly_lambda_dropp = monthly_counts / monthly_counts_total  

            prob_dict[item]['dropp'] = monthly_lambda_dropp

            monthly_counts = target.groupby('월').apply(lambda x: (x['평균가격(원)'] > x['평년 평균가격(원)']).sum())
            monthly_counts_total = target.groupby('월').size()

            monthly_lambda = monthly_counts / monthly_counts_total

            quarterly_counts = target.groupby('분기').apply(lambda x: (x['평균가격(원)'] > x['평년 평균가격(원)']).sum())
            quarterly_counts_total = target.groupby('분기').size()

            quarterly_lambda = quarterly_counts / quarterly_counts_total

            prob_dict[item]['high'] = monthly_lambda
            prob_dict[item]['high_quarter'] = quarterly_lambda
        
        return prob_dict
    

def fe_prob(df, item, prob_dict, t=None):
    df_ = df.copy()
    df_['price_diff'] = df_['평년 평균가격(원)'] - df_['평균가격(원)']

    # 가격 변화율 피처 생성
    for i in range(1, 9):
        df_[f'price_pct_{i}'] = df_['평균가격(원)'].pct_change(periods=i)

    # 평균가격을 로그 변환
    df_['평균가격(원)'] = df_['평균가격(원)'].apply(lambda x: np.log1p(x))

    # 평년 평균 가격 삭제
    df_.drop(['평년 평균가격(원)'], axis='columns', inplace=True)

    # 과거 가격 정보(시차) 및 이동 통계 추가
    for i in range(1, 9):
        df_[f'price_lag{i}'] = df_['평균가격(원)'].shift(i)
    for i in range(2, 9):
        df_[f'price_{i}_mean'] = df_['평균가격(원)'].rolling(window=i).mean()
        df_[f'price_{i}_std'] = df_['평균가격(원)'].rolling(window=i).std()

    # 타겟 값 생성 (t개월 후 평균 가격)
    if t:
        df_[f'target_price_{t}'] = df_['평균가격(원)'].shift(-t)

    # 확률 계산을 위한 월 추출
    df_['월'] = df_['시점'].str[4:6]
    df_['분기'] = df_['월'].astype(int).apply(lambda x: (x - 1) // 3 + 1)

    k = 2
    monthly_prob = {month: 1 - poisson.cdf(k - 1, lamb) for month, lamb in prob_dict[item]['risep'].items()}
    df_['월별_20%_상승_확률_k=2'] = df_['월'].map(monthly_prob)

    monthly_prob = {month: 1 - poisson.cdf(k - 1, lamb) for month, lamb in prob_dict[item]['dropp'].items()}
    df_['월별_20%_하락_확률_k=2'] = df_['월'].map(monthly_prob)

    monthly_prob = {month: 1 - poisson.cdf(k - 1, lamb) for month, lamb in prob_dict[item]['high'].items()}
    df_['월별_평년_이상_확률_k=2'] = df_['월'].map(monthly_prob)

    quarterly_prob = {quarter: 1 - poisson.cdf(k - 1, lamb) for quarter, lamb in prob_dict[item]['high_quarter'].items()}
    df_['분기별_평년_이상_확률_k=2'] = df_['분기'].map(quarterly_prob)

    df_['월'] = df_['월'].astype(int)    

    # 불필요한 열 삭제
    df_.drop(['품종명'], axis='columns', inplace=True)

    return df_.dropna().reset_index(drop=True)


def fe_event(df, item, t=None):
    df_ = df.copy()

    # 가격 차이
    df_['price_diff'] = df_['평년 평균가격(원)'] - df_['평균가격(원)']
    df_['price_diff_abs'] = df_['price_diff'].abs()

    # 가격 변화율
    for i in range(1, 9):
        df_[f'price_pct_{i}'] = df_['평균가격(원)'].pct_change(periods=i)

    # 로그 변환
    df_['평균가격(원)'] = df_['평균가격(원)'].apply(lambda x: np.log1p(x))

    # 이전 가격 값(lag)
    for i in range(1, 9):
        df_[f'price_lag{i}'] = df_['평균가격(원)'].shift(i)

    # 연월 추출
    df_ = extract_year_month(df_)
    df_['season'] = df_['월'].apply(lambda x: (x % 12 + 3) // 3)  # 계절을 숫자로 표현 (봄=1, 여름=2, 가을=3, 겨울=4)
    df_['fourier_sin'] = np.sin(2 * np.pi * df_['월'] / 12)
    df_['fourier_cos'] = np.cos(2 * np.pi * df_['월'] / 12)

    # # 이벤트 관련 피처(기존처럼 필요시 사용 가능)
    if item == '감자 수미':
        df_.loc[df_['월'] == 1, 'event'] = 1
        df_.loc[df_['월'] == 2, 'event'] = 1
        df_.loc[df_['event'].isnull(), 'event'] = 0
    elif item == '대파(일반)':
        df_.loc[df_['월'] == 2, 'event'] = 1
        df_.loc[df_['월'] == 3, 'event'] = 1
        df_.loc[df_['월'] == 4, 'event'] = 1
        df_.loc[df_['월'] == 5, 'event'] = 1
        df_.loc[df_['월'] == 8, 'event'] = 1
        df_.loc[df_['월'] == 9, 'event'] = 1
        df_.loc[df_['월'] == 10, 'event'] = 1
        df_.loc[df_['event'].isnull(), 'event'] = 0
    elif item == '무':
        df_.loc[df_['월'] == 3, 'event'] = 1
        df_.loc[df_['월'] == 5, 'event'] = 1
        df_.loc[df_['월'] == 7, 'event'] = 1
        df_.loc[df_['월'] == 8, 'event'] = 1
        df_.loc[df_['월'] == 9, 'event'] = 1
        df_.loc[df_['월'] == 10, 'event'] = 1
        df_.loc[df_['event'].isnull(), 'event'] = 0
    elif item == '배':
        df_.loc[df_['월'] == 6, 'event'] = 1
        df_.loc[df_['월'] == 7, 'event'] = 1
        df_.loc[df_['월'] == 8, 'event'] = 1
        df_.loc[df_['event'].isnull(), 'event'] = 0
    elif item == '배추':
        df_.loc[df_['월'] == 3, 'event'] = 1
        df_.loc[df_['월'] == 5, 'event'] = 1
        df_.loc[df_['월'] == 6, 'event'] = 1
        df_.loc[df_['월'] == 7, 'event'] = 1
        df_.loc[df_['월'] == 8, 'event'] = 1
        df_.loc[df_['event'].isnull(), 'event'] = 0
    elif item == '사과':
        df_.loc[df_['월'] == 8, 'event'] = 1
        df_.loc[df_['월'] == 9, 'event'] = 1
        df_.loc[df_['event'].isnull(), 'event'] = 0
    elif item == '상추':
        df_.loc[df_['월'] == 6, 'event'] = 1
        df_.loc[df_['월'] == 7, 'event'] = 1
        df_.loc[df_['월'] == 8, 'event'] = 1
        df_.loc[df_['월'] == 9, 'event'] = 1
        df_.loc[df_['event'].isnull(), 'event'] = 0
    elif item == '양파':
        df_.loc[df_['월'] == 1, 'event'] = 1
        df_.loc[df_['월'] == 2, 'event'] = 1
        df_.loc[df_['월'] == 3, 'event'] = 1
        df_.loc[df_['월'] == 4, 'event'] = 1
        df_.loc[df_['월'] == 10, 'event'] = 1
        df_.loc[df_['월'] == 11, 'event'] = 1
        df_.loc[df_['월'] == 12, 'event'] = 1
        df_.loc[df_['event'].isnull(), 'event'] = 0

    if 'event' in df.columns:
        df_['event'] = df_['event'].astype('int')

    df_.drop(['품종명'], axis='columns', inplace=True)

    if t:
        df_[f'target_price_{t}'] = df_['평균가격(원)'].shift(-t)

    return df_.dropna().reset_index(drop=True)


def fe_autogluon(df, train=True, item= None, t= None):
    df_ = df.copy()

    for i in range(1, 9):
        df_[f'price_pct_{i}'] = df_['평균가격(원)'].pct_change(periods=i)
    step = 9
    for i in range(step):
        df_[f'price_lag{i}'] = df_['평균가격(원)'].shift(i)
        if item in ['대파(일반)','양파','배추','감자 수미']:
            df_[f'price_ratio{i}'] = df_[f'price_lag{i}'] / df_['평년 평균가격(원) Common Year SOON'].shift(i)
    k = 3 # window size
    df_['price_mean0']=df_['평균가격(원)'].rolling(k, min_periods=1, center=True).mean()
    mean_step = 6
    for i in range(mean_step):
        df_[f'price_mean{i}']=df_[df_['price_mean0'].notnull()]['price_mean0'].shift(i)
    if not train:
        df_ = df_.dropna().reset_index(drop=True)
    df_['Month'] = df_['시점'].str[4:6].astype(float) / 12
    df_['Month_sin'] = np.sin(2 * np.pi * df_['Month'])
    df_['Month_cos'] = np.cos(2 * np.pi * df_['Month'])
    if t:
        df_[f'target_price_{t}'] = df_['평균가격(원)'].shift(-t)
    df_.drop(['Month','평년 평균가격(원) Common Year SOON','평균가격(원)'], axis=1, inplace=True)
    
    return df_.dropna().reset_index(drop=True)