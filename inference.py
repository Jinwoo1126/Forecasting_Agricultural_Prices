import os
import random
import joblib
import warnings
import json
import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

from src.prep import (
    Fluctuation_Probability, 
    fe_prob,
    fe_event,
    fe_autogluon
)

from src.utils import (
    process_data,
    set_seed,
    extract_year_month,
    price_agg,
    price_log,
    add_fourier_features,
    sliding_window,
    NMAE,
)

from src.model import (
    worker_init_fn,
    Data,
    Nlinear,
    CatBoost_Forecast,
    XGB_Forecast,
    RandomForest_Forecast,
    LGBM_Forecast,
)


filter_conditions = {
    '건고추': {'품종명': ['화건'], '거래단위': '30kg', '등급': '상품'},
    '사과': {'품종명': ['홍로', '후지'], '거래단위': '10개', '등급': '상품'},
    '감자 수미': {'품종명': [None], '거래단위': '20키로상자', '등급': '상'},
    '배': {'품종명': ['신고'], '거래단위': '10개', '등급': '상품'},
    '깐마늘(국산)': {'품종명': ['깐마늘(국산)'], '거래단위': '20kg', '등급': '상품'},
    '무': {'품종명': [None], '거래단위': '20키로상자', '등급': '상'},
    '상추': {'품종명': ['청'], '거래단위': '100g', '등급': '상품'},
    '배추': {'품종명': [None], '거래단위': '10키로망대', '등급': '상'},
    '양파': {'품종명': [None], '거래단위': '1키로', '등급': '상'},
    '대파(일반)': {'품종명': [None], '거래단위': '1키로단', '등급': '상'}
}

target_col = '평균가격(원)'
test_loss_list = []


if __name__=='__main__':
    # config = json.load(open('config.json'))

    root_path = './'
    test_df_path = os.path.join(root_path,'test')
    
    pretrained_path = os.path.join(root_path, 'pretrained_model')
    
    for file in sorted(os.listdir(test_df_path)):
        if '_1.csv' in file:
            prefix_file = file.split("_1.csv")[0]
            df_path2 = os.path.join(test_df_path, f'{prefix_file}_2.csv')
            df_path1 = os.path.join(test_df_path, f'{prefix_file}_1.csv')

            test1_df = pd.read_csv(df_path1)
            test2_df = pd.read_csv(df_path2)
            test_df = process_data(test1_df, test2_df)


            for item in filter_conditions.keys():
                pretrained_item_path = os.path.join(pretrained_path,f'{item} weight')
                item_condition = filter_conditions[item]

                target_test_df = test_df[
                    (test_df['품목명'] == item) &
                    (test_df['품종명'].isin(item_condition['품종명'])) &
                    (test_df['거래단위'] == item_condition['거래단위']) &
                    (test_df['등급'] == item_condition['등급'])
                    ].copy()

                if item == '배추':
                    ### autogluon
                    cat_col = ["시점", '품목명', '품종명', '거래단위', '등급','평년 평균가격(원) Common Year SOON']
                    target_test_df.fillna("None",inplace=True)
                    cur_preds = []
                    for step in [1,2,3]:
                        model_save_path = f'{item}_autogluon_step{step}.pkl'
                        model = joblib.load(os.path.join(pretrained_path,model_save_path))
                        tree_traget_col = [f'target_price_{step}']
                        fe_target_test_df = fe_autogluon(target_test_df, train=False,item=item)
                        pred_feature_idx = [col for col in fe_target_test_df.columns if col not in cat_col + tree_traget_col]
                        test_col = [col for col in fe_target_test_df.columns if col not in cat_col]
                        cur_preds.append(model.predict(fe_target_test_df[test_col].astype(float)).values[0])