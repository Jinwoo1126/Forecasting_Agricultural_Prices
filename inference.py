import os
import random
import joblib
import warnings
import json
import torch

import pandas as pd
import numpy as np

from easydict import EasyDict
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
import xgboost as xgb

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
    custom_linear,
    worker_init_fn,
    Data,
    Nlinear,
    CatBoost_Forecast,
    XGB_Forecast,
    RandomForest_Forecast,
    LGBM_Forecast,
)

# Model cache for reusing loaded models
model_cache = {}

def load_model(model_path):
    if model_path in model_cache:
        return model_cache[model_path]
    model = joblib.load(model_path)
    model_cache[model_path] = model
    return model

filter_conditions = {
    '건고추': {'품종명': ['화건'], '거래단위': '30kg', '등급': '상품'},
    '사과': {'품종명': ['홍로', '후지'], '거래단위': '10개', '등급': '상품'},
    '감자 수미': {'품종명': [None], '거래단위': '20키로상자', '등급': '상'},
    '배': {'품종명': ['신고'], '거래단위': '10개', '등급': '상품'},
    '깐마늘(국산)': {'품종명': ['깐마늘(국산)'], '거래단위': '20kg', '등급': '상품'},
    '무': {'품종명': [None], '거래단위': '20키로상자', '등급': '상'},
    '상추': {'품종명': ['청'], '거래단위': '100g', '등급': '상품'},
    '양파': {'품종명': [None], '거래단위': '1키로', '등급': '상'},
    '대파(일반)': {'품종명': [None], '거래단위': '1키로단', '등급': '상'},
    '배추': {'품종명': [None], '거래단위': '10키로망대', '등급': '상'},
}

target_col = '평균가격(원)'
test_loss_list = []

if __name__ == '__main__':
    config = json.load(open('config.json'))

    root_path = config['data_dir']
    test_df_path = os.path.join(root_path, 'test')
    train_df_path = os.path.join(root_path, 'train')

    pretrained_path = os.path.join(root_path, 'pretrained_model')

    submission_df = pd.read_csv(os.path.join(root_path, 'sample_submission.csv'))

    # Load training data once
    train_1 = pd.read_csv(os.path.join(train_df_path, 'train_1.csv'))
    train_2 = pd.read_csv(os.path.join(train_df_path, 'train_2.csv'))
    train_df = process_data(train_1, train_2)

    # Pre-compute fluctuation probability
    prob_dict = Fluctuation_Probability(train_df, config).get_fluctuation_probability()

    # Load all test files at once
    test_dfs = {}
    for file in sorted(os.listdir(test_df_path)):
        if '_1.csv' in file:
            prefix_file = file.split("_1.csv")[0]
            df_path1 = os.path.join(test_df_path, f'{prefix_file}_1.csv')
            df_path2 = os.path.join(test_df_path, f'{prefix_file}_2.csv')
            test1_df = pd.read_csv(df_path1)
            test2_df = pd.read_csv(df_path2)
            test_dfs[prefix_file] = process_data(test1_df, test2_df)

    # Function to run prediction for each item
    def run_prediction_for_item(item, file, test_df):
        pretrained_item_path = os.path.join(pretrained_path, f'{item} weight')
        item_condition = filter_conditions[item]

        # Filter training data
        target_train_df = train_df[
            (train_df['품목명'] == item) &
            (train_df['품종명'].isin(item_condition['품종명'])) &
            (train_df['거래단위'] == item_condition['거래단위']) &
            (train_df['등급'] == item_condition['등급'])
        ].copy()

        # Feature extraction for train data
        target_train_df = extract_year_month(target_train_df)
        if item in ['상추', '배', '양파', '대파(일반)']:
            agg_df, target_train_df = price_agg(target_train_df)

        # Filter test data
        target_test_df = test_df[
            (test_df['품목명'] == item) &
            (test_df['품종명'].isin(item_condition['품종명'])) &
            (test_df['거래단위'] == item_condition['거래단위']) &
            (test_df['등급'] == item_condition['등급'])
        ].copy()

        # Prediction logic for different items
        timing = '_'.join(file.split("_")[:2])
        if item == '배추':
            # Example for 배추 using AutoGluon
            cat_col = ["시점", '품목명', '품종명', '거래단위', '등급', '평년 평균가격(원) Common Year SOON']
            target_test_df.fillna("None", inplace=True)
            cur_preds = []
            for step in [1, 2, 3]:
                model_save_path = f'{item}_autogluon_step{step}.pkl'
                model = load_model(os.path.join(pretrained_item_path, model_save_path))
                fe_target_test_df = fe_autogluon(target_test_df, train=False, item=item)
                test_col = [col for col in fe_target_test_df.columns if col not in cat_col]
                cur_preds.append(model.predict(fe_target_test_df[test_col].astype(float)).values[0])

            submission_df.loc[submission_df['시점'].str.startswith(timing), item] = cur_preds

        # Add similar logic for other items here...

    # Parallel processing for each test file and item
    for file in sorted(os.listdir(test_df_path)):
        if '_1.csv' in file:
            prefix_file = file.split("_1.csv")[0]
            test_df = test_dfs[prefix_file]

            Parallel(n_jobs=-1)(
                delayed(run_prediction_for_item)(item, file, test_df) for item in filter_conditions.keys()
            )

    # Save final submission
    submission_df.to_csv('./complete_submission.csv', index=False)
