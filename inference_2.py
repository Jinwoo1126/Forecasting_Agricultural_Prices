import os
import random
import joblib
import warnings
import json
import torch

import pandas as pd
import numpy as np

from easydict import EasyDict
from torch.utils.data import DataLoader
import time

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
    sliding_window,
)

from src.model import (
    Data,
    Nlinear,
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
    tic= time.time()
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

        # Prediction logic for each item type
        timing = '_'.join(file.split("_")[:2])

        if item == '배추':
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

        elif item == '무':
            n_splits = 10
            for step in [1, 2, 3]:
                fe_target_train_df = fe_event(target_train_df, item, t=step)
                agg_df, fe_target_train_df = price_agg(fe_target_train_df)

            fe_target_test_df = fe_event(target_test_df, item)
            fe_target_test_df = pd.merge(fe_target_test_df, agg_df, on=['월', '순'], how='left')

            if 'event' in fe_target_test_df.columns:
                fe_target_test_df['event'] = fe_target_test_df['event'].astype('int')

            cat_pred = []
            for step in [1, 2, 3]:
                cat_pred_ = 0
                for i in range(1, n_splits + 1):
                    model = load_model(os.path.join(pretrained_item_path, f'{item}_cat_fold{i}_step{step}.pkl'))
                    tree_feature = [col for col in fe_target_test_df.columns if col not in ["시점", '품목명', '품종명', '거래단위', '등급'] + [f'target_price_{step}']]
                    cat_pred_ += np.expm1(model.predict(fe_target_test_df.loc[0:, tree_feature])[0]) / n_splits
                cat_pred.append(cat_pred_)

            submission_df.loc[submission_df['시점'].str.startswith(timing), item] = cat_pred

        elif item == '감자 수미':
            mapping_table = {
                '감자 수미': {
                    '품목명': '감자',
                    '품종명': '수미',
                    '시장명': '*전국도매시장',
                }
            }
            test_dome = pd.read_csv(os.path.join(test_df_path, "meta/" + ''.join(file.split("_")[0] + '_경락정보_전국도매_' + file.split("_")[1] + '.csv')))
            test_dome = test_dome.sort_values('YYYYMMSOON').reset_index(drop=True)

            target_test_df['총반입량(kg)'] = test_dome[(test_dome['품목명'] == mapping_table[item]['품목명']) &
                                                     (test_dome['품종명'] == mapping_table[item]['품종명']) &
                                                     (test_dome['시장명'] == mapping_table[item]['시장명'])]['총반입량(kg)'].values
            for i in range(1, 9):
                target_test_df[f'income_lag{i}'] = target_test_df['총반입량(kg)'].shift(i)

            fe_target_test_df = fe_prob(target_test_df, item, prob_dict)

            rf_pred = []
            for step in [1, 2, 3]:
                tree_feature = [col for col in fe_target_test_df.columns if col not in ["시점", '품목명', '품종명', '거래단위', '등급'] + [f'target_price_{step}']]
                rf_pred_ = 0
                for i in range(1, n_splits + 1):
                    model = load_model(os.path.join(pretrained_item_path, f'{item}_rf_step{step}.pkl'))
                    rf_pred_ += np.expm1(model.predict(fe_target_test_df.loc[0:, tree_feature])[0]) / n_splits
                rf_pred.append(rf_pred_)

            test_x, test_y = sliding_window(target_test_df[target_col], 9, 0)
            test_ds = Data(test_x, test_y)
            test_dl = DataLoader(test_ds, batch_size=test_y.shape[0], shuffle=False)
            nlinear_prediction = 0
            for fold in range(n_splits):
                n_linear = Nlinear(EasyDict({
                    'ltsf_window_size': 9,
                    'output_step': 3,
                    'individual': True,
                    'num_item': 1,
                    'num_experts': 4,
                    'attention_heads': 4,
                    'batch_size': 4,
                    'epoch': 100,
                    'lr': 0.01,
                    'kernel_size': 9
                }))
                n_linear.load_state_dict(torch.load(os.path.join(pretrained_item_path, f"{item}_fold{fold + 1}_nlinear_model.pth"), weights_only=True))
                n_linear.eval()
                with torch.no_grad():
                    for data, target in test_dl:
                        nlinear_prediction += n_linear(data).numpy().reshape(-1) / n_splits

            ensemble = rf_pred * 0.9 + nlinear_prediction * 0.1
            submission_df.loc[submission_df['시점'].str.startswith(timing), item] = ensemble

        elif item in ['건고추', '깐마늘(국산)', '사과']:
            args = {
                "ltsf_window_size": 9,
                "output_step": 3,
                "individual": True,
                "num_item": 1,
                "batch_size": 4,
                "epoch": 100,
                "lr": 0.001
            }
            n_splits = 3

            test_x, test_y = sliding_window(target_test_df[target_col], 9, 0)
            test_ds = Data(test_x, test_y)
            test_dl = DataLoader(test_ds, batch_size=test_y.shape[0], shuffle=False)

            nlinear_prediction = 0
            for fold in range(n_splits):
                n_linear = Nlinear(args)
                n_linear.load_state_dict(torch.load(os.path.join(pretrained_item_path, f"{item}_fold{fold + 1}_nlinear_model.pth"), weights_only=True))
                n_linear.eval()
                with torch.no_grad():
                    for data, target in test_dl:
                        nlinear_prediction += n_linear(data).numpy().reshape(-1) / n_splits

            submission_df.loc[submission_df['시점'].str.startswith(timing), item] = nlinear_prediction

    # Parallel processing for each test file and item
    for file in sorted(os.listdir(test_df_path)):
        if '_1.csv' in file:
            prefix_file = file.split("_1.csv")[0]
            test_df = test_dfs[prefix_file]

            joblib.Parallel(n_jobs=-1)(
                joblib.delayed(run_prediction_for_item)(item, file, test_df) for item in filter_conditions.keys()
            )

    # Save final submission
    submission_df.to_csv('./complete_submission2.csv', index=False)
    toc = time.time()
    print(f"Time {toc-tic:.4f}s")