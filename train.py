import os
import random
import joblib
import warnings
import json
import torch

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

from src.prep import (
    Fluctuation_Probability, 
    fe_prob,
    fe_event
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
    CatBoost_Forecast,
    XGB_Forecast,
    RandomForest_Forecast,
    LGBM_Forecast,
)



set_seed(42)
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    config = json.load(open('config.json'))

    root_path = config['data_dir']

    train_1 = pd.read_csv(os.path.join(root_path, 'train/train_1.csv'))
    train_2 = pd.read_csv(os.path.join(root_path, 'train/train_2.csv'))
    submission_df = pd.read_csv(os.path.join(root_path, 'sample_submission.csv'))
    train_df = process_data(train_1, train_2)

    train_dome = pd.read_csv(os.path.join(root_path, 'train/meta/TRAIN_경락정보_전국도매_2018-2022.csv'))
    ## sorted by YYYYMMSOON
    train_dome = train_dome.sort_values('YYYYMMSOON').reset_index(drop=True)

    prob_dict = Fluctuation_Probability(train_df, config).get_fluctuation_probability()


    ## train
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

    cat_submission_df = submission_df.copy()
    xgb_submission_df = submission_df.copy()
    lgb_submission_df = submission_df.copy()
    nlinear_submission_df = submission_df.copy()
    target_col = '평균가격(원)'
    test_loss_list = []


    for item in filter_conditions.keys():
        if item != '감자 수미':
            continue
        item_condition = filter_conditions[item]
        target_train_df = train_df[
            (train_df['품목명'] == item) &
            (train_df['품종명'].isin(item_condition['품종명'])) &
            (train_df['거래단위'] == item_condition['거래단위']) &
            (train_df['등급'] == item_condition['등급'])
            ].copy()
        
        if item == '감자 수미':
            for step in [1,2,3]:
                fe_target_train_df = fe_prob(target_train_df, item, prob_dict, t=step)
                tree_traget_col = [f'target_price_{step}']
                tree_feature = [col for col in fe_target_train_df.columns if col not in ["시점", '품목명', '품종명', '거래단위', '등급']+tree_traget_col]
                x = fe_target_train_df[tree_feature]
                y = fe_target_train_df[tree_traget_col]

                n_splits = 10
                kfold = KFold(n_splits=n_splits, random_state=1991, shuffle=True)
                for train_index, test_index in kfold.split(x):
                    x_train = x.iloc[train_index]
                    y_train = y.iloc[train_index]
                    x_valid = x.iloc[test_index]
                    y_valid = y.iloc[test_index]
                    cls_rf_model = RandomForest_Forecast(item, config)
                    model = cls_rf_model.train(x_train, y_train)
                    model_save_path = f'{item}_rf_step{step}.pkl'
                    joblib.dump(model, model_save_path)

        if item == '양파':
            # FE
            target_train_df = extract_year_month(target_train_df)
            agg_df, target_train_df = price_agg(target_train_df)
            target_train_df = add_fourier_features(target_train_df)
            target_train_df = price_log(target_train_df, target_col)
            extra_hist_col = ['mean', 'std', 'month_sin', 'month_cos', 'log_평균가격(원)']
            extra_x_, _= sliding_window(target_train_df[extra_hist_col], 9, 3)

            for step in [1,2,3]:
                fe_target_train_df = fe_event(target_train_df, item, t=step)
                tree_traget_col = [f'target_price_{step}']
                tree_feature = [col for col in fe_target_train_df.columns if col not in ["시점", '품목명', '품종명', '거래단위', '등급']+tree_traget_col]

                if 'event' in fe_target_train_df.columns:
                    tree_feature = ['평균가격(원)', '년도', '월', '순', 'season', 'event', 'mean', 'std', 'price_pct_1', 'fourier_sin', 'fourier_cos']
                    cat_col = ['년도', '월', '순', 'season', 'event']
                    fe_target_train_df['event'] = fe_target_train_df['event'].astype('int')
                else:
                    tree_feature = ['평균가격(원)', '년도', '월', '순', 'season', 'event', 'mean', 'std', 'price_pct_1', 'fourier_sin', 'fourier_cos']
                    cat_col = ['년도', '월', '순', 'season']

                x = fe_target_train_df[tree_feature]
                y = fe_target_train_df[tree_traget_col]

                n_splits = 10
                kfold = KFold(n_splits=n_splits, random_state=1991, shuffle=True)
                for idx, (train_index, test_index) in enumerate(kfold.split(x)):
                    x_train = x.iloc[train_index]
                    y_train = y.iloc[train_index]
                    x_valid = x.iloc[test_index]
                    y_valid = y.iloc[test_index]

                    cls_cat_model = CatBoost_Forecast(item, config)
                    cat_model = cls_cat_model.train(x_train, y_train, x_valid, y_valid, cat_col)
                    model_save_path = f'cat_{item}_fold{idx+1}_step{step}.pkl'
                    joblib.dump(cat_model, model_save_path)

                    cls_xgb_model = XGB_Forecast(item, config)
                    xgb_model = cls_xgb_model.train(x_train, y_train, x_valid, y_valid)
                    model_save_path = f'xgb_{item}_fold{idx+1}_step{step}.pkl'
                    joblib.dump(xgb_model, model_save_path)

                    cls_lgb_model = LGBM_Forecast(item, config)
                    lgb_model = cls_lgb_model.train(x_train, y_train, x_valid, y_valid)
                    model_save_path = f'lgb_{item}_fold{idx+1}_step{step}.pkl'
                    joblib.dump(lgb_model, model_save_path)