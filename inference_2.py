import os
import time
import joblib
import warnings
import json
import torch

import pandas as pd
import numpy as np

from easydict import EasyDict
from torch.utils.data import DataLoader
import xgboost as xgb

from src.prep import (
    Fluctuation_Probability, 
    fe_prob,
    fe_event,
    fe_autogluon
)

from src.utils import (
    process_data,
    extract_year_month,
    price_agg,
    price_log,
    add_fourier_features,
    sliding_window,
)

from src.model import (
    custom_linear,
    Data,
    Nlinear,
)

warnings.filterwarnings("ignore", category=UserWarning)
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

item_ensemble_weight_dict = {
    '배': [0.0, 0.0, 0.65, 0.35],
    '양파': [0.3, 0.0, 0.15, 0.55],
    '대파(일반)': [0.1, 0.1, 0.4, 0.4],
    '상추': [0.0, 0.45, 0.21, 0.34],
}

target_col = '평균가격(원)'
test_loss_list = []
cache = {}

if __name__=='__main__':
    config = json.load(open('config.json'))

    # root_path = config['data_dir']
    test_df_path = os.path.join(root_path,'test')
    train_df_path = os.path.join(root_path,'train')
    pretrained_path = os.path.join(root_path , 'pretrained_model')

    submission_df = pd.read_csv(os.path.join(root_path,'sample_submission.csv'))

    train_1 = pd.read_csv(os.path.join(train_df_path,'train_1.csv'))
    train_2 = pd.read_csv(os.path.join(train_df_path,'train_2.csv'))
    train_df = process_data(train_1, train_2)
    prob_dict = Fluctuation_Probability(train_df, config).get_fluctuation_probability()

    # 모델 미리 로딩하여 저장
    model_cache = {}
    for item in filter_conditions.keys():
        model_cache[item] = {}
        for step in [1, 2, 3]:
            model_cache[item][step] = {
                'cat': [],
                'xgb': [],
                'lgb': [],
                'nlinear': []
            }
            pretrained_item_path = os.path.join(pretrained_path, f'{item} weight')
            for i in range(1, 11):
                if os.path.exists(os.path.join(pretrained_item_path, f'{item}_cat_fold{i}_step{step}.pkl')):
                    model_cache[item][step]['cat'].append(joblib.load(os.path.join(pretrained_item_path, f'{item}_cat_fold{i}_step{step}.pkl')))
                if os.path.exists(os.path.join(pretrained_item_path, f'xgb_{item}_fold{i}_step{step}.pkl')):
                    model_cache[item][step]['xgb'].append(joblib.load(os.path.join(pretrained_item_path, f'xgb_{item}_fold{i}_step{step}.pkl')))
                if os.path.exists(os.path.join(pretrained_item_path, f'lgb_{item}_fold{i}_step{step}.pkl')):
                    model_cache[item][step]['lgb'].append(joblib.load(os.path.join(pretrained_item_path, f'lgb_{item}_fold{i}_step{step}.pkl')))
                if os.path.exists(os.path.join(pretrained_item_path, f'{item}_fold{i}_nlinear_model.pth')):
                    model_cache[item][step]['nlinear'].append(torch.load(os.path.join(pretrained_item_path, f'{item}_fold{i}_nlinear_model.pth')))
    
    # Test 데이터 처리 및 모델 예측
    for file in sorted(os.listdir(test_df_path)):
        if '_1.csv' in file:
            prefix_file = file.split("_1.csv")[0]
            df_path2 = os.path.join(test_df_path, f'{prefix_file}_2.csv')
            df_path1 = os.path.join(test_df_path, f'{prefix_file}_1.csv')

            test1_df = pd.read_csv(df_path1)
            test2_df = pd.read_csv(df_path2)
            test_df = process_data(test1_df, test2_df)

            for item in filter_conditions.keys():
                item_condition = filter_conditions[item]
                
                # train_df 처리 (최적화된 피처 엔지니어링)
                if f'{item}_train_df' not in cache:
                    target_train_df = train_df[
                        (train_df['품목명'] == item) &
                        (train_df['품종명'].isin(item_condition['품종명'])) &
                        (train_df['거래단위'] == item_condition['거래단위']) &
                        (train_df['등급'] == item_condition['등급'])
                        ].copy()

                    target_train_df = extract_year_month(target_train_df)
                    if item in ['상추','배','양파','대파(일반)']:
                        agg_df, target_train_df = price_agg(target_train_df)
                    
                    cache[f'{item}_train_df'] = target_train_df
                else:
                    target_train_df = cache[f'{item}_train_df']
                
                # test_df 처리
                target_test_df = test_df[
                    (test_df['품목명'] == item) &
                    (test_df['품종명'].isin(item_condition['품종명'])) &
                    (test_df['거래단위'] == item_condition['거래단위']) &
                    (test_df['등급'] == item_condition['등급'])
                    ].copy()

                timing = '_'.join(file.split("_")[:2])

                if item == '배추':
                    cur_preds = []
                    for step in [1,2,3]:
                        fe_target_test_df = fe_autogluon(target_test_df, train=False, item=item)
                        cat_col = ["시점", '품목명', '품종명', '거래단위', '등급','평년 평균가격(원)']
                        test_col = [col for col in fe_target_test_df.columns if col not in cat_col]
                        model = joblib.load(os.path.join(pretrained_item_path, f'{item}_autogluon_step{step}.pkl'))
                        cur_preds.append(model.predict(fe_target_test_df[test_col].astype(float)).values[0])
                    submission_df.loc[submission_df['시점'].str.startswith(timing), item] = cur_preds

                if item in ['상추','배','양파','대파(일반)']:
                    item_ensemble_weight_list = item_ensemble_weight_dict[item]

                    fe_target_test_df = fe_event(target_test_df, item)
                    fe_target_test_df = pd.merge(fe_target_test_df, agg_df, on=['월','순'], how='left')
                    fe_target_test_df['event'] = fe_target_test_df['event'].astype('int')
                    tree_feature = ['평균가격(원)', '년도', '월', '순', 'season', 'event', 'mean', 'std', 'price_pct_1', 'fourier_sin', 'fourier_cos']

                    # XGB 예측
                    if item_ensemble_weight_list[2] > 0:
                        xgb_pred = []
                        for step in [1,2,3]:
                            xgb_pred_ = 0
                            for model in model_cache[item][step]['xgb']:
                                xgb_pred_ += np.expm1(model.predict(xgb.DMatrix(data=pd.get_dummies(fe_target_test_df.loc[0:, tree_feature])))[0])/len(model_cache[item][step]['xgb'])
                            xgb_pred.append(xgb_pred_)
                        xgb_pred = np.array(xgb_pred)
                    else:
                        xgb_pred = np.array([0, 0, 0])

                    # LGB 예측
                    if item_ensemble_weight_list[1] > 0:
                        lgb_pred = []
                        for step in [1,2,3]:
                            lgb_pred_ = 0
                            for model in model_cache[item][step]['lgb']:
                                lgb_pred_ += np.expm1(model.predict(fe_target_test_df.loc[0:, tree_feature])[0])/len(model_cache[item][step]['lgb'])
                            lgb_pred.append(lgb_pred_)
                        lgb_pred = np.array(lgb_pred)
                    else:
                        lgb_pred = np.array([0, 0, 0])
                    
                    # CAT 예측
                    if item_ensemble_weight_list[0] > 0:
                        cat_pred = []
                        for step in [1,2,3]:
                            cat_pred_ = 0
                            for model in model_cache[item][step]['cat']:
                                cat_pred_ += np.expm1(model.predict(fe_target_test_df.loc[0:, tree_feature])[0])/len(model_cache[item][step]['cat'])
                            cat_pred.append(cat_pred_)
                        cat_pred = np.array(cat_pred)
                    else:
                        cat_pred = np.array([0, 0, 0])

                    # Nlinear 예측
                    if item_ensemble_weight_list[3] > 0:
                        custom_linear_prediction = 0
                        test_x, test_y = sliding_window(target_test_df[target_col], 9, 0)
                        test_ds = Data(test_x, test_y)
                        test_dl = DataLoader(test_ds, batch_size=test_y.shape[0], shuffle=False)
                        for model in model_cache[item][step]['nlinear']:
                            model.eval()
                            with torch.no_grad():
                                for data, target in test_dl:
                                    custom_linear_prediction += model(data).numpy().reshape(-1)/len(model_cache[item][step]['nlinear'])
                    else:
                        custom_linear_prediction = np.array([0, 0, 0])

                    # 앙상블
                    ensemble = custom_linear_prediction * item_ensemble_weight_list[3] + \
                               xgb_pred * item_ensemble_weight_list[2] + \
                               lgb_pred * item_ensemble_weight_list[1] + \
                               cat_pred * item_ensemble_weight_list[0]
                    submission_df.loc[submission_df['시점'].str.startswith(timing), item] = ensemble

            print(file, "Done")

    submission_df.to_csv('./complete_submission.csv',index=False)
