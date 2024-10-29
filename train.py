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
    sliding_window,
    add_fourier_features,
    NMAE,
)

from src.model import (
    worker_init_fn,
    Data,
    Nlinear,
    custom_linear,
    CatBoost_Forecast,
    XGB_Forecast,
    RandomForest_Forecast,
    LGBM_Forecast,
)

from autogluon.tabular import TabularDataset, TabularPredictor

set_seed(42)
warnings.filterwarnings("ignore", category=UserWarning)



if __name__ == '__main__':
    config = json.load(open('config.json'))

    root_path = config['data_dir']
    # root_path = './data'

    train_1 = pd.read_csv(os.path.join(root_path, 'train/train_1.csv'))
    train_2 = pd.read_csv(os.path.join(root_path, 'train/train_2.csv'))
    submission_df = pd.read_csv(os.path.join(root_path, 'sample_submission.csv'))
    train_df = process_data(train_1, train_2)

    train_dome = pd.read_csv(os.path.join(root_path, 'train/meta/TRAIN_경락정보_전국도매_2018-2022.csv'))
    ## sorted by YYYYMMSOON
    train_dome = train_dome.sort_values('YYYYMMSOON').reset_index(drop=True)

    # prob_dict = Fluctuation_Probability(train_df, config).get_fluctuation_probability()


    ## train
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

    cat_submission_df = submission_df.copy()
    xgb_submission_df = submission_df.copy()
    lgb_submission_df = submission_df.copy()
    nlinear_submission_df = submission_df.copy()
    target_col = '평균가격(원)'
    test_loss_list = []


    for item in filter_conditions.keys():
        item_condition = filter_conditions[item]
        target_train_df = train_df[
            (train_df['품목명'] == item) &
            (train_df['품종명'].isin(item_condition['품종명'])) &
            (train_df['거래단위'] == item_condition['거래단위']) &
            (train_df['등급'] == item_condition['등급'])
            ].copy()

        if item == '배추':
            os.makedirs(os.path.join(root_path,'autogluon_result'),exist_ok=True)
            cat_col = ["시점", '품목명', '품종명', '거래단위', '등급','평년 평균가격(원)']
            target_train_df.fillna("None",inplace=True)
            for step in [1,2,3]:
                model_save_path = f'{item}_autogluon_step{step}.pkl'
                fe_target_train_df = fe_autogluon(target_train_df, t=step)
                tree_traget_col = [f'target_price_{step}']
                now_train_df = fe_autogluon(target_train_df, train=True, item=item, t=step)
                train_col = [col for col in now_train_df.columns if col not in cat_col+tree_traget_col]
                now_train_df = now_train_df[train_col + [f'target_price_{step}']]

                train_data = TabularDataset(now_train_df.astype(float))
                model = TabularPredictor(label=f'target_price_{step}', eval_metric='mean_absolute_percentage_error',verbosity=1, problem_type='regression')
                model.fit(train_data,
                        presets='medium_quality',
                        time_limit=240,
                        auto_stack=True,
                            )
                joblib.dump(model, os.path.join(root_path,'autogluon_result',model_save_path))
                results = model.fit_summary()
                
        if item in ['건고추', '사과', '깐마늘(국산)']:
            ## NLinear Model
            n_splits = 3
            kfold = KFold(n_splits=n_splits, random_state=42, shuffle=True)
            x_, y_ = sliding_window(target_train_df[target_col], 9, 3)
            loss_list_ = []
            for idx, (train_index, valid_index) in enumerate(kfold.split(x_)):
                train_x = x_[train_index]
                train_y = y_[train_index]
                valid_x = x_[valid_index]
                valid_y = y_[valid_index]

                train_ds = Data(train_x, train_y)
                train_dl = DataLoader(
                    train_ds,
                    batch_size = config['model_params'][item]['Nlinear']['batch_size'],
                    shuffle=True,
                    worker_init_fn=worker_init_fn,
                    generator=torch.Generator().manual_seed(42)
                    )
                valid_ds = Data(valid_x, valid_y)
                valid_dl = DataLoader(
                    valid_ds,
                    batch_size = valid_x.shape[0],
                    shuffle=False,
                    worker_init_fn=lambda worker_id: np.random.seed(42),
                    generator=torch.Generator().manual_seed(42)
                    )
                torch.manual_seed(42)
                n_linear = Nlinear(config['model_params'][item]['Nlinear'])
                optimizer = torch.optim.Adam(n_linear.parameters(), lr=config['model_params'][item]['Nlinear']['lr'])
                max_loss = 999999999

                for epoch in tqdm(range(1, config['model_params'][item]['Nlinear']['epoch']+1)):
                    loss_list = []
                    n_linear.train()
                    for batch_idx, (data, target) in enumerate(train_dl):
                        optimizer.zero_grad()
                        output = n_linear(data)
                        loss = NMAE(output.squeeze(), target)
                        loss.backward()
                        optimizer.step()
                        loss_list.append(loss.item())
                    train_loss = np.mean(loss_list)

                    n_linear.eval()
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(valid_dl):
                            output = n_linear(data)
                            valid_loss = NMAE(output, target.unsqueeze(-1))

                        if valid_loss < max_loss:
                            max_loss = valid_loss
                            torch.save(n_linear.state_dict(), f"{item}_fold{idx+1}_nlinear_model.pth")

                    if epoch % 10 == 0:
                        print("epoch: {}, Train_NMAE: {:.3f}, Valid_NMAE: {:.3f}".format(epoch, train_loss, valid_loss))
                loss_list_.append(max_loss.item())
            test_loss_list.append(np.mean(loss_list_))
     
        
        if item == '감자 수미':
            ## NLinear Model
            n_splits = 3
            kfold = KFold(n_splits=n_splits, random_state=42, shuffle=True)
            x_, y_ = sliding_window(target_train_df[target_col], 9, 3)
            loss_list_ = []
            for idx, (train_index, valid_index) in enumerate(kfold.split(x_)):
                train_x = x_[train_index]
                train_y = y_[train_index]
                valid_x = x_[valid_index]
                valid_y = y_[valid_index]

                train_ds = Data(train_x, train_y)
                train_dl = DataLoader(
                    train_ds,
                    batch_size = config['model_params'][item]['Nlinear']['batch_size'],
                    shuffle=True,
                    worker_init_fn=worker_init_fn,
                    generator=torch.Generator().manual_seed(42)
                    )
                valid_ds = Data(valid_x, valid_y)
                valid_dl = DataLoader(
                    valid_ds,
                    batch_size = valid_x.shape[0],
                    shuffle=False,
                    worker_init_fn=lambda worker_id: np.random.seed(42),
                    generator=torch.Generator().manual_seed(42)
                    )
                torch.manual_seed(42)
                n_linear = Nlinear(config['model_params'][item]['Nlinear'])
                optimizer = torch.optim.Adam(n_linear.parameters(), lr=config['model_params'][item]['Nlinear']['lr'])
                max_loss = 999999999

                for epoch in tqdm(range(1, config['model_params'][item]['Nlinear']['epoch']+1)):
                    loss_list = []
                    n_linear.train()
                    for batch_idx, (data, target) in enumerate(train_dl):
                        optimizer.zero_grad()
                        output = n_linear(data)
                        loss = NMAE(output.squeeze(), target)
                        loss.backward()
                        optimizer.step()
                        loss_list.append(loss.item())
                    train_loss = np.mean(loss_list)

                    n_linear.eval()
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(valid_dl):
                            output = n_linear(data)
                            valid_loss = NMAE(output, target.unsqueeze(-1))

                        if valid_loss < max_loss:
                            max_loss = valid_loss
                            torch.save(n_linear.state_dict(), f"{item}_fold{idx+1}_nlinear_model.pth")

                    if epoch % 10 == 0:
                        print("epoch: {}, Train_NMAE: {:.3f}, Valid_NMAE: {:.3f}".format(epoch, train_loss, valid_loss))
                loss_list_.append(max_loss.item())
            test_loss_list.append(np.mean(loss_list_))

            ## RF Model
            target_train_df['총반입량(kg)'] = (train_dome[(train_dome['품목명'] == config['mapping_table'][item]['품목명']) & 
                                        (train_dome['품종명'] == config['mapping_table'][item]['품종명']) & 
                                        (train_dome['시장명'] == config['mapping_table'][item]['시장명'])]['총반입량(kg)'].values)
                
            for i in range(1, 9):
                target_train_df[f'income_lag{i}'] = target_train_df['총반입량(kg)'].shift(i)
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


        if item in ['양파', '배', '상추', '대파(일반)']:
            # F파
            target_train_df = extract_year_month(target_train_df)
            agg_df, target_train_df = price_agg(target_train_df)
            target_train_df = add_fourier_features(target_train_df)
            target_train_df = price_log(target_train_df, target_col)

            n_splits = 10
            kfold = KFold(n_splits=n_splits, random_state=42, shuffle=True)
            x_, y_ = sliding_window(target_train_df[target_col], 9, 3)

            loss_list_ = []
            for idx, (train_index, valid_index) in enumerate(kfold.split(x_)):
                train_x = x_[train_index]
                train_y = y_[train_index]
                valid_x = x_[valid_index]
                valid_y = y_[valid_index]

                train_ds = Data(train_x, train_y)
                train_dl = DataLoader(
                    train_ds,
                    batch_size = config['model_params'][item]['Customlinear']['batch_size'],
                    shuffle=True,
                    worker_init_fn=worker_init_fn,
                    generator=torch.Generator().manual_seed(42),
                    drop_last=True
                    )
                valid_ds = Data(valid_x, valid_y)
                valid_dl = DataLoader(
                    valid_ds,
                    batch_size = valid_x.shape[0],
                    shuffle=False,
                    worker_init_fn=lambda worker_id: np.random.seed(42),
                    generator=torch.Generator().manual_seed(42)
                    )
                torch.manual_seed(42)
                model = custom_linear(config['model_params'][item]['Customlinear'])
                optimizer = torch.optim.Adam(model.parameters(), lr=config['model_params'][item]['Customlinear']['lr'])
                max_loss = 999999999

                for epoch in tqdm(range(1, config['model_params'][item]['Customlinear']['epoch']+1)):
                    loss_list = []
                    model.train()

                    for batch_idx, (data, target) in enumerate(train_dl):
                        optimizer.zero_grad()
                        output = model(data)
                        loss = NMAE(output.squeeze(), target)
                        loss.backward()
                        optimizer.step()
                        nmae = NMAE(output.squeeze(), target)
                        loss_list.append(nmae.item())
                    train_nmae = np.mean(loss_list)

                    model.eval()
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(valid_dl):
                            output = model(data)
                            valid_nmae = NMAE(output, target.unsqueeze(-1))

                        if valid_nmae < max_loss:
                            max_loss = valid_nmae
                            torch.save(model.state_dict(), f"{item}_fold{idx+1}_custom_linear_model.pth")

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


        if item == '무':
            for step in [1,2,3]:
                fe_target_train_df = fe_event(target_train_df, item, t=step)
                agg_df, fe_target_train_df = price_agg(fe_target_train_df)
                tree_traget_col = [f'target_price_{step}']
                tree_feature = [col for col in fe_target_train_df.columns if col not in ["시점", '품목명', '품종명', '거래단위', '등급']+tree_traget_col]

                if 'event' in fe_target_train_df.columns:
                    tree_feature = ['평균가격(원)', '년도', '월', '순', 'season', 'event', 'mean', 'std', 'price_pct_1', 'fourier_sin', 'fourier_cos']
                    cat_col = ['년도', '월', '순', 'season', 'event']
                    fe_target_train_df['event'] = fe_target_train_df['event'].astype('int')
                else:
                    tree_feature = ['평균가격(원)', '년도', '월',  'season', 'mean', 'std', 'price_pct_1', 'fourier_sin', 'fourier_cos']
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
                    model_save_path = f'{item}_cat_fold{idx+1}_step{step}.pkl'
                    joblib.dump(cat_model, model_save_path)
