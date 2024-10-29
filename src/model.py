import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor



class NaiveForecaster:
    def __init__(self, steps):
        self.last_value = None
        self.steps = steps

    def fit(self, data):
        self.last_value = data.iloc[-1]

    def predict(self, steps=1):
        return [self.last_value] * self.steps
    

class Data(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Nlinear(torch.nn.Module):
    def __init__(self, args):
        super(Nlinear, self).__init__()
        self.window_size = args.ltsf_window_size
        self.forcast_size = args.output_step
        self.individual = args.individual
        self.channels = args.num_item
        self.dropout = torch.nn.Dropout(p=0.25)

        if self.individual == True:
            self.Linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)

    
    def worker_init_fn(self, worker_id):
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)

    def forward(self, x):
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = self.dropout(x)
        x = x + seq_last
        return x
    

class LGBM_Forecast:
    def __init__(self, item, config):
        self.params = config['model_params'][item]['LGBM']

    def train(self, x_train, y_train, x_valid, y_valid):
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_set=train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round = 10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                ],
            init_model = None,
        )

        return self.model
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def get_model(self):
        return self.model
    

class RandomForest_Forecast:
    def __init__(self, item, config):
        self.params = config['model_params'][item]['RF']

    def train(self, x_train, y_train):
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(x_train, y_train)

        return self.model
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def get_model(self):
        return self.model
    

class XGB_Forecast:
    def __init__(self, item, config):
        self.params = config['model_params'][item]['XGB']

    def train(self, x_train, y_train, x_valid, y_valid):
        train_data = xgb.DMatrix(data=pd.get_dummies(x_train), label=y_train)
        valid_data = xgb.DMatrix(data=pd.get_dummies(x_valid), label=y_valid)

        self.model = xgb.train(
            self.params,
            train_data,
            num_boost_round=10000,
            evals=[(train_data, 'train'), (valid_data, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        return self.model
    
    def predict(self, x_test):
        return self.model.predict(xgb.DMatrix(pd.get_dummies(x_test)))
    
    def get_model(self):
        return self.model


class CatBoost_Forecast:
    def __init__(self, item, config):
        self.params = config['model_params'][item]['CatBoost']

    def train(self, x_train, y_train, x_valid, y_valid, cat_col):
        self.model = CatBoostRegressor(
            learning_rate=0.05,
            iterations=10000,
            loss_function='RMSE',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50
        )

        self.model.fit(
            x_train, y_train,
            eval_set=(x_valid, y_valid),
            cat_features=cat_col,
            use_best_model=True
        )

        return self.model
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def get_model(self):
        return self.model