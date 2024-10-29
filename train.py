import os
import random
import joblib
import warnings
import json
import torch

import pandas as pd
import numpy as np

from src.utils import (
    process_data,
    set_seed
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