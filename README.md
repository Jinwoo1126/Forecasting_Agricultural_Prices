# [Dacon] BreadcrumbsForecasting_Agricultural_Prices

## 어떻게 데이터 저장할지?
- 현재 디렉토리에 아래와 같이 저장
```
    root_path ('./')

    .
    ./train
      ㄴ ...
    ./test
      ㄴ ...
    .sample_submission.csv
    .inference.py
    .train.py

```
## 개발환경
- Google colab (OS Linux)
```
    품목: 배, 상추, 양파, 배추, 대파, 무
```
- Mac OS
```
    chip : Apple M2 Ultra
    OS : Sonoma 14.5
    품목: 사과, 깐마늘, 건고추, 감자 수미
```

## Requisites for TRAINING
### Requirements for Training on Colab (python 3.10.12)
```
    !pip install -U uv
    !pip install dask[dataframe]
    !uv pip install autogluon --system
    !pip install lightgbm
```
### Requirements for Training on Mac for 사과, 깐마늘, 건고추, 감자 수미
```
    pip install -r requirements_mac.txt
```

### Requirements for Training on Linux for 사과, 깐마늘, 건고추, 감자 수미
```
    pip install -r requirements.txt
    pip install autogluon
```

## Requisites for INFERENCE
### Requirements for Inference
```
    pip install -r requirements.txt
    pip install autogluon
```

## Implementation

1. `config.json`의 `data_dir`을 맞게 수정 (대회 주최측 제공 데이터 구조에 맞게 저장한 부모 디렉토리)

2. Train code implementation
```
    python train.py
```

3. Inference from pretrained weight (반드시 inference_3으로 실행시켜주세요!)
```
    python inference_3.py
```

4. Inference by using trained model in this implementation
- 먼저 `inference.py` 내의 72번째줄 pretrained_path 변수를 아래와 같이 변경한다.
```
    pretrained_path = os.path.join(root_path, 'saved_model')
    # pretrained_path = os.path.join( root_path , 'pretrained_model')
```
