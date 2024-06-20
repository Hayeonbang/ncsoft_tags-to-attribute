# Tags-to-Attribute Model
> 모델에 관한 자세한 설명은 [최종 발표 슬라이드](https://docs.google.com/presentation/d/1JzoVWzqFVYNkx6XQRSdILHz0U5Dt4oICPWp7WAYOvUY/edit?usp=sharing)를 참조해주세요.



## 코드 및 데이터 설명
Pre-trained weight: [Download](https://pytorch.org/get-started/previous-versions/)
```
├──dataset
│   ├── attribute_results.csv
│   ├── tagging_results.csv
│   ├── attribute_preprocessed.csv (train.py 실행 후 저장됨)
├──save
│   ├── predcited_results.csv (eval.py 실행 후 저장됨)
│   ├── trained_model.pth (train.py 실행 후 저장됨)
├── data_preprocess.py   
├── dataset.py
├── model.py  
├── utils.py
├── train.py      
└── eval.py
```

1. Environment Preparation 
(Please install torch according to your [CUDA version](https://pytorch.org/get-started/previous-versions/))
    ```python
    pip install -r requirements.txt
    ```
2. Train the model
    ```python
    python train.py
    ```
3. Evaluate the model
    ```python
    python eval.py
    ```
