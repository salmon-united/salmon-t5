# 2022 kt dev chellenge X Salmon-T5

![Salmon-T5_Logo](https://user-images.githubusercontent.com/53106649/191055714-15321db0-78ce-4210-9ee2-51268c65f24e.png)


## preprocess.py  
제일 첫 단계에 사용하는 스크립트로, 최초 txt 데이터부터, 데이터프레임까지의 전처리를 다룹니다  
위쪽에 있는 함수들은 소기능 만을 담당하며, 맨 아래에 있는 get_train_df, get_test_df를 사용하면  
모든 전처리가 완료된 데이터 프레임을 반환합니다  

## data.py  
전처리된 데이터를 바탕으로, 데이터를 Stratified 하게 샘플링해서 validation data를 추출하고, 데이터프레임에서 허깅페이스 데이터셋으로 변환합니다.  
get_train_valid_ds를 사용하면 모든 기능을 활용합니다  

## trainer.py  
trainer.py를 실행하면 학습이 진행되도록 되어 있으며, 추후에 토크나이저나 모델을 변경해서 실험해볼 수 있도록 매개변수를 구성했습니다  
__main__ 아래에는 전처리부터 모델 학습까지의 모든 파이프라인이 있으며
필요에 따라서 get_train_df에 데이터 경로를 잘 지정해주시면 됩니다  
