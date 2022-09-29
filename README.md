# 2022 kt dev chellenge X Salmon-T5

![Salmon-T5_Logo](https://user-images.githubusercontent.com/53106649/191055714-15321db0-78ce-4210-9ee2-51268c65f24e.png)


## preprocess.py  
제일 첫 단계에 사용하는 스크립트로, 최초 txt 데이터부터, 데이터프레임까지의 전처리를 다룹니다  
위쪽에 있는 함수들은 소기능 만을 담당하며, 맨 아래에 있는 get_train_df, get_test_df를 사용하면  
모든 전처리가 완료된 데이터 프레임을 반환합니다.  

## data.py  
전처리된 데이터를 바탕으로, 데이터를 Stratified 하게 샘플링해서 validation data를 추출하고, 데이터프레임에서 허깅페이스 데이터셋으로 변환합니다.  
get_train_valid_ds를 사용하면 모든 기능을 활용합니다.  

## trainer_metric.py  
trainer에서 사용하기 위해 만들어진 metric 스크립트 입니다.  
trainer에서 지정한 step에 따라 eval 진행간에 사용하며, trainer에 metric이 입력된 상태에서 trainer 객체로  
prediction을 진행하면 test loss, metric에 설정된 정확도가 모두 산출됩니다.  
bleu metric은 보조지표로서 활용했으며, f1,acc metric은 대회 기준으로 작성되었습니다

## trainer.py  
trainer.py를 실행하면 학습이 진행되도록 되어 있으며, 추후에 토크나이저나 모델을 변경해서 실험해볼 수 있도록 매개변수를 구성했습니다  
__main__ 아래에는 전처리부터 모델 학습까지의 모든 파이프라인이 있으며
필요에 따라서 get_train_df에 데이터 경로를 잘 지정해주시면 됩니다.  
해당 파일 경로에서 python trainer.py로 실행하면 학습 부터 예측까지 진행됩니다

trainer.py __main__아래에 있는 내용들 입니다.

    # trainer에 사용되는 파라미터
    batch_size = 64 
    gradient_accumulation = 2 # 2가 최적
    num_train_epochs = 3
    learning_rate = 5e-4
    
    # 20000개 데이터중에, train valid fold 나누는 기준
    # 10일 시 10개 폴드이므로, train:18000, valid:2000 비율이 최적
    # stratified k fold에 사용하는 n_split 값
    train_valid_n_split = 10


    # n이면 n+1배 만퀌 증가
    # n이 3이면 원본대비 4배 증가 성능향상 없어서 주석처리
    num_data_augment = 0

    # input text의 전체 길이 설정, 97은 input text의 맥스값으로 설정한 상태 -> prefix때문에 100으로 변경
    input_max_length = 100
    # label text의 전체 길이 설정, 88은 train label의 맥스값으로 설정한 상태 -> prefix때문에 90으로 변경
    label_max_length = 90
    beam_search = True
    
    # 파일들이 save 될 장소를 설정해야함, 안그러면 겹칠 수 있음!
    base_path = base_dir
    
    # 띄어쓰기 필요없음
    # prefix는 input text에, label_prefix는 label text에 추가되어 학습됨
    prefix = 'extract entity:'
    label_prefix = 'extracted entity:'

    # 폴더내에 남길 log message
    log_message = f'input text: {prefix}: \n, label text: {label_prefix}'
    
    # kt 서버내의 모델 path
    base_model_path = '/home/work/team03/model/kt-ulm-base'
    small_model_path = '/home/work/team03/model/kt-ulm-small'
    
    # 타 모델과 비교를 위한 변수
    ket5_base_path = 'KETI-AIR/ke-t5-base'
    ket5_large_path = 'KETI-AIR/ke-t5-large'

    
    # 원하는 모델로 model_path에 입력
    model_path = base_model_path
