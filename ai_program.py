## ai_program.py 구조초안 ##

# 각 데이터는 load_data 함수를 통해 불러올 수 있다.(전처리 포함)
# 데이터를 plot_data 함수를 통해 웹에서 원하는 방식으로 표현
# 필요시 검증데이터를 랜덤으로 생성할 수 있다. (VOC 반영)
# 각 모델은 class의 인스턴스로 구현.
# 모델 class는 train과 predict, performance와 save를 메소드로가지고, 파라미터를 속성으로 가진다.

"""데이터 파이프라인
1. 데이터 불러오기
2. 모델인스턴스 생성 (이때 모델을 불러오지 못하면 자동학습)
3. 학습된 모델 인스턴스를 웹과 상호작용 시킨다. (파라미터적용, 훈련, 추론, 성능평가)
4. 웹서버 종료시킬때, 혹은 저장버튼을 누르면 변경된 모델 인스턴스 저장
"""
import time
import joblib
import os
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.utils import compute_sample_weight
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score, recall_score, precision_score
# from sklearn.metrics import precision_recall_curve

from xgboost import XGBRegressor
from xgboost import XGBClassifier

PATH = os.path.dirname(__file__)


# *중요! 각 csv파일은 무조건 data폴더 안에 있어야합니다.
def load_data1(df1=None, mode='split'):
    """
    data1을 불러와서 전처리된 데이터셋을 반환 
    df를 입력하면 해당데이터셋 처리

    전처리:
    1. 'Sex'컬럼 원핫인코딩
    2. 수치형 표준화

    args:
        df1 : 전처리할 데이터프레임
        mode : 'raw'= Xy만분리 'prepro'=Xy분리후 전처리, 'split'=전처리이후 train test분리 
    returns:
        raw : X, y, df1
        prepro : X, y
        split => X_train, X_test, y_train, y_test, data1_ct
    """
    if df1 is None:
        file_path = os.path.join(PATH, 'data/Regression_data.csv')
        df1 = pd.read_csv(file_path)

    num_cols = list(df1.drop(columns=['Sex', 'Rings']))
    cat_cols = ['Sex']

    X = df1.drop('Rings', axis=1)
    y = df1.Rings

    if mode=='raw':
        return X, y, df1

    elif mode == 'split':
        data1_ct = make_column_transformer(
            (OneHotEncoder(sparse_output=False), cat_cols),
            (StandardScaler(), num_cols),
            remainder='passthrough'
        )    
        data1_ct.set_output(transform='pandas')
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=.25,
                                                            random_state=42)
        
        X_train = data1_ct.fit_transform(X_train)
        X_test = data1_ct.transform(X_test)
        return X_train, X_test, y_train, y_test, data1_ct
    
    elif mode == 'prepro':
        data1_ct = joblib.load(os.path.join(PATH, 'data/reg_ct.pkl'))
        X = data1_ct.transform(X)
        return X, y
    

def load_data2(df2=None, mode='split'):
    """
    data2을 불러와서 전처리된 데이터셋을 반환 
    df를 입력하면 해당데이터셋 처리

    전처리:
    1. 수치형 표준화

    args:
        df2 : 전처리할 데이터프레임
        mode : 'raw'= Xy만분리 'prepro'=Xy분리후 전처리, 'split'=전처리이후 train test분리 
    returns:
        raw : X, y, df2
        prepro : X, y
        split => X_train, X_test, y_train, y_test, data2_ct
    """
    if df2 is None:
        file_path = os.path.join(PATH, 'data/binary_classification_data.csv')
        df2 = pd.read_csv(file_path)

    data2_ct = StandardScaler()
    data2_ct.set_output(transform='pandas')

    X = df2.drop(columns='target_class')
    y = df2.target_class

    if mode=='raw':
        return X, y, df2
        
    elif mode=='split':
        X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                            test_size=.25,
                                                            stratify=y,
                                                            random_state=42)
        
        X_train = data2_ct.fit_transform(X_train)
        X_test = data2_ct.transform(X_test)
        return X_train, X_test, y_train, y_test, data2_ct
    
    elif mode == 'prepro':
        data2_ct = joblib.load(os.path.join(PATH, 'data/bin_ct.pkl'))
        X = data2_ct.transform(X)
        return X, y

def load_data3(df3=None, mode='split'):
    """
    data3을 불러와서 전처리된 데이터셋을 반환 
    df를 입력하면 해당데이터셋 처리

    전처리:  
    1. A400컬럼 드랍 drop
    2. 범주형 원핫인코딩
    3. 수치형 표준화

    args:
        df3 : 전처리할 데이터프레임
        mode : 'raw'= Xy만분리 'prepro'=Xy분리후 전처리, 'split'=전처리이후 train test분리 
    returns:
        raw : X, y, df3
        prepro : X, y
        split => X_train, X_test, y_train, y_test, data3_ct
    """
    if df3 is None:
        file_path = os.path.join(PATH, 'data/mulit_classification_data.csv')
        df3 = pd.read_csv(file_path)

    
    y = df3.loc[:,'Pastry':]
    X = df3.drop(columns=y.columns, axis=1)
    y['label'] = np.argmax(y.values, axis=1) # label구분을 위한 컬럼추가

    if mode == 'raw':
        return X, y, df3
    
    X = X.drop(columns='TypeOfSteel_A400')

    # 범주형 : TypeOfSteel_A300, Outside_Global_Index
    cat_cols = ['TypeOfSteel_A300', 'Outside_Global_Index']
    num_cols = list(X.drop(columns=cat_cols))

    # 범주형 ohe, 나머지 표준화
    if mode == 'split':
        data3_ct = make_column_transformer(
            (StandardScaler(), num_cols),
            (OneHotEncoder(sparse_output=False, drop='if_binary'), cat_cols),
            remainder='passthrough'
        )
        data3_ct.set_output(transform='pandas')

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            stratify=y['label'],
                                                            random_state=42)
        # 독립변수 28개, 종속변수 7개
        X_train = data3_ct.fit_transform(X_train)
        X_test = data3_ct.transform(X_test)
        return X_train, X_test, y_train, y_test, data3_ct
    
    elif mode == 'prepro':
        data3_ct = joblib.load(os.path.join(PATH, 'data/multi_ct.pkl'))
        X = data3_ct.transform(X)
        return X, y


# 각 모델 클래스는 위에서 선언한 model을 상속받아 필요한것만 오버라이딩
class Model:
    model_num = None # load_data호출시 모델번호. 
    
    def __init__(self):
        # params : 모델파라미터
        # model : 모델
        # ct : 입력데이터 전처리기
        # features : 독립변수들 리스트
        # target : 종속변수
        X, y, _ = self.load_data(mode='raw')
        self.params = {}
        self.model = None
        self.ct = None
        self.features = list(X)
        self.target = [y.name] if isinstance(y, pd.Series) else list(y)

        match self.model_num:
            case 1:
                name = 'reg'
            case 2:
                name = 'bin'
            case 3:
                name = 'multi'
            case _:
                raise ValueError(f'model_num 이상. 입력값 : {self.model_num}')
            
        # 먼저 전처리기 불러오기
        file_path = os.path.join(PATH, 'data/'+name+'_ct.pkl')
        if os.path.exists(file_path):
            self.ct = joblib.load(file_path)
            print('전처리기 불러오기 성공')
        else:
            print('전처리기 불러오기 실패.. 학습진행')
            self.ct = self.load_data()[-1]
            
        # 모델 불러오기
        file_path = os.path.join(PATH, 'data/'+name+'_BestModel.pkl')
        if os.path.exists(file_path):
            self.model = joblib.load(file_path)
            self.params = self.model.get_params()
            print(f'{name} 모델 불러오기 성공')
        else:
            print(f'{name} 모델 불러오기 실패.. 학습진행')
            self.model, train_time = self.train()
            self.params = self.model.get_params()

            # 학습 소요시간 print. 0.1초보다 작으면 ms단위로
            if train_time < 0.1:
                print(f'모델훈련 완료. 소요시간 {train_time:.2f}')
            else:
                print(f'모델훈련 완료. 소요시간 {train_time*1000:.2f}ms')



    # 각 모델에 맞는 데이터를 불러오는 함수
    @classmethod
    def load_data(cls, df=None, mode='split'):
        match cls.model_num:
            case 1:
                return load_data1(df1=df, mode=mode)
            case 2:
                return load_data2(df2=df, mode=mode)
            case 3:
                return load_data3(df3=df, mode=mode)
            case _:
                raise ValueError(f'model_num 이상. 입력값 : {cls.model_num}')
            

    # 아예 파라미터를 새로 설정하려면 self.params = {..}
    # 기존의 파라미터를 업데이트 하려면 self.update_params(param1=값1, param2=값2..)
    def update_params(self, **params):
        self.params.update(params)
        self.model.set_params(**self.params)
    

    def _to_frame(self, data, target=False):
        """
        내부사용함수. 입력데이터를 모델에 넣을 수 있도록 pd.DataFrame으로 변환해준다.
        target==True이면 타겟데이터변환, False이면 feature변환 오류나면 None반환  

        args:
            data : 변환할 데이터 (array-like)
            target : 타겟데이터를 변환하는건인지 확인
        returns:
            DataFrame : 데이터프레임으로 변경된 데이터
        """
    
        # 이미 DataFrame이면 그대로 반환
        if isinstance(data, pd.DataFrame):
            return data

        def transform(data):
            # X가 DataFrame이 아니면 DataFrame으로 최종변환
            if not isinstance(data, pd.DataFrame):
                # 리스트나 튜플이면 np.ndarray로 변환
                if isinstance(data, (list|tuple)):
                    data = np.array(data)

                if isinstance(data, np.ndarray):
                    # [data1, data2, ..] 이면 [[data1, data2.. ]]로 변환
                    if data.ndim == 1:
                        data = data.reshape(1,-1)
                    elif data.ndim !=2:
                        print(f'입력데이터의 차원이 이상합니다. {data.ndim}차원이 아닌 1차원 혹은 2차원이 되어야합니다.')
                        return None
                    # DataFrame으로 변환
                    data = pd.DataFrame(data)

                if isinstance(data, pd.Series):
                    data = data.to_frame().T
            return data

        data = transform(data=data)
        if data is None:
            return None
       
        if target: 
            # 컬럼숫자가 안맞으면 에러메시지 출력하고 None 리턴
            if data.shape[1] != len(self.target):
                print(f'타겟데이터의 컬럼수가 안맞습니다. {data.shape[1]}개가 아닌 {len(self.features)}개가 되어야합니다.')
                return None
            data.columns = self.target
            return data # y
        else:
            if data.shape[1] != len(self.features):
                print(f'피쳐데이터의 컬럼수가 안맞습니다. {data.shape[1]}개가 아닌 {len(self.features)}개가 되어야합니다.')
                return None
            data.columns = self.features
            return data # X


    def train(self, params=None, df=None):
        """
        하이퍼파라미터와 데이터를 받아서 학습된 모델을 반환
        
        args:
            params : 파라미터 목록(dictionary)
            df : 학습할 데이터프레임 혹은 np.array
                (없을시 csv불러와서 학습)
        returns:
            model : xgb 모델
            train_time : 훈련소요시간(초)
        """

        # 데이터셋 준비
        if df is None:
            X_train, X_test, y_train, y_test = self.load_data()[:4]
        else:
            X_train, X_test, y_train, y_test = self.load_data(df=df)[:4]
    
        # 모델 선언
        if self.model_num == 1:
            model = XGBRegressor(random_state=42,
                                objective='reg:squarederror',
                                tree_method='hist',
                                eval_metric='rmse',
                                early_stopping_rounds=30)
            weight = None
        elif self.model_num == 2:
            model = XGBClassifier(random_state=42,
                                  objective='binary:logistic',
                                  eval_metric='aucpr',
                                  tree_method='hist',
                                  early_stopping_rounds=50)
            weight = compute_sample_weight(class_weight='balanced', y=y_train)
        model.set_params(**params)

        # 학습
        t1 = time.time()
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  sample_weight=weight,
                  verbose=0)
        t2 = time.time()
        train_time = t2-t1
        
        # 모델과 훈련 소요시간 반환
        return model, train_time
    

    def predict(self, X):
        """
        dataframe 혹은 array-like(np.ndarray, list, tuple) 형태로 데이터를 받아서 예측  

        args:
            X : 입력데이터 (pd.DataFrame or array-like)
        returns:
            pred : 모델 에측값 (np.ndarray)
        """
        if not isinstance(X, pd.DataFrame):
            X = self._to_frame(X, target=False)
            if X is None:
                return None
        
        X = self.load_data(df=X, mode='prepro')[0]
        pred = self.model.predict(X)

        return pred
    

    def performance(self, df=None):
        """
        모델의 성능을 평가하는 함수.

        df가 있으면 해당데이터로, 없으면 기본 검증데이터로 평가
        
        args:
            df : 검증 데이터프레임
        
        returns:
            rmse, acc, r2 : 회귀모델인경우  

            1의 precision, 1의 recall, 1의 f1, acc : 이진분류의 경우
        """
        if df is not None:
            # array-like이면 dataframe으로 변환
            if isinstance(df, (np.ndarray|list|tuple)):
                df = pd.DataFrame(df, columns=self.features+self.target)
            
            X_test, y_true = self.load_data(df, mode='prepro')[:2]
        else:
            _, X_test, _, y_true, _ = self.load_data()
        
        y_pred = self.model.predict(X_test)

        if isinstance(self.model, XGBRegressor):
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            acc = np.mean(1-abs((y_pred - y_true) / y_true))
            r2 = r2_score(y_true, y_pred)
            return rmse, acc, r2
        
        elif isinstance(self.model, XGBClassifier):
            precision = precision_score(y_true, y_pred, pos_label=1)
            recall = recall_score(y_true, y_pred, pos_label=1)
            f1 = f1_score(y_true, y_pred, pos_label=1)
            acc = accuracy_score(y_true, y_pred)
            return precision, recall, f1, acc


    def save_model(self):
        """
        모델을 저장하는 함수
        """
        match self.model_num:
            case 1:
                file_path = os.path.join(PATH, 'data/reg_BestModel.pkl')
            case 2:
                file_path = os.path.join(PATH, 'data/bin_BestModel.pkl')
            case 3:
                file_path = os.path.join(PATH, 'data/multi_BestModel.pkl')
            case _:
                raise ValueError(f'model_num 이상. 값 {self.model_num}')
            
        joblib.dump(self.model, file_path)


    def make_val_data(self, n_samples:int=100, seed:int=None):
        """
        검증용 샘플 데이터프레임을 반환하는 함수

        args:
            n_samples : 뽑아올 샘플(row)수

            seed : 랜덤시드

        returns:
            val_df : 검증용 데이터프레임
        """
        val_df = self.load_data(mode='raw')[2] # df
        if n_samples > val_df.shape[0]:
            print(f"검증 샘플수가 데이터셋보다 많습니다. 최대범위 : {val_df.shape[0]}")
            return None
        
        if seed:
            val_df = val_df.sample(n_samples, random_state=seed)
        else:
            val_df = val_df.sample(n_samples)
        
        return val_df
        
        
class Model1(Model):
    model_num = 1
    
    # train만 오버라이딩
    def train(self, params=None, df=None):
        # params 들어온게 없으면, 최적파라미터로 초기설정 (rmse ~ 2.12)
        if params is None:
            params = {
                'learning_rate': 0.02087425763287998,
                'n_estimators': 1550,
                'max_depth': 17,
                'colsample_bytree': 0.5,
                'reg_lambda': 10.670146505870857,
                'reg_alpha': 0.0663394675391197,
                'gamma': 9.015017136084957
            }
        return super().train(params, df)


class Model2(Model):
    model_num = 2

    # train만 오버라이딩
    def train(self, params=None, df=None):
        # params 들어온게 없으면, 최적파라미터로 초기설정 (f1 ~ 0.871)
        # 현재 튜닝성능이 안나와서 파라미터 개선 예정입니다!
        if params is None:
            params = {
                'learning_rate': 0.03233685808565227,
                'n_estimators': 1200,
                'max_depth': 20,
                'colsample_bytree': 0.5,
                'reg_lambda': 0.004666963217784473,
                'reg_alpha': 0.002792083422830542,
                'gamma': 0.036934880241175236,
                'scale_pos_weight': 7.0
            }
        return super().train(params, df)
    

class Model3(Model):
    model_num = 3

    def __init__(self):
        super().__init__()
        # self.model = Other_Faults 이진분류 모델
        # self.model2 = 나머지 6개 다중분류 모델
        self.params2 = None
        self.model2 = None

        # 모델2 불러오기
        file_path = os.path.join(PATH, 'data/multi_BestModel2.pkl')
        if os.path.exists(file_path):
            self.model2 = joblib.load(file_path)
            self.params2 = self.model.get_params()
            print('multi 모델2 불러오기 성공')
        else:
            print('multi 모델2 불러오기 실패.. 학습진행')
            self.model2, train_time = self.train(case=2)
            self.params2 = self.model2.get_params()

            # 학습 소요시간 print. 0.1초보다 작으면 ms단위로
            if train_time < 0.1:
                print(f'multi 모델A 훈련완료. 소요시간 {train_time:.2f}')
            else:
                print(f'multi 모델B 훈련완료. 소요시간 {train_time*1000:.2f}ms')


    def train(self, case=1, params=None, df=None):
        """
        case = 1이면 1번모델 학습 후 반환, 2이면 2번모델 반환
        """
        # 데이터셋 준비
        if df is None:
            X_train, X_test, y_train, y_test = self.load_data()[:4]
        else:
            X_train, X_test, y_train, y_test = self.load_data(df=df)[:4]
        
        # 이진분류 모델
        if case == 1:
            if params is None:
                params = {
                    'learning_rate': 0.012202469692864067,
                    'n_estimators': 700,
                    'max_depth': 11,
                    'colsample_bytree': 0.7,
                    'reg_lambda': 0.6998936289657887,
                    'reg_alpha': 0.07423629936782049,
                    'gamma': 0.04343495370664839,
                    'scale_pos_weight': 1.2
                }
            X_trainA = X_train
            X_testA = X_test
            y_trainA = y_train.Other_Faults
            y_testA = y_test.Other_Faults

            model = XGBClassifier(random_state=42,
                                  objective='binary:logistic',
                                  eval_metric='error',
                                  tree_method='hist',
                                  early_stopping_rounds=30)
            model.set_params(**params)

            t1 = time.time()
            model.fit(X_trainA, y_trainA,
                      eval_set=[(X_testA, y_testA)],
                      verbose=0)
            t2 = time.time()
            train_time = t2-t1
            return model, train_time
        
        # 다중분류 모델
        elif case == 2:
            if params is None:
                params = {
                    'learning_rate': 0.07522487380833985,
                    'n_estimators': 250,
                    'max_depth': 4,
                    'colsample_bytree':0.6,
                    'reg_lambda': 0.001648272236870337,
                    'reg_alpha': 0.01657588037413299,
                    'gamma': 0.002792373320363197
                }
            y_trainB = y_train[y_train.Other_Faults == 0].label
            y_testB = y_test[y_test.Other_Faults == 0].label
            X_trainB = X_train.loc[y_trainB.index]
            X_testB = X_test.loc[y_testB.index]

            # 다중분류는 scale_pos_weight 설정이 불가함으로 sklearn을 활용,
            # sample weight로 weight 설정
            weight = compute_sample_weight(class_weight='balanced', y=y_trainB)

            model2 = XGBClassifier(random_state=42,
                                   objective='multi:softprob',
                                   eval_metric='aucpr',
                                   tree_method='hist',
                                   early_stopping_rounds=30,
                                   num_class=6)
            model2.set_params(**params)

            t1 = time.time()
            model2.fit(X_trainB, y_trainB,
                       eval_set=[(X_testB, y_testB)],
                       sample_weight=weight,
                       verbose=0)
            t2 = time.time()
            train_time = t2-t1
            return model2, train_time
    

    def predict(self, X, th=0.50012, name_out=True):
        """
        기존 함수와 차이

        th : 이진분류 threshold (기본값 최적)

        name_out : 이름으로 출력할지 라벨로 출력할지
        """
    
        if not isinstance(X, pd.DataFrame):
            X = self._to_frame(X, target=False)
            if X is None:
                return None
        else:
            X = X.copy()

        if 'TypeOfSteel_A400' in X:
            X.drop(columns='TypeOfSteel_A400')
        for col in self.target:
            if col in X:
                X.drop(columns=col, inplace=True)

        try:
            X = self.ct.transform(X)
        except:
            pass
        # 먼저 Other_Faults인지 아닌지 이진분류
        pred = np.where(self.model.predict_proba(X)[:,1] >= th, 6, 0)
        pred = pd.Series(pred, index=X.index, name='label')

        # Other_Faults가 아니라고 예측한 항목들 다중분류
        idx = pred[pred == 0].index
        pred2 = self.model2.predict(X.loc[idx])
        pred2 = pd.Series(pred2, index=idx, name='label')

        # 다중분류한 라벨 원래 pred에 업데이트
        pred.update(pred2)

        if name_out:
            # 라벨을 결함이름으로 변환
            pred = pred.map(dict(zip([0,1,2,3,4,5,6], self.target))).to_numpy()

        return pred
        
    
    def performance(self, df=None):
        """
        모델의 성능을 평가하는 함수.  

        df가 있으면 해당데이터로, 없으면 기본 검증데이터로 평가
        
        args:
            df : 검증 데이터프레임
        
        returns:
            macro precision, macro recall, macro f1, acc : 이진분류의 경우
        """
        if df is not None:
            # array-like이면 dataframe으로 변환
            if isinstance(df, (np.ndarray|list|tuple)):
                df = pd.DataFrame(df, columns=self.features+self.target)
            
            X_test, y_true = self.load_data(df=df, mode='prepro')
            y_true = y_true.label
        else:
            _, X_test, _, y_true, _ = self.load_data()
            y_true = y_true.label
        
        y_pred = self.predict(X_test, name_out=False)

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        return precision, recall, f1, acc
    

    def save_model(self):
        super().save_model()
        file_path = os.path.join(PATH, 'data/multi_BestModel2.pkl')
        joblib.dump(self.model2, file_path)
        

# 웹과의 상호작용을 하는 함수
def main():
    """동작순서 간단히
    1. 데이터 불러오기
    2. 모델 인스턴스화
    3. 루프로 상호작용 반복
    4. 특정 조건이 되면 종료
    """
    print('프로그램 시작.')
    # 모델 선택 단계
    while True:
        print('불러올 모델을 선택하세요. (model1, model2, model3)')
        print('작업취소, 혹은 종료를 원할때 "q"를 입력하세요.')
        inputs = input('입력 : ')
        if inputs == 'model1':
            model = Model1()
            model_name = '회귀'
        elif inputs == 'model2':
            model = Model2()
            model_name = '이진분류'
        elif inputs == 'model3':
            model = Model3()
            model_name = '다중분류'
        elif inputs =='q':
            break
        else:
            print('잘못된 입력입니다. model1, model2, model3 중 하나를 선택하세요.\n')
            continue
        
        val_df= None
        # 작업 단계
        while True:
            
            print(f'{model_name}모델 작업을 선택하세요. (훈련, 검증데이터생성, 예측, 평가, 저장)')
            inputs = input('입력 : ')
            if inputs == 'q':
                break

            elif inputs == '훈련':
                print(f'{model_name}모델 훈련시작.')
                if model.model_num == 3:
                    model.model, train_time = model.train(case=1)
                    print(f'모델A 훈련완료. 소요시간 {train_time}초')
                    model.model2, train_time = model.train(case=2)
                    print(f'모델B 훈련완료. 소요시간 {train_time}초\n')
                else:
                    model.model, train_time = model.train()
                    print(f'모델 훈련완료. 소요시간 {train_time}초\n')
            
            elif inputs == '검증데이터생성':
                print('검증데이터는 csv파일로부터 랜덤으로 생성됩니다.')
                while True:
                    inputs = input('생성할 샘플숫자 입력(정수) : ')
                    if inputs == 'q':
                        break

                    try:
                        inputs = int(inputs)
                    except:
                        print('입력데이터가 정수가 아닙니다. 다시 입력해주세요.\n')
                        continue
                    else:
                        val_df = model.make_val_data(n_samples=inputs)
                        break

                # 오류나서 None을 반환시 
                if val_df is None:
                    print('문제발생. 작업으로 돌아갑니다.')
                else:
                    print(f'검증데이터 생성 성공. shape = {val_df.shape}\n')
            
            elif inputs == '예측':
                print('예측 : 데이터를 feature수에 맞게 입력하거나, 검증세트를 사용하세요.')
                print('검증세트 사용시 "검증" 입력. 수기입력시 [featue1, feature2 .. ] 입력')
                print(f'현재 모델 feature 수 : {len(model.features)}')
                while True:
                    inputs = input('입력 : ')
                    if inputs == '검증':
                        if val_df is None:
                            print('검증세트가 없습니다. 재입력')
                            continue
                        else:
                            pred = model.predict(val_df)

                        if pred is None:
                            print('문제발생. 작업으로 돌아갑니다.')
                        else:
                            print(f'검증세트에 대한 모델 예측값 : {pred}\n')
                            break

                    elif inputs == 'q':
                        break
                    
                    else:
                        try:
                            inputs = ast.literal_eval(inputs)
                        except:
                            print('입력이 이상합니다. 재입력')
                            continue

                        if len(inputs) == len(model.features) and isinstance(inputs, (list|tuple)):
                            pred = model.predict(inputs)
                            if pred is None:
                                print('문제발생. 작업으로 돌아갑니다.')
                            else:
                                print(f'입력값에 대한 모델 예측값 : {pred}\n')
                                break
                        else:
                            print('feature숫자가 안맞거나, 입력이 이상합니다. 재입력')
            
            elif inputs == '평가':
                print('평가 : 검증세트를 평가하거나, 현재성능을 평가')
                print('검증세트를 평가할시 "검증" 입력. 현재성능을 평가할시 공란')
                inputs = input('입력 : ')
                
                if inputs == '검증':
                    if val_df is None:
                        print('검증세트가 없습니다. 작업으로 돌아갑니다.')
                        continue
                    
                    if model.model_num == 1:
                        rmse, acc, r2 = model.performance(val_df)
                        print(f'검증세트 rmse : {rmse:.3f}\tacc : {acc:.3f}\tr2 : {r2:.3f}\n')
                    else:
                        precision, recall, f1, acc = model.performance(val_df)
                        print(f'검증세트 precision : {precision:.3f}\trecall : {recall:.3f}\tf1 : {f1:.3f}\tacc : {acc:.3f}\n')

                elif inputs == '': # 공란입력
                    if model.model_num == 1:
                        rmse, acc, r2 = model.performance()
                        print(f'현재성능 rmse : {rmse:.3f}\tacc : {acc:.3f}\tr2 : {r2:.3f}\n')
                    else:
                        precision, recall, f1, acc = model.performance()
                        print(f'현재성능 precision : {precision:.3f}\trecall : {recall:.3f}\tf1 : {f1:.3f}\tacc : {acc:.3f}\n')
                else:
                    print('입력이 잘못되었습니다. 작업으로 돌아갑니다.')
                
            elif inputs =='저장':
                print(f'{model_name}모델을 data 폴더에 pkl로 저장합니다.')
                try:
                    model.save_model()
                except:
                    print('모델 저장실패. 작업으로 돌아갑니다.')
                else:
                    print('모델 저장 성공')
            
            else:
                print('알맞는 명령어를 입력해주세요 : 훈련, 검증데이터생성, 예측, 평가, 저장 중 하나')
    
        
    
# py 실행하면 main()실행
if __name__ == '__main__':
    main()
    # main() 종료 이후 서버에 로그 남기기? (여유되면)