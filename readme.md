# AIB 18기 Team Project1 레포입니다.

* 팀명 : 부지런한 거위
* 팀원 : 강규욱, 박경훈, 배나연, 조윤서
* 주제 : AI 소프트웨어 업그레이드 프로젝트

* 역할:
0. 공통작업 : EDA, 전처리, 하이퍼파라미터 튜닝
1. 강규욱 : 도메인지식 조사, ppt작성 및 편집
2. 박경훈 : git 레포정리, ai_program.py 작성
3. 배나연 : 도메인지식 조사, ppt작성 및 편집
4. 조윤서 : WEP APP구현, 발표영상 편집

## 프로젝트 전체 목적
* 기존에 ipynb형태로 구성된 레거시코드를 리팩터링
* 레거시코드의 모델성능 고도화

## 세부과제 설명
### 과제1(data1) : 전복의 크기, 무게등의 feature로 Rings예측 (회귀)
### 과제2(data2) : 천체의 특성자료로부터 펄사를 식별 (이진분류)
### 과제3(data3) : 27개의 features로 부터 7종류의 강판결함 분류 (다중분류)
### 과제4 : 위의 3개 과제를 수행하는 모델을 포함하는 ai_program.py 프로그램 작성

## 기존성능 vs 프로젝트 결과물 성능 요약
### 기존 (레거시코드)
* data1 :  rmse=2.46,  acc=0.827
* data2 : precision=0.926, recall=0.789, f1=0.852, acc=0.976
* data3 : acc=0.412
### 개선 (프로젝트 결과물)
* data1 : rmse=2.056, acc=0.862, r2=0.609
* data2 : precision=0.912, recall=0.880, f1=0.896, acc=0.981
* data3 : precision=0.807, recall=0.837, f1=0.819, acc=0.807

## 파일시스템 설명
* EDA.ipynb : EDA를 진행한 결과를 기록
* preprocessing : 여러가지 전처리기법들을 적용한 결과를 기록
* modeling.ipynb : 하이퍼파라미터튜닝을 시도한 결과를 기록
* **ai_program.py : 실제 구동파일**
* Regression_data.csv : 전복 Rings예측 데이터셋 (data1)
* binary_classification_data.csv : 펄사 식별 데이터셋 (data2)
* mulit_classification_data.csv : 강판 결함 분류 데이터셋 (data3)
* (reg, bin, multi)_BestModel.pkl : 각 과제의 최고점수 모델
* (reg, bin, multi)_ct.pkl : 각 데이터를 전처리해주는 전처리기
* (reg, bin, multi)_study.pkl : optuna의 튜닝히스토리가 담겨있는 파일


## requirements
### .ipynb requirements
* python = 3.11.3
* numpy = 1.24.3
* pandas = 1.5.3
* scipy = 1.10.1
* matplotlib = 3.7.1
* scikit-learn = 1.3.0
* xgboost = 1.7.6
* tensorflow = 2.12.1
* optuna = 3.2.0
* plotly = 5.15.0
* shap = 0.42.1

### ai_program.py requirements
* python = 3.11.3
* numpy = 1.24.3
* pandas = 1.5.3
* scikit-learn = 1.3.0
* xgboost = 1.7.6