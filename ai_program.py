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


def load_data1():
    # 데이터1 불러오기 및 전처리하는 함수
    # X1_train, X1_test, y1_train, y1_test 반환
    pass

def load_data2():
    # 데이터2 불러오기 및 전처리하는 함수
    # X2_train, X2_test, y2_train, y2_test 반환
    pass

def load_data3():
    # 데이터3 불러오기 및 전처리하는 함수
    # X3_train, X3_test, y3_train, y3_test 반환
    pass

def plot_data():
    # 원하는 형태로 데이터를 가공해서 웹에 뿌리는 함수
    pass

def make_validation_data():
    # 검증데이터를 생성하고 반환하는 함수 
    # X_val, y_val 반환
    pass

class model:
    def __init__(self):
        # 최적 파라미터로 파라미터 속성초기화
        # 모델이 있는데 파라미터를 따로 속성으로 선언하는 이유
        #   => 웹 배포시 슬라이드로 조정할때 편하게 하기위함
        """
        self.param1 = ..
        self.param2 = ..
        혹은
        self.params = {param1:.. , param2:..}
        """

        # if 모델있으면: 불러오기만
        #     self.model = pickle.load(model)
        # else 모델없으면: 모델만들고 학습
        #     self.model = model(**params)
        #     self.train(model)
        pass
    

    def train(self):
        # 인자로 모델을 받아서 훈련
        # 훈련된 모델, 학습소요시간 반환
        pass
    

    def predict(self):
        # 인자로 모델과 feature를 받아서 결과 도출
        # 결과는 모델마다 메소드 오버라이딩 해서 사용
        pass

    
    def performance(self):
        # 인자로 모델을 받아서 classfication report혹은 필요한 metric반환
        # 모델마다 메소드 오버라이딩 해서 사용
        pass

    
    def save_model(self):
        # 모델을 웹서버에 저장
        # .pkl로 저장. 텐서플로우모델이면 .keras로 저장
        pass


# 각 모델 클래스는 위에서 선언한 model을 상속받아 필요한것만 오버라이딩
class model1(model):
    pass

class model2(model):
    pass

class model3(model):
    # 모델3은 인스턴스안에 이진분류, 다중분류모델이 차례로 들어가므로, 오버라이딩 반드시 필요
    pass


# 웹과의 상호작용을 하는 함수
def main():
    """동작순서 간단히
    1. 데이터 불러오기
    2. 모델 인스턴스화
    3. 루프로 상호작용 반복
    4. 특정 조건이 되면 종료
    """

# py 실행하면 main()실행
if __name__() == '__main__':
    main()
    # main() 종료 이후 서버에 로그 남기기? (여유되면)