# 이름과 학번을 작성해주세요
__author__ = "강채원"
__id__ = "2023000000"

# 넘파이 이외의 어떠한 라이브러리도 호출할 수 없습니다.
import numpy as np


# 아래에 코드를 작성해주세요.
class NaiveBayesClassifier:
    def __init__(self, smoothing=1):
        """
        이 함수는 수정하지 않아도 됩니다.
        """
        self.author = __author__
        self.id = __id__
        self.smoothing=smoothing
        self.epsilon = 1e-10


    def fit(self, x, y):
        """
        이 함수는 수정하지 않아도 됩니다.
        
        실질적인 훈련(확률 계산)이 이뤄지는 부분입니다.
        self.data에 [도큐먼트수 x 단어피쳐수(500)]인 넘파이 행렬이
        self.labels에 각 도큐먼트에 대한 라벨인 [도큐먼트수, ]인 넘파이 행렬이 저장됩니다.

        이후, label_index, prior, likelihood를 순차적으로 계산하여 줍니다.
        """
        self.data = x
        self.labels = y

        self.get_label_index()
        self.get_prior()
        self.get_likelihood()


    def predict(self, x):
        """
        이 함수는 수정하지 않아도 됩니다.
        likelihood, prior를 활용하여 실제 데이터에 대해 posterior를 구하고 확률로 변환하는 함수입니다.
        """
        posterior = self.get_posterior(x)
        return np.argmax(posterior, axis=1)
    

    def get_label_index(self):
        """
        이 함수는 수정하지 않아도 됩니다.

        본 함수가 호출된 이후, self.label_index 변수에 아래와 같은 딕셔너리형태가 저장됩니다.
        self.label_index = {0: array([   1,    4,    5, ..., 3462, 3463, 3464]), 1: array([   0,    2,    3, ..., 3447, 3449, 3453])}
        0번 라벨의 도큐먼트 id가 넘파이 array 형태로, 1번 라벨의 도큐먼트 id가 넘파이 array 형태로 위와 같이 정리됩니다.
        """
        self.label_index = dict()
        self.label_name = set(self.labels)
        for lab in self.label_name:
            self.label_index[lab] = []

        for index, label in enumerate(self.labels):
            self.label_index[label].append(index)

        for lab in self.label_name:
            self.label_index[lab] = np.array(self.label_index[lab])


    def get_prior(self):
        """
        prior를 계산하는 함수입니다. 아무것도 return하지 않습니다.
        본 함수가 처리된 이후, self.prior 변수에 라벨이 key, 라벨에 대한 prior가 value로 들어가도록 하세요.
        self.prior = {0: 0번 라벨 prior[실수값], 1: 1번 라벨 prior[실수값]}

        단 채점시, 라벨이 2개 이상일 수도 있습니다.
        라벨 2개인 경우에서만 잘 작동할 경우, 점수가 부여되지 않습니다.
        """
        self.prior = dict()
        
        # prior(사전 확률)을 구하기 위해, (각 라벨의 빈도수 / 전체 데이터 수)를 구한 후, 각 라벨을 key로 하는 self.prior 생성
        for lab in self.label_name:
            self.prior[lab] = sum(self.labels == lab) / len(self.labels)


    def get_likelihood(self):
        """
        likelihood를 계산하는 함수입니다. 아무것도 return하지 않습니다.
        본 함수가 처리된 이후, self.likelihood에 라벨이 key, 라벨에 대한 단어별 likelihood를 계산하여 value로 넣어주세요.

        예를 들어, 현재 CountVectorizer의 max_features가 500개 이므로, 단어를 500개라 하면
        self.likelihood = {
            0: (라벨 0에 대한 500개 단어에 대한 likelihood로 구성된 numpy어레이),
            1: (라벨 1에 대한 500개 단어에 대한 likelihood로 구성된 numpy어레이)
        }
        로 구성되도록 코딩하세요.

        단, 채점시, 단어의 개수가 500개보다 더 많거나 적을 수도 있고, 라벨도 2개 이상일 수 있습니다.
        500개 단어, 라벨 2개인 경우에서만 잘 작동할 경우, 점수가 부여되지 않습니다.
        """
        self.likelihood = dict()
        
        # 계산을 위해 필요한 값은 각 라벨별 각 단어의 등장 빈도 및 각 라벨별 전체 단어의 등장 수 임
        
        # 각 라벨별 계산
        for lab in self.label_name:
        
            # 각 라벨별 전체 전체 단어의 등장 수
            lab_total_word_cnt = len(self.data[self.label_index[lab]].reshape(-1))
            
            # 각 라벨별 각 단어의 등장 빈도 
            lab_word_cnt = self.data[self.label_index[lab]].sum(axis=0)
            
            # 각 단어별 likelihood 계산 후 할당
            self.likelihood[lab] = lab_word_cnt / lab_total_word_cnt
        

    def get_posterior(self, x):
        """
        self.likelihood와 self.prior를 활용하여 posterior를 계산하는 함수입니다.
        0, 1 라벨에 대한 posterior를 계산하세요.

        @args:
            (도큐먼트수 X 피쳐수)를 입력으로 하는 데이터
        @return:
            (도큐먼트수 X 라벨수)로 하는 확률값
            ex: [[0.7, 0.3],[0.4, 0.6] ... [0.3, 0.7]]
        
        Overflow를 막기위해 log와 exp를 활용합니다. 아래의 식을 고려해서 posterior를 계산하세요.
        posterior 
        = prior * likelihood 
        = exp(log(prior * likelihood))  refer. log(ab) = log(a) + log(b)
        = exp(log(prior) + log(likelihood))

        nan을 막기 위해 possibility 계산시에 분모에 self.epsilon을 더해주세요.

        단, 채점시, 단어의 개수가 500개보다 더 많거나 적을 수도 있고, 라벨도 2개 이상일 수 있습니다.
        500개 단어, 라벨 2개인 경우에서만 잘 작동할 경우, 점수가 부여되지 않습니다.
        """
        
        posteriors = []
        
        # 입력 데이터에 대해 for문으로 각 문서에 대해 계산
        for doc in x:
        
            # 각 문서의 라벨별 확률을 담기 위해 정의
            prob_list = []
            
            for lab in self.label_name:
                # 해당 라벨의 사전확률 저장
                log_prior = np.log(self.prior[lab])
                
                # doc의 각 단어에 대한 likelihood를 통해 전체 likelihood 계산(빈도가 존재하는 단어에 대해서)
                log_likelihood = np.sum(np.log(self.likelihood[lab][doc > 0]))
                
                # log prior와 log likelihood 더한 후, prob_list에 추가
                prob_list.append(np.exp(log_prior + log_likelihood))
            
            prob_list = np.array(prob_list)
            
            # prob_list 내 각 확률 계산 시, epsilon을 더하여 nan 방지
            prob_list = prob_list / (np.sum(prob_list) + self.epsilon)
            
            posteriors.append(prob_list.tolist())
        
        return posteriors