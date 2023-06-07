# msp_ai

## MNIST DATASET을 사용해 Deep Learning을 C로 구현

## Structure

##### rawAI 폴더 
소스 코드
##### model.bin 
가장 성능이 좋았던 모델을 binary 형태로 저장
##### rawAi.sln
visual studio 프로젝트 파일

## rawAI 폴더 내부

##### loadData.c, loadData.hpp 
데이터 로드 (데이터 로드에서만 이미지를 읽기 위해 opencv 라이브러리 사용)
##### activeFunction.cpp, activeFunction.h 
활성화 함수와 활성화 함수 미분 정의
##### address.cpp, address.h 
포인터 주소 계산 함수 정의 (모든 리스트를 1중 포인터로 계산하였으므로 주소 계산 필수)
##### layer.h 
MLP(fcn) 레이어와 CNN 레이어에 필요한 변수들을 담은 구조체 정의
##### metric.cpp, metric.h 
MSE 함수와 그 미분 정의
##### model.cpp, model.h 
모델을 이루는 레이어에 필요한 변수들 초기화 및 포인터 destroy 정의 및 순전파 역전파 정의, 모델 저장 함수 정의
##### type.h 
프로젝트에서 사용하는 사용자 정의 데이터형 정의 (typedef)
##### weight.cpp, weight.h 
가중치 초기화 기법 정의 (각 레이어 가중치)

## 사용법

- read_PNG() 함수로 데이터셋 read (입력 값 : 데이터를 저장할 Data 구조체, 데이터 경로)<br>
- create_Model() 함수로 모델 생성 (입력 값 : Model_Input 구조체) (main.cpp에서 Model_Input 구조체 정의 참고 (input2 변수))<br>
- train_Model() 함수로 학습 (입력 값 : Model 구조체, 학습 데이터, 테스트 데이터, Epoch, 모델 저장 여부)<br>
- accuracy_score() 함수로 모델의 정확도 평가 (입력 값 : 평가할 데이터 셋, Model 구조체)<br>
