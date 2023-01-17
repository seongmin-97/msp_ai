#pragma warning(disable:4996)
// c2440 에러 (이미지 불러오기 단계) : 프로젝트 -> 속성 -> 구성속성 -> C/C++ -> 언어 창 준수모드 아니오

#include<stdio.h>
#include"loadData.h"
#include"model.h"
#include"metric.h"
#include"activeFunction.h"
#include"layer.h"

char train_data_path[100] = "D:\\msp_AI\\MNIST\\training\\";
char test_data_path[100] = "D:\\msp_AI\\MNIST\\testing\\";

int main()
{
	Data trainData;
	printf("Start Read Train Data\n");
	read_PNG(&trainData, train_data_path);
	printf("\n\n");
	Data testData;
	printf("Start Read Test Data\n");
	read_PNG(&testData, test_data_path);

	printf("----------------------------------------------\n");

	// 가중치 파라미터 : 184500 노드 수 : 380
	UInt32 node_count2[] = { 80, 10 };                // fcn 레이어 노드 수,   fcn layer 수 만큼
	Activation_Function activation2[] = { Relu, None, Relu, None, Relu, Softmax };       // 활성화 함수,          전체 layer 수 만큼
	Layer layer2[] = { Convolution, MaxPooling, Convolution, MaxPooling, Fully_Connected, Fully_Connected }; // 레이어 종류 순서대로, 전체 레이어 수 만큼
	UInt32 layer_count2 = 6;                                   // 모든 레이어의 수
	UInt32 channel_count2[] = { 40, 30 };                 // 각 레이어 별 채널 수, conv layer 수 만큼
	UInt32 cnn_kernel_size2[] = { 3, 3 };  // 커널 사이즈,          conv layer 수 만큼
	UInt32 stride2[] = { 1, 1 };           // 보폭 크기,            conv layer 수 만큼
	UInt32 pooling_window_size2[] = { 2, 2 };                            // 풀링 window size,     pooling layer 수 만큼
	Float64 learning_rate2 = 0.002;                               // 학습률
	Metric metric2 = softmax_cross_entropy;               // 손실 함수

	Model_Input input2;

	input2.layer = layer2;
	input2.layer_count = layer_count2;

	input2.node_count = node_count2;
	input2.padding = True;

	input2.channel_count = channel_count2;
	input2.cnn_kernel_size = cnn_kernel_size2;
	input2.stride = stride2;

	input2.pooling_window_size = pooling_window_size2;

	input2.activation = activation2;
	input2.learning_rate = learning_rate2;
	input2.metric = softmax_cross_entropy;

	Model model2 = create_Model(input2);
	//printf("weight 수 : %d, node 수 : %d \n", model.initializer.all_weights_count, model.initializer.all_nodes_count); // fcn 가중치만 포함

	train_Model(&model2, &trainData, &testData, 20);
	model_destroy(&model2);


	data_destroy(&trainData);
	data_destroy(&testData);

	return 0;
}