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

	//// 가중치 파라미터 : 184500 노드 수 : 380
	//UInt32 node_count0[] = { 50, 10 };                // fcn 레이어 노드 수,   fcn layer 수 만큼
	//Activation_Function activation0[] = { Relu, Softmax };       // 활성화 함수,          전체 layer 수 만큼
	//Layer layer0[] = { Fully_Connected, Fully_Connected }; // 레이어 종류 순서대로, 전체 레이어 수 만큼
	//UInt32 layer_count0 = 2;                                   // 모든 레이어의 수
	//UInt32 channel_count0[] = { 20 };                 // 각 레이어 별 채널 수, conv layer 수 만큼
	//UInt32 cnn_kernel_size0[] = { 3 };                      // 커널 사이즈,          conv layer 수 만큼
	//UInt32 stride0[] = { 1, 1, 1 };                      // 보폭 크기,            conv layer 수 만큼
	//UInt32 pooling_window_size0[] = { 2 };                            // 풀링 window size,     pooling layer 수 만큼
	//Float64 learning_rate0 = 0.002;                               // 학습률
	//Metric metric0 = softmax_cross_entropy;               // 손실 함수

	//Model_Input input0;

	//input0.layer = layer0;
	//input0.layer_count = layer_count0;

	//input0.node_count = node_count0;
	//input0.padding = True;

	//input0.channel_count = channel_count0;
	//input0.cnn_kernel_size = cnn_kernel_size0;
	//input0.stride = stride0;

	//input0.pooling_window_size = pooling_window_size0;

	//input0.activation = activation0;
	//input0.learning_rate = learning_rate0;
	//input0.metric = softmax_cross_entropy;
	//printf("conv pooling FCN layer\n");
	//Model model0 = create_Model(input0);
	////printf("weight 수 : %d, node 수 : %d \n", model.initializer.all_weights_count, model.initializer.all_nodes_count); // fcn 가중치만 포함

	//train_Model(&model0, &trainData, &testData, 10);
	//printf("테스트 데이터\n");

	//accuracy_score(&testData, &model0);
	//model_destroy(&model0);

	//printf("----------------------------------------------\n");

	//// 가중치 파라미터 : 184500 노드 수 : 380
	//UInt32 node_count1[] = { 50, 10 };                // fcn 레이어 노드 수,   fcn layer 수 만큼
	//Activation_Function activation1[] = { Relu, Relu, Relu, Softmax };       // 활성화 함수,          전체 layer 수 만큼
	//Layer layer1[] = { Convolution, MaxPooling, Fully_Connected, Fully_Connected }; // 레이어 종류 순서대로, 전체 레이어 수 만큼
	//UInt32 layer_count1 = 4;                                   // 모든 레이어의 수
	//UInt32 channel_count1[] = { 20 };                 // 각 레이어 별 채널 수, conv layer 수 만큼
	//UInt32 cnn_kernel_size1[] = { 3 };                      // 커널 사이즈,          conv layer 수 만큼
	//UInt32 stride1[] = { 1, 1, 1 };                      // 보폭 크기,            conv layer 수 만큼
	//UInt32 pooling_window_size1[] = { 2 };                            // 풀링 window size,     pooling layer 수 만큼
	//Float64 learning_rate1 = 0.075;                               // 학습률
	//Metric metric1 = softmax_cross_entropy;               // 손실 함수

	//Model_Input input1;

	//input1.layer = layer1;
	//input1.layer_count = layer_count1;

	//input1.node_count = node_count1;
	//input1.padding = True;

	//input1.channel_count = channel_count1;
	//input1.cnn_kernel_size = cnn_kernel_size1;
	//input1.stride = stride1;

	//input1.pooling_window_size = pooling_window_size1;

	//input1.activation = activation1;
	//input1.learning_rate = learning_rate1;
	//input1.metric = softmax_cross_entropy;
	//printf("conv pooling FCN layer\n");
	//Model model1 = create_Model(input1);
	////printf("weight 수 : %d, node 수 : %d \n", model.initializer.all_weights_count, model.initializer.all_nodes_count); // fcn 가중치만 포함

	//train_Model(&model1, &trainData, &testData, 10);
	//printf("테스트 데이터\n");

	//accuracy_score(&testData, &model1);
	//model_destroy(&model1);

	//printf("----------------------------------------------\n");

	// 가중치 파라미터 : 184500 노드 수 : 380
	UInt32 node_count2[] = { 50, 10 };                // fcn 레이어 노드 수,   fcn layer 수 만큼
	Activation_Function activation2[] = { Relu, Relu, Relu, Relu, Relu, Softmax };       // 활성화 함수,          전체 layer 수 만큼
	Layer layer2[] = { Convolution, MaxPooling, Convolution, MaxPooling, Fully_Connected, Fully_Connected }; // 레이어 종류 순서대로, 전체 레이어 수 만큼
	UInt32 layer_count2 = 6;                                   // 모든 레이어의 수
	UInt32 channel_count2[] = { 15, 10 };                 // 각 레이어 별 채널 수, conv layer 수 만큼
	UInt32 cnn_kernel_size2[] = { 3, 3 };                      // 커널 사이즈,          conv layer 수 만큼
	UInt32 stride2[] = { 1, 1 };                      // 보폭 크기,            conv layer 수 만큼
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
	printf("conv pooling FCN layer\n");
	Model model2 = create_Model(input2);
	//printf("weight 수 : %d, node 수 : %d \n", model.initializer.all_weights_count, model.initializer.all_nodes_count); // fcn 가중치만 포함

	train_Model(&model2, &trainData, &testData, 10);
	printf("테스트 데이터\n");

	accuracy_score(&testData, &model2);
	model_destroy(&model2);

	printf("----------------------------------------------\n");

	data_destroy(&trainData);
	data_destroy(&testData);

	return 0;
}