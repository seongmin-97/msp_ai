#pragma warning(disable:4996)
// c2440 ���� (�̹��� �ҷ����� �ܰ�) : ������Ʈ -> �Ӽ� -> �����Ӽ� -> C/C++ -> ��� â �ؼ���� �ƴϿ�

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

	//// ����ġ �Ķ���� : 184500 ��� �� : 380
	//UInt32 node_count0[] = { 50, 10 };                // fcn ���̾� ��� ��,   fcn layer �� ��ŭ
	//Activation_Function activation0[] = { Relu, Softmax };       // Ȱ��ȭ �Լ�,          ��ü layer �� ��ŭ
	//Layer layer0[] = { Fully_Connected, Fully_Connected }; // ���̾� ���� �������, ��ü ���̾� �� ��ŭ
	//UInt32 layer_count0 = 2;                                   // ��� ���̾��� ��
	//UInt32 channel_count0[] = { 20 };                 // �� ���̾� �� ä�� ��, conv layer �� ��ŭ
	//UInt32 cnn_kernel_size0[] = { 3 };                      // Ŀ�� ������,          conv layer �� ��ŭ
	//UInt32 stride0[] = { 1, 1, 1 };                      // ���� ũ��,            conv layer �� ��ŭ
	//UInt32 pooling_window_size0[] = { 2 };                            // Ǯ�� window size,     pooling layer �� ��ŭ
	//Float64 learning_rate0 = 0.002;                               // �н���
	//Metric metric0 = softmax_cross_entropy;               // �ս� �Լ�

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
	////printf("weight �� : %d, node �� : %d \n", model.initializer.all_weights_count, model.initializer.all_nodes_count); // fcn ����ġ�� ����

	//train_Model(&model0, &trainData, &testData, 10);
	//printf("�׽�Ʈ ������\n");

	//accuracy_score(&testData, &model0);
	//model_destroy(&model0);

	//printf("----------------------------------------------\n");

	//// ����ġ �Ķ���� : 184500 ��� �� : 380
	//UInt32 node_count1[] = { 50, 10 };                // fcn ���̾� ��� ��,   fcn layer �� ��ŭ
	//Activation_Function activation1[] = { Relu, Relu, Relu, Softmax };       // Ȱ��ȭ �Լ�,          ��ü layer �� ��ŭ
	//Layer layer1[] = { Convolution, MaxPooling, Fully_Connected, Fully_Connected }; // ���̾� ���� �������, ��ü ���̾� �� ��ŭ
	//UInt32 layer_count1 = 4;                                   // ��� ���̾��� ��
	//UInt32 channel_count1[] = { 20 };                 // �� ���̾� �� ä�� ��, conv layer �� ��ŭ
	//UInt32 cnn_kernel_size1[] = { 3 };                      // Ŀ�� ������,          conv layer �� ��ŭ
	//UInt32 stride1[] = { 1, 1, 1 };                      // ���� ũ��,            conv layer �� ��ŭ
	//UInt32 pooling_window_size1[] = { 2 };                            // Ǯ�� window size,     pooling layer �� ��ŭ
	//Float64 learning_rate1 = 0.075;                               // �н���
	//Metric metric1 = softmax_cross_entropy;               // �ս� �Լ�

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
	////printf("weight �� : %d, node �� : %d \n", model.initializer.all_weights_count, model.initializer.all_nodes_count); // fcn ����ġ�� ����

	//train_Model(&model1, &trainData, &testData, 10);
	//printf("�׽�Ʈ ������\n");

	//accuracy_score(&testData, &model1);
	//model_destroy(&model1);

	//printf("----------------------------------------------\n");

	// ����ġ �Ķ���� : 184500 ��� �� : 380
	UInt32 node_count2[] = { 50, 10 };                // fcn ���̾� ��� ��,   fcn layer �� ��ŭ
	Activation_Function activation2[] = { Relu, Relu, Relu, Relu, Relu, Softmax };       // Ȱ��ȭ �Լ�,          ��ü layer �� ��ŭ
	Layer layer2[] = { Convolution, MaxPooling, Convolution, MaxPooling, Fully_Connected, Fully_Connected }; // ���̾� ���� �������, ��ü ���̾� �� ��ŭ
	UInt32 layer_count2 = 6;                                   // ��� ���̾��� ��
	UInt32 channel_count2[] = { 15, 10 };                 // �� ���̾� �� ä�� ��, conv layer �� ��ŭ
	UInt32 cnn_kernel_size2[] = { 3, 3 };                      // Ŀ�� ������,          conv layer �� ��ŭ
	UInt32 stride2[] = { 1, 1 };                      // ���� ũ��,            conv layer �� ��ŭ
	UInt32 pooling_window_size2[] = { 2, 2 };                            // Ǯ�� window size,     pooling layer �� ��ŭ
	Float64 learning_rate2 = 0.002;                               // �н���
	Metric metric2 = softmax_cross_entropy;               // �ս� �Լ�

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
	//printf("weight �� : %d, node �� : %d \n", model.initializer.all_weights_count, model.initializer.all_nodes_count); // fcn ����ġ�� ����

	train_Model(&model2, &trainData, &testData, 10);
	printf("�׽�Ʈ ������\n");

	accuracy_score(&testData, &model2);
	model_destroy(&model2);

	printf("----------------------------------------------\n");

	data_destroy(&trainData);
	data_destroy(&testData);

	return 0;
}