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

	printf("----------------------------------------------\n");

	// ����ġ �Ķ���� : 184500 ��� �� : 380
	UInt32 node_count2[] = { 80, 10 };                // fcn ���̾� ��� ��,   fcn layer �� ��ŭ
	Activation_Function activation2[] = { Relu, None, Relu, None, Relu, Softmax };       // Ȱ��ȭ �Լ�,          ��ü layer �� ��ŭ
	Layer layer2[] = { Convolution, MaxPooling, Convolution, MaxPooling, Fully_Connected, Fully_Connected }; // ���̾� ���� �������, ��ü ���̾� �� ��ŭ
	UInt32 layer_count2 = 6;                                   // ��� ���̾��� ��
	UInt32 channel_count2[] = { 40, 30 };                 // �� ���̾� �� ä�� ��, conv layer �� ��ŭ
	UInt32 cnn_kernel_size2[] = { 3, 3 };  // Ŀ�� ������,          conv layer �� ��ŭ
	UInt32 stride2[] = { 1, 1 };           // ���� ũ��,            conv layer �� ��ŭ
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

	Model model2 = create_Model(input2);
	//printf("weight �� : %d, node �� : %d \n", model.initializer.all_weights_count, model.initializer.all_nodes_count); // fcn ����ġ�� ����

	train_Model(&model2, &trainData, &testData, 20);
	model_destroy(&model2);


	data_destroy(&trainData);
	data_destroy(&testData);

	return 0;
}