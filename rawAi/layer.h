#pragma once
#include"type.h"
#include"activeFunction.h"

typedef enum metric
{
	mse, softmax_cross_entropy
} Metric;

typedef enum layer
{
	Fully_Connected, Convolution, MaxPooling, AvgPooling
} Layer;

typedef struct model_initializer
{
	UInt32 FCN_layer_count;
	UInt32 Conv_layer_count;
	UInt32 Pooling_layer_count;
	UInt32 Max_Pooling_layer_count;

	UInt32 input_dimension;

	UInt32 fcn_weights_count;
	UInt32 fcn_nodes_count;

	UInt32* weight_start_point; // idx�� layer, layer �� weight ���� ��ġ
	UInt32* node_start_point;   // idx�� layer, layer �� node ���� ��ġ

	UInt32 cnn_weights_count;
	UInt32 cnn_feature_map_size;

	UInt32* kernel_start_point;
	UInt32* feature_map_start_point;

	UInt32* layer_index;        // ���� ���, (Conv, Pooling, Conv, Pooling, Conv, Conv, Pooling, FCN, FCN) ���̾�� ������ ���
}Model_Initializer;             //            [0,    1,       2,    3,       4,    5,    6,       0,   1] ��ȯ, ��, �� ���������� �ε����� �� ���� ���� (conv�� pooling�� ���� ������ �з�)

typedef struct Input
{
	Layer* layer;
	UInt32 layer_count;

	UInt32* node_count;              // idx�� layer
	Bool padding;

	UInt32* channel_count;			 // conv layer ���� ��ŭ
	UInt32* cnn_kernel_size;         // idx�� layer (conv layer ���� ��ŭ)
	UInt32* stride;                  // idx�� layer (conv layer ���� ��ŭ)

	UInt32* pooling_window_size;     // idx�� layer (pooling layer ���� ��ŭ)

	LR learning_rate;
	Activation_Function* activation; // idx�� layer
	Metric metric;
	//UInt32 batch_size;
}Model_Input;

typedef struct fcn_params
{
	Weight* weights; // idx�� ���� weights, weights�� row�� ���� ��� �� column�� ���� ��� �� -> w * x = y
	Output* nodes;   // idx�� ���� node
}FCN_Parameters;

typedef struct cnn_params
{
	Weight* kernels;                 // conv layer ���� ��ŭ
	Output* feature_map;             // conv layer + pooling layer ���� ��ŭ
	UInt32* feature_map_width;
	UInt32* feature_map_height;
	UInt32* max_pooling_address_map;
	UInt32* max_pooling_address_map_start_point;
	UInt32 max_pooling_address_map_size;
}CNN_Parameters;


typedef struct params
{
	FCN_Parameters FCN;
	CNN_Parameters CNN;
}Parameters;

typedef struct model
{
	Model_Input input;
	Model_Initializer initializer;
	Parameters parameters; 
}Model;
