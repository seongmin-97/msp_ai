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

	UInt32* weight_start_point; // idx가 layer, layer 별 weight 시작 위치
	UInt32* node_start_point;   // idx가 layer, layer 별 node 시작 위치

	UInt32 cnn_weights_count;
	UInt32 cnn_feature_map_size;

	UInt32* kernel_start_point;
	UInt32* feature_map_start_point;

	UInt32* layer_index;        // 예를 들어, (Conv, Pooling, Conv, Pooling, Conv, Conv, Pooling, FCN, FCN) 레이어로 구성될 경우
}Model_Initializer;             //            [0,    1,       2,    3,       4,    5,    6,       0,   1] 반환, 즉, 각 종류마다의 인덱스를 한 번에 통합 (conv와 pooling은 같은 종류로 분류)

typedef struct Input
{
	Layer* layer;
	UInt32 layer_count;

	UInt32* node_count;              // idx가 layer
	Bool padding;

	UInt32* channel_count;			 // conv layer 개수 만큼
	UInt32* cnn_kernel_size;         // idx가 layer (conv layer 개수 만큼)
	UInt32* stride;                  // idx가 layer (conv layer 개수 만큼)

	UInt32* pooling_window_size;     // idx가 layer (pooling layer 개수 만큼)

	LR learning_rate;
	Activation_Function* activation; // idx가 layer
	Metric metric;
	//UInt32 batch_size;
}Model_Input;

typedef struct fcn_params
{
	Weight* weights; // idx가 개별 weights, weights는 row가 현재 노드 수 column이 이전 노드 수 -> w * x = y
	Output* nodes;   // idx가 개별 node
}FCN_Parameters;

typedef struct cnn_params
{
	Weight* kernels;                 // conv layer 개수 만큼
	Output* feature_map;             // conv layer + pooling layer 개수 만큼
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
