
#include<stdio.h>
#include<stdlib.h>
#include"address.h"

UInt32 get_before_layer_count(Model* model, Int32 layer)
{
	return (layer == 0) ? model->initializer.input_dimension : model->input.node_count[layer - 1];
}

UInt32 get_node_location(Model_Initializer* initializer, Int32 layer, Int32 node)
{
	return (layer == -1) ? node : initializer->node_start_point[layer] + node;
}
// 이전 레이어의 노드 개수, 현재 레이어의 가중치가 시작되는 위치, 현재 layer index, 현재 layer에서 node index, 이전 layer node index
UInt32 get_weight_location(Int32 before_layer_node_count, Int32 weight_start_point, Int32 node, Int32 input_node) 
{
	return weight_start_point + node * before_layer_node_count + input_node;
}
// kernel_value에 가져올 kernel의 주소를 다 알려줌 kernel_index는 해당 레이어의 커널 중에서 몇 번째 커널을 가져올 것인가, 이전 channel 개수 (커널의 두께), all_kernels는 모든 kernel이 저장되어 있는 포인터
int get_kernel(UInt32 kernel_start_point, UInt32 kernel_index, UInt32 kernel_size, UInt32 channel, Weight* all_kernels, Weight* kernel_value)
{
	UInt32 kernel_block_size = kernel_size * kernel_size * channel;
	UInt32 start_address = kernel_start_point + kernel_index * kernel_block_size;

	for (int i = 0; i < kernel_block_size; i++)
		kernel_value[i] = all_kernels[start_address + i];

	return 0;
}

int get_inverse_kernel(Weight* kernel, Weight* inverse_kernel, UInt32 kernel_size, UInt32 channel_count)
{
	UInt32 kernel_area = kernel_size * kernel_size;
	for (int c = 0; c < channel_count; c++)
		for (int i = 0; i < kernel_area; i++)
			inverse_kernel[c * kernel_area + i] = kernel[c * kernel_area + (kernel_area - 1) - i];

	return 0;
}

int get_feature_map(UInt32 feature_map_start_point, UInt32 width, UInt32 height, UInt32 channel_idx, Output* all_feature_maps, Output* feature_map_value)
{
	UInt32 feature_map_size = width * height;
	UInt32 start_address = feature_map_start_point + channel_idx * feature_map_size;

	for (int i = 0; i < feature_map_size; i++)
		feature_map_value[i] = all_feature_maps[start_address + i];

	return 0;
}

int get_result_feature_map(UInt32 feature_map_start_point, UInt32 width, UInt32 height, UInt32 channel, Output* all_feature_maps, Output* feature_map_value)
{
	UInt32 feature_map_size = width * height * channel; // 블록 크기
	UInt32 start_address = feature_map_start_point;

	for (int i = 0; i < feature_map_size; i++)
		feature_map_value[i] = all_feature_maps[start_address + i];

	return 0;
}
// 시작 포인트, channel index (해당 레이어에서 몇 번째 feature map?), feature map의 너비와 높이, feature map에서 구하고자 하는 좌표 x, y 
UInt32 get_feature_map_address(UInt32 feature_map_start_point, UInt32 feature_map_index, UInt32 width, UInt32 height, UInt32 x, UInt32 y)
{
	UInt32 feature_map_size = width * height;
	UInt32 address = feature_map_start_point + feature_map_index * feature_map_size + y * width + x;

	return address;
}
// 시작 포인트, channel count (해당 레이어의 채널 수), feature map의 너비와 높이, 블록의 중심 좌표 x, y (2차원 단에서), 커널사이즈, padding 할 값, 모든 feature map이 있는 포인터, 결과를 저장할 block 포인터
int get_block(UInt32 feature_map_start_point, UInt32 channel_count, UInt32 width, UInt32 height, Int32 center_x, Int32 center_y, UInt32 kernel_size, UInt32 padding, Output* all_feature_map, Output* block)
{
	Int32 half_of_kernel = (kernel_size - 1) / 2;
	for (int channel = 0; channel < channel_count; channel++)
		for (int y = -half_of_kernel; y <= half_of_kernel; y++)
			for (int x = -half_of_kernel; x <= half_of_kernel; x++)
			{
				if (center_x + x < 0 || center_x + x >= width || center_y + y < 0 || center_y + y >= height)
				{
					block[channel * kernel_size * kernel_size + (y + half_of_kernel) * kernel_size + (x + half_of_kernel)] = padding;
				}
				else 
				{
					UInt32 address = get_feature_map_address(feature_map_start_point, channel, width, height, center_x + x, center_y + y);
					block[channel * kernel_size * kernel_size + (y + half_of_kernel) * kernel_size + (x + half_of_kernel)] = all_feature_map[address];
				}
			}

	return 0;
}
// 시작 포인트, channel count (해당 레이어의 채널 수), feature map의 너피와 높이, 추출하고자 하는 patch의 index, patch의 크기, 왼쪽 상단의 x, y 좌표, 모든 feature map이 있는 포인터, block 데이터가 있다면 block 포인터에 입력, 결과를 저장할 patch 포인터, block 추출 실행 여부
int get_patch(UInt32 feature_map_start_point, UInt32 channel_count, UInt32 width, UInt32 height, Int32 channel_idx, UInt32 patch_size, Int32 top_left_x, Int32 top_left_y, Output* all_feature_map, Output* block, Output* patch, Bool get_block)      // block 포인터에 유의미한 값이 있으면 추출할 필요 x
{
	// 1. 블록 추출 2. patch 추출
	if (get_block == True) 
	{
		for (int channel = 0; channel < channel_count; channel++)
			for (int y = 0; y < patch_size; y++)
				for (int x = 0; x < patch_size; x++)
				{
					if (top_left_x + x >= width || top_left_y + y >= height)
					{
						block[channel * patch_size * patch_size + y * patch_size + x] = 0;
					}
					else
					{
						UInt32 address = get_feature_map_address(feature_map_start_point, channel, width, height, top_left_x + x, top_left_y + y);
						block[channel * patch_size * patch_size + y * patch_size + x] = all_feature_map[address];
					}
				}
	}

	for (int y = 0; y < patch_size; y++)
		for (int x = 0; x < patch_size; x++)
			patch[y * patch_size + x] = block[channel_idx * patch_size * patch_size + y * patch_size + x];

	return 0;
}

int get_output_node(Model* model, Output* output)
{
	UInt32 output_layer_start = model->initializer.node_start_point[model->initializer.FCN_layer_count - 1];
	for (int i = 0; i < LABEL; i++)
		output[i] = model->parameters.FCN.nodes[output_layer_start + i];

	return 0;
}

int get_diff(Model* model, UInt8 ground_truth, Output* loss_vector)
{
	if (LABEL != model->input.node_count[model->initializer.FCN_layer_count - 1])
	{
		printf("Label 개수와 출력 계층 node의 개수가 일치하지 않습니다.");
		return -1;
	}
	UInt32 ouput_layer_start = model->initializer.node_start_point[model->initializer.FCN_layer_count - 1];

	for (int idx = 0; idx < LABEL; idx++)
	{
		Output output = model->parameters.FCN.nodes[ouput_layer_start + idx];
		loss_vector[idx] = (idx == ground_truth) ? output - 1.0 : output - 0.0;
	}

	return 0;
}