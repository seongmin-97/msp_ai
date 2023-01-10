#pragma once
#include"type.h"
#include"layer.h"

UInt32 get_before_layer_count(Model* model, Int32 layer);
UInt32 get_node_location(Model_Initializer* initializer, Int32 layer, Int32 node);
UInt32 get_weight_location(Int32 before_layer_node_count, Int32 weight_start_point, Int32 node, Int32 input_node);

int get_kernel(UInt32 kernel_start_point, UInt32 kernel_index, UInt32 kernel_size, UInt32 channel, Weight* all_kernels, Weight* kernel_value);
int get_inverse_kernel(Weight* kernel, Weight* inverse_kernel, UInt32 kernel_size, UInt32 channel_count);

int get_feature_map(UInt32 feature_map_start_point, UInt32 width, UInt32 height, UInt32 channel_idx, Output* all_feature_maps, Output* feature_map_value);
int get_result_feature_map(UInt32 feature_map_start_point, UInt32 width, UInt32 height, UInt32 channel, Output* all_feature_maps, Output* feature_map_value);
UInt32 get_feature_map_address(UInt32 feature_map_start_point, UInt32 feature_map_index, UInt32 width, UInt32 height, UInt32 x, UInt32 y);

int get_block(UInt32 feature_map_start_point, UInt32 channel_count, UInt32 width, UInt32 height, Int32 center_x, Int32 center_y, UInt32 kernel_size, UInt32 padding, Output* all_feature_map, Output* block);
int get_patch(UInt32 feature_map_start_point, UInt32 channel_count, UInt32 width, UInt32 height, Int32 channel_idx, UInt32 patch_size, Int32 top_left_x, Int32 top_left_y, Output* all_feature_map, Output* block, Output* patch, Bool get_block);

int get_output_node(Model* model, Output* output);
int get_diff(Model* model, UInt8 ground_truth, Output* loss_vector);