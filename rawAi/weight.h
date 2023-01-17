#pragma once
#include"type.h"

const Float64 AVERAGE = 0.0;
const Float64 STD = 0.01;

int random_weights(Weight* weights, UInt32 count_weight, Float64 range);
int He_initialization(Weight* weights, UInt32* weight_start_point, UInt32* weight_count, UInt32* input_node_count, UInt32 layer_count);
int random_gaussian_weights(Weight* weights, UInt32 count_weight);
Float64 gaussianRandom(Float64 average, Float64 std);
int weights_distribution(Weight* weights, UInt32 count_weight);