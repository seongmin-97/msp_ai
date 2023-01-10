#pragma once
#include"type.h"

const Float64 AVERAGE = 0.0;
const Float64 STD = 0.01;

int random_weights(Weight* weights, UInt32 count_weight, Float64 range);
int random_gaussian_weights(Weight* weights, UInt32 count_weight);
Float64 gaussianRandom(Float64 average, Float64 std);
int weights_distribution(Weight* weights, UInt32 count_weight);