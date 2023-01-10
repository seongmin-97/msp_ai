#pragma once

#include<math.h>
#include"type.h"

// 활성화 함수 추가될 때마다 active_function과 active_function_derivative도 추가하자
typedef enum active_function
{
	Sigmoid, Relu, Softmax, None
} Activation_Function;

Output sigmoid(Output output);
Output relu(Output output);

Output softmax(Output* output, UInt32 size, UInt32 idx);

Output derivative_sigmoid(Output output);
Output derivative_relu(Output output);

Output activation_function(Output output, Activation_Function activation_function);
Output derivative_activation_function(Output output, Activation_Function activation_function);