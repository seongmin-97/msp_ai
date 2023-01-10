#include<float.h>

#include"activeFunction.h"

Output sigmoid(Output output) 
{
	return 1.0 / (1 + exp(-output));
}

// 입력값이 sigmoid 통과한 입력
// http://taewan.kim/post/sigmoid_diff/
Output derivative_sigmoid(Output output)
{
	return output * (1.0 - output);
}

Output relu(Output output)
{
	return (output > 0) ? output : 0.0;
}

// 일단 0에서의 미분은 0으로
Output derivative_relu(Output output)
{
	return (output > 0) ? 1.0 : 0.0;
}

Output softmax(Output* output, UInt32 size, UInt32 idx)
{
	Float64 max = -DBL_MAX;
	for (int i = 0; i < size; i++)
	{
		if (output[i] > max)
			max = output[i];
	}
	Float64 sum = 0.0;
	for (int i = 0; i < size; i++)
		sum += exp(output[i] - max);
	
	return exp(output[idx] - max) / sum;
}

Output activation_function(Output output, Activation_Function activation_function)
{
	Output result = 0;
	switch (activation_function)
	{
		case Sigmoid :
			result = sigmoid(output);
			break;
		case Relu :
			result = relu(output);
			break;
		case None :
			result = output;
			break;
	}
	return result;
}

Output derivative_activation_function(Output output, Activation_Function activation_function)
{
	Output result = 0;
	switch (activation_function)
	{
		case Sigmoid :
			result = derivative_sigmoid(output);
			break;
		case Relu :
			result = derivative_relu(output);
			break;
		case Softmax :
			result = 1.0;
			break;
		case None :
			result = 1.0;
			break;
	}
	return result;
}