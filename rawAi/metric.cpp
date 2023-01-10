#include<float.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include"metric.h"
#include"layer.h"
#include"type.h"
#include"address.h"

Loss MSE(Output* output, Output* ground_truth)
{
	Loss loss = 0;
	for (int i = 0; i < LABEL; i++)
		loss += 1.0 / 2.0 * (output[i] - ground_truth[i]) * (output[i] - ground_truth[i]);
	return loss;
}



int derivative_metric(Model* model, UInt8 ground_truth, Output* loss_vector, Metric metric)
{
	switch (metric)
	{
	case mse :
		get_diff(model, ground_truth, loss_vector);
		break;
	case softmax_cross_entropy :
		get_diff(model, ground_truth, loss_vector);
		break;
	}
	return 0;
}