#pragma once
#include"type.h"
#include"layer.h"


Loss MSE(Output* output, Output* ground_truth);

int get_diff(Model* model, UInt8 ground_truth, Output* loss_vector);
//int derivative_metric(Model* model, UInt8 ground_truth, Output* loss_vector, Metric metric);

