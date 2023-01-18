#pragma once
#include"type.h"
#include"activeFunction.h"
#include"layer.h"



Model create_Model(Model_Input func_input);
int initialize_Model(Model* model);
int train_Model(Model* model, Data* trainData, Data* testData, UInt32 Epoch, Bool saveModel);
int model_destroy(Model* model);
int model_input_destroy(Model* model);

int fcn_forward_propagation(Float64* input_data, Model* model);
int fcn_backward_propagation(Model* model, Float64* input_data, UInt8 label, LR learning_rate, Output* loss_function_derv_fcn, Bool get_derv);

Output convolution(Output* filter, Float64* patch, UInt32 filter_size);
Output get_gradient(Output* kernel, Float64* block, Output* feature_patch, UInt32 block_size, Activation_Function active_function);
Output max_pooling(Output* window, UInt32 window_size, UInt32* max_pooling_address_map, UInt32 address);
Output avg_pooling(Output* window, UInt32 window_size);

int cnn_forward_propagation(Float64* input_data, Model* model);
int cnn_backward_propagation(Model* model, Float64* input_data, LR learning_rate, Output* loss_func_derv_fcn);

int forward_propagation(Float64* input_data, Model* model, Output* fcn_input);
int backward_propagation(Model* model, Float64* input_data, UInt8 label, LR learning_rate, Output* fcn_input);

int save_model(Model* model, Weight* weights, Str* filename);
Model load_model(Str* filename);
int save_weights(Model* model, Weight* weights, Float64 score);

UInt8 predict(Float64* image, Model* model);
Float64 accuracy_score(Data* testdata, Model* model);

void DoProgress(const char label[], int step, int total);
