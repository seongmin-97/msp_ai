#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<string.h>
#include<time.h>

#include"model.h"
#include"weight.h"
#include"address.h"

char model_file_name[100] = "D:\\msp_AI\\rawAi\\model.bin";

Model create_Model(Model_Input func_input)
{
	Model model;
	model.input = func_input;

	initialize_Model(&model);

	return model;
}

int initialize_Model(Model* model)
{
	printf("모델 생성 중 \n");

	Model_Initializer* initializer = &(model->initializer);
	Model_Input* model_input = &(model->input);
	Parameters* parameters = &(model->parameters);

	// fcn 레이어의 개수, cnn 레이어의 개수 count
	initializer->FCN_layer_count = 0;
	initializer->Conv_layer_count = 0;
	initializer->Pooling_layer_count = 0;
	initializer->Max_Pooling_layer_count = 0;

	initializer->layer_index = (UInt32*)calloc(model_input->layer_count, sizeof(UInt32));
	initializer->input_dimension = WIDTH * HEIGHT;

	initializer->cnn_weights_count = 0;
	initializer->fcn_weights_count = 0;

	for (int layer = 0; layer < model_input->layer_count; layer++)
	{
		switch (model_input->layer[layer])
		{ // layer_index 체크 (필요 없을거 같음 필요해지면 체크)
		case Fully_Connected :
			initializer->layer_index[layer] = initializer->FCN_layer_count;
			initializer->FCN_layer_count++;
			break;
		case Convolution :
			initializer->layer_index[layer] = initializer->Conv_layer_count;
			initializer->Conv_layer_count++;
			break;
		case MaxPooling :
			initializer->Max_Pooling_layer_count++;
		case AvgPooling :
			initializer->layer_index[layer] = initializer->Pooling_layer_count;
			initializer->Pooling_layer_count++;
			break;
		}
	}

	// conv_layer 초기화 (체크해야함)
	if (initializer->Conv_layer_count > 0)
	{
		initializer->kernel_start_point = (UInt32*)calloc(initializer->Conv_layer_count, sizeof(UInt32));


		// kernel weight에 대한 설정
		UInt32 before_layer_weights_count = 0;
		UInt32 before_channel_count = 0;
		UInt32 now_layer_weights_count = 0;
		UInt32* kernel_weight_count = (UInt32*)calloc(initializer->Conv_layer_count, sizeof(UInt32));
		UInt32* input_feature_count = (UInt32*)calloc(initializer->Conv_layer_count, sizeof(UInt32)); // He_initalization 조건에 충족하는, kernel이 한 번에 받는 인풋

		for (int layer = 0; layer < initializer->Conv_layer_count; layer++)
		{
			before_layer_weights_count = (layer == 0) ? 0 : now_layer_weights_count; 
			before_channel_count = (layer == 0) ? CHANNEL : model_input->channel_count[layer - 1];

			now_layer_weights_count = model_input->channel_count[layer] * model_input->cnn_kernel_size[layer] * model_input->cnn_kernel_size[layer] * before_channel_count;
			initializer->kernel_start_point[layer] = (layer == 0) ? 0 : initializer->kernel_start_point[layer - 1] + before_layer_weights_count;

			initializer->cnn_weights_count += now_layer_weights_count;
			kernel_weight_count[layer] = now_layer_weights_count;
			input_feature_count[layer] = model_input->cnn_kernel_size[layer] * model_input->cnn_kernel_size[layer] * ((layer == 0) ? CHANNEL : model_input->channel_count[layer - 1]);
		}
		parameters->CNN.kernels = (Output*)calloc(initializer->cnn_weights_count, sizeof(Output));
		//random_weights(parameters->CNN.kernels, initializer->cnn_weights_count, 0.2);
		He_initialization(parameters->CNN.kernels, initializer->kernel_start_point, kernel_weight_count, input_feature_count, initializer->Conv_layer_count);

		free(kernel_weight_count);
		free(input_feature_count);

		// feature map에 대한 설정
		UInt32 Conv_plus_Pooling = initializer->Conv_layer_count + initializer->Pooling_layer_count;
		parameters->CNN.feature_map_width = (UInt32*)calloc(Conv_plus_Pooling, sizeof(UInt32));
		parameters->CNN.feature_map_height = (UInt32*)calloc(Conv_plus_Pooling, sizeof(UInt32));
		initializer->feature_map_start_point = (UInt32*)calloc(Conv_plus_Pooling, sizeof(UInt32));

		Int32 conv_layer_idx = -1;
		Int32 pooling_layer_idx = -1;
		UInt32 all_feature_map_size = 0;
		UInt32 before_layer_feature_map_size = 0;
		UInt32 now_layer_feature_map_size = 0;
		for (int layer = 0; layer < Conv_plus_Pooling; layer++)
		{
			if (model_input->layer[layer] == Convolution)
			{
				if (model_input->padding)
				{
					conv_layer_idx++;
					parameters->CNN.feature_map_width[layer] = (layer == 0) ? WIDTH / model_input->stride[conv_layer_idx] : (parameters->CNN.feature_map_width[layer - 1]) / model_input->stride[conv_layer_idx];
					parameters->CNN.feature_map_height[layer] = (layer == 0) ? HEIGHT / model_input->stride[conv_layer_idx] : (parameters->CNN.feature_map_height[layer - 1]) / model_input->stride[conv_layer_idx];
				}
				else
				{
					printf("yet...");
				}
			}
			else // pooling
			{
				pooling_layer_idx++;
				parameters->CNN.feature_map_width[layer] = (layer == 0) ? WIDTH / model_input->pooling_window_size[pooling_layer_idx] : (parameters->CNN.feature_map_width[layer - 1]) / model_input->pooling_window_size[pooling_layer_idx];
				parameters->CNN.feature_map_height[layer] = (layer == 0) ? HEIGHT / model_input->pooling_window_size[pooling_layer_idx] : (parameters->CNN.feature_map_height[layer - 1]) / model_input->pooling_window_size[pooling_layer_idx];
			}
			
			before_layer_feature_map_size = (layer == 0) ? 0 : now_layer_feature_map_size;
			now_layer_feature_map_size = parameters->CNN.feature_map_width[layer] * parameters->CNN.feature_map_height[layer] * model_input->channel_count[conv_layer_idx];
			
			initializer->feature_map_start_point[layer] = (layer == 0) ? 0 : initializer->feature_map_start_point[layer - 1] + before_layer_feature_map_size;
			
			all_feature_map_size += now_layer_feature_map_size;
		}
		initializer->cnn_feature_map_size = all_feature_map_size;
		initializer->input_dimension = parameters->CNN.feature_map_width[Conv_plus_Pooling - 1] * parameters->CNN.feature_map_height[Conv_plus_Pooling - 1] * model_input->channel_count[conv_layer_idx];
		parameters->CNN.feature_map = (Output*)calloc(all_feature_map_size, sizeof(Output));
	

		// 먼저, max pooling layer의 feature map 크기를 구하자
		if (initializer->Max_Pooling_layer_count > 0)
			parameters->CNN.max_pooling_address_map_start_point = (UInt32*)calloc(initializer->Max_Pooling_layer_count, sizeof(UInt32));

		UInt32 feature_map_size = 0;
		UInt32 max_pooling_feature_map_size = 0;
		conv_layer_idx = -1;
		Int32 max_pooling_layer_idx = -1;
		for (int i = 0; i < model_input->layer_count; i++)
		{
			switch (model_input->layer[i])
			{
			case Convolution:
				conv_layer_idx++;
				break;
			case MaxPooling:
				max_pooling_layer_idx++;
				parameters->CNN.max_pooling_address_map_start_point[max_pooling_layer_idx] = (max_pooling_layer_idx == 0) ? 0 : parameters->CNN.max_pooling_address_map_start_point[max_pooling_layer_idx - 1] + feature_map_size;
				feature_map_size = parameters->CNN.feature_map_width[i] * parameters->CNN.feature_map_height[i] * model_input->channel_count[conv_layer_idx];
				max_pooling_feature_map_size += feature_map_size;
				break;
			}
		}

		if (initializer->Max_Pooling_layer_count)
		{
			parameters->CNN.max_pooling_address_map = (UInt32*)calloc(max_pooling_feature_map_size, sizeof(UInt32));
			parameters->CNN.max_pooling_address_map_size = max_pooling_feature_map_size;
		}
	}

	// fcn_layer 초기화

	if (initializer->FCN_layer_count > 0)
	{
		UInt32 count_weight = 0;
		UInt32 count_output = 0;

		// 입력으로 넣어준 모델 참고하여, 레이어 개수 노드 개수 이용해 총 가중치 수 계산 + 각 레이어의 weight, output 시작 주소 계산
		initializer->weight_start_point = (UInt32*)calloc(initializer->FCN_layer_count, sizeof(UInt32));
		initializer->node_start_point = (UInt32*)calloc(initializer->FCN_layer_count, sizeof(UInt32));
		UInt32* weight_count = (UInt32*)calloc(initializer->FCN_layer_count, sizeof(UInt32));
		UInt32* input_node_count = (UInt32*)calloc(initializer->FCN_layer_count, sizeof(UInt32));

		for (int layer = 0; layer < initializer->FCN_layer_count; layer++)
		{
			// 이전 계층의 가중치 개수, 현재 계층의 가중치 개수
			UInt32 before_layer_weights_count = (layer == 0) ? 0 : ((layer == 1) ? initializer->input_dimension * model_input->node_count[layer - 1] : model_input->node_count[layer - 2] * model_input->node_count[layer - 1]);
			UInt32 now_layer_weights_count = (layer == 0) ? initializer->input_dimension * model_input->node_count[layer] : model_input->node_count[layer - 1] * model_input->node_count[layer];

			initializer->weight_start_point[layer] = (layer == 0) ? 0 : initializer->weight_start_point[layer - 1] + before_layer_weights_count; // 주소찾기 편하게 각 레이어 계층에서 시작하는 가중치 주소와 노드 주소 찾기
			initializer->node_start_point[layer] = (layer == 0) ? 0 : model_input->node_count[layer - 1] + initializer->node_start_point[layer - 1];

			count_weight += now_layer_weights_count;
			count_output += model_input->node_count[layer];
			weight_count[layer] = now_layer_weights_count;
			input_node_count[layer] = (layer == 0) ? model->initializer.input_dimension : model_input->node_count[layer - 1];
		}

		initializer->fcn_weights_count = count_weight;
		initializer->fcn_nodes_count = count_output;
		model->parameters.FCN.weights = (Weight*)malloc(sizeof(Weight) * count_weight);
		model->parameters.FCN.nodes = (Output*)calloc(count_output, sizeof(Output));

		//random_weights(model->parameters.FCN.weights, count_weight, 0.2);
		He_initialization(model->parameters.FCN.weights, initializer->weight_start_point, weight_count, input_node_count, initializer->FCN_layer_count);
		
		free(weight_count);
		free(input_node_count);
	}

	return 0;
}

int train_Model(Model* model, Data* trainData, Data* testData, UInt32 Epoch, Bool saveModel)
{
	Weight* weights = { 0 };

	if (saveModel)
		weights = (Weight*)calloc(model->initializer.cnn_weights_count + model->initializer.fcn_weights_count, sizeof(Weight));

	for (int epoch = 0; epoch < Epoch; epoch++)
	{
		printf("Epoch %d Training\n", epoch + 1);
		
		//UInt32 max_index = trainData->label_count[trainData->label_max];
		UInt32 used_data[LABEL] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};		// 각 LABEL 별로 현 epoch에서 사용한 데이터의 개수 
		UInt8 complete_train_data_num = 0;								// 학습에 모든 데이터가 사용 완료된 label의 수
		
		UInt8 random_label = 0;											// 학습 데이터가 label에 상관 없이 순서가 정해지도록 label을 랜덤으로 뽑음
		UInt32 cnn_plus_pooling = model->initializer.Conv_layer_count + model->initializer.Pooling_layer_count;

		Float64* input_image = (Float64*)calloc(WIDTH * HEIGHT, sizeof(Float64));
		Output* fcn_input = (Output*)calloc(model->initializer.input_dimension, sizeof(Output));
		UInt32 train_num = 0;

		while (1)
		{
			//srand(time(NULL));
			random_label = rand() % 10;
			UInt32 idx = used_data[random_label];
			UInt32 random_label_count = trainData->label_count[random_label];
			
			// 모든 label 학습 완료 시 종료
			if (complete_train_data_num == 10)
				break;
			
			// 해당 label의 데이터를 모두 사용한 경우
			if (used_data[random_label] == trainData->label_count[random_label])
			{	
				complete_train_data_num++;
				continue;
			}

			Float64* label_data = trainData->data[random_label];

			for (int pixel = 0; pixel < WIDTH * HEIGHT; pixel++)
				input_image[pixel] = label_data[idx * WIDTH * HEIGHT + pixel];
			
			forward_propagation(input_image, model, fcn_input);
			backward_propagation(model, input_image, random_label, model->input.learning_rate, fcn_input);

			used_data[random_label]++;
			train_num++;

			if (train_num % 100 == 0)
				DoProgress("Train ", train_num, 60000);
		}

		free(input_image);
		free(fcn_input);

		Float64 score = accuracy_score(trainData, model);
		printf("Epoch %d Train Data 적용 결과 : %f\n", epoch + 1, score);
		score = accuracy_score(testData, model);
		printf("Epoch %d Test Data 적용 결과 : %f\n", epoch + 1, score);
		printf("------------------------------------------\n");
		
		if (saveModel)
			save_weights(model, weights, score);

	}
	save_model(model, weights, model_file_name);
	free(weights);
	return 0;
}

int fcn_forward_propagation(Float64* input_data, Model* model)
{
	Model_Initializer* initializer = &(model->initializer);
	Model_Input* model_input = &(model->input);
	FCN_Parameters* parameter = &(model->parameters.FCN);

	// 각 레이어 반복, 각 레이어의 노드 반복 : 이중 반복문 안에 가중치 행렬과 input vector와의 계산
	for (int layer = 0; layer < initializer->FCN_layer_count; layer++)                                                   // layer 반복
	{
		for (int node = 0; node < model_input->node_count[layer]; node++)                                                 // node 반복
		{
			Output node_value = 0.0;
			UInt32 before_layer_node_count = get_before_layer_count(model, layer);
			for (int input_node = 0; input_node < before_layer_node_count; input_node++)                                                // 이전 노드 x 가중치 k는 이전 노드의 인덱스
			{
				UInt32 weight_location = get_weight_location(before_layer_node_count, initializer->weight_start_point[layer], node, input_node); // 계산할 가중치 주소 찾기
				Weight current_weight = parameter->weights[weight_location];                                                // location으로 찾은 가중치 주소로 가중치 값을 불러옴

				UInt32 before_node_location = get_node_location(initializer, layer - 1, input_node);
				Output before_node_value = (layer == 0) ? input_data[before_node_location] : parameter->nodes[before_node_location];
				Output weight_sum = before_node_value * current_weight;                            // 현재 node에 더해주어야 할 값
				node_value += weight_sum;                                                                                     // 행렬 연산을 반복문으로 수행하는 것
			}

			UInt32 node_location = get_node_location(initializer, layer, node);
			if (model_input->activation[layer] != Softmax)
			{
				parameter->nodes[node_location] = activation_function(node_value, model_input->activation[layer]);
			}
			else {
				parameter->nodes[node_location] = node_value;
			}
		}

		if (model_input->activation[layer] == Softmax)
		{
			Output* output = (Output*)calloc(LABEL, sizeof(Output));
			get_output_node(model, output);
			for (int node = 0; node < model_input->node_count[layer]; node++)
				parameter->nodes[initializer->node_start_point[layer] + node] = softmax(output, LABEL, node);
			free(output);
		}
	}


	return 0;
}

int fcn_backward_propagation(Model* model, Float64* input_data, UInt8 label, LR learning_rate, Output* loss_function_derv_fcn, Bool get_derv)      
{
	Model_Initializer* initializer = &(model->initializer);
	Model_Input* model_input = &(model->input);

	Weight* weight_update = (Weight*)calloc(model->initializer.fcn_weights_count, sizeof(Weight));
	Float64* buffer_node = (Float64*)calloc(model->initializer.fcn_nodes_count, sizeof(Float64));

	Loss* loss_vector = (Loss*)calloc(LABEL, sizeof(Loss));
	get_diff(model, label, loss_vector);

	// 업데이트 되는 step 계산
	for (int layer = initializer->FCN_layer_count - 1; layer >= 0; layer--) // for(변수 정의; 반복문이 도는 조건 (참이어야 돈다) 별표; 증감)
	{
		UInt32 before_layer_node_count = (layer == 0) ? initializer->input_dimension : model_input->node_count[layer - 1];// 이전 레이어 노드 수
		for (int input_node = 0; input_node < before_layer_node_count; input_node++) // j
		{
			Float64 sigma = 0.0;
			UInt32 before_node_location = (layer > 0) ? initializer->node_start_point[layer - 1] + input_node : input_node;
			for (int node = 0; node < model_input->node_count[layer]; node++) // k
			{
				UInt32 node_location = initializer->node_start_point[layer] + node;                                               // 현재 노드 위치
				Int32  weight_location = initializer->weight_start_point[layer] + node * before_layer_node_count + input_node;     // 계산할 가중치 주소 찾기
				Output before_node_output = (layer == 0) ? input_data[input_node] : model->parameters.FCN.nodes[before_node_location];// 현재 역전파 계산에 사용될 이전 노드 output

				Float64 buffer_value = (layer == (initializer->FCN_layer_count - 1)) ? loss_vector[node] : buffer_node[node_location];// 현재 역전파 계산에 사용될 버퍼 값

				Output node_value = model->parameters.FCN.nodes[node_location];
				Float64 loss_fuction_derv_before_node = buffer_value * derivative_activation_function(model->parameters.FCN.nodes[node_location], model->input.activation[layer]);
				Float64 gradient = before_node_output * loss_fuction_derv_before_node;

				sigma += model->parameters.FCN.weights[weight_location] * loss_fuction_derv_before_node;

				Float64 step = -1.0 * learning_rate * gradient;
				weight_update[weight_location] = step;
			}
			if (layer > 0)
				buffer_node[before_node_location] = sigma;
			else if (layer == 0 && get_derv)
				loss_function_derv_fcn[before_node_location] = sigma;
		}
	}

	// 업데이트
	for (int i = 0; i < initializer->fcn_weights_count; i++)
		model->parameters.FCN.weights[i] += weight_update[i];

	free(loss_vector);
	free(weight_update);
	free(buffer_node);

	return 0;
}

Output get_gradient(Output* kernel, Float64* block, Output* feature_patch, UInt32 block_size, Activation_Function active_function)
{
	Output result = 0;

	for (int i = 0; i < block_size; i++)
		result += kernel[i] * block[i] * derivative_activation_function(feature_patch[i], active_function);

	return result;
}

Output convolution(Output* kernel, Float64* block, UInt32 block_size)
{
	Output result = 0;

	for (int i = 0; i < block_size; i++)
		result += kernel[i] * block[i];

	return result;
}

Output max_pooling(Output* window, UInt32 window_size, UInt32* max_pooling_address_map, UInt32 address)
{
	Output max = window[0];
	for (int i = 0; i < window_size; i++)
		if (max < window[i])
		{
			max = window[i];
			max_pooling_address_map[address] = i;
		}
	
	return max;
}

Output avg_pooling(Output* window, UInt32 window_size)
{
	Output mean = 0;
	for (int i = 0; i < window_size; i++)
		mean += window[i];

	return mean / (Output) window_size;
}
// cnn이 항상 fcn보다 앞에 올 경우만 고려하여 작성함
int cnn_forward_propagation(Float64* input_data, Model* model)
{
	Model_Initializer* initializer = &(model->initializer);
	Model_Input* model_input = &(model->input);
	CNN_Parameters* parameters = &(model->parameters.CNN);

	UInt32 conv_plus_pooling = initializer->Conv_layer_count + initializer->Pooling_layer_count;
	UInt32 before_width, before_height, before_channel, before_feature_map_start_point, width, height, channel, feature_map_start_point, kernel_size, pooling_window_size, stride, kernel_start_point, block_size, patch_size, pixel_location, max_pooling_address_location, max_pooling_address_map_start_point;
	
	Int32 conv_layer_idx = -1;
	Int32 pooling_layer_idx = -1;
	Int32 max_pooling_layer_idx = -1;

	// 불러올 kernel과 block을 저장할 변수
	Output* block = (Output*)calloc(1, sizeof(Output));
	Output* patch = (Output*)calloc(1, sizeof(Output));
	Weight* kernel = (Weight*)calloc(1, sizeof(Weight));

	// feature map은 이전 레이어의 feature map을 불러와야 함
	// kernel은 현재 kernel을 사용
	// 블록과 커널의 크기는 현재 커널의 사이즈 * 현재 커널의 사이즈 * 이전 채널 크기 : 이게 현재 채널 크기 개수만큼 존재
	for (int layer = 0; layer < conv_plus_pooling; layer++)
	{
		Layer layer_type = model_input->layer[layer];

		switch (layer_type)
		{
		case Convolution :
			conv_layer_idx++;
			break;
		case MaxPooling :
			max_pooling_layer_idx++;
		case AvgPooling :
			pooling_layer_idx++;
			break;
		}

		// 불러와야 할 feature map 크기 관련 (이전 레이어와 관련), 저장할 feature map 관련 (현재 레이어와 관련)
		before_width =  (layer == 0) ? WIDTH  : parameters->feature_map_width[layer - 1];
		before_height = (layer == 0) ? HEIGHT : parameters->feature_map_height[layer - 1];
		before_channel = (conv_layer_idx == 0) ? CHANNEL : model_input->channel_count[conv_layer_idx - 1];
		before_feature_map_start_point = (layer == 0) ? 0 : initializer->feature_map_start_point[layer - 1];

		width = parameters->feature_map_width[layer];
		height = parameters->feature_map_height[layer];
		channel = model_input->channel_count[conv_layer_idx];
		feature_map_start_point = initializer->feature_map_start_point[layer];

		// 현재 커널/컨볼루션 정보
		if (layer_type == Convolution)
		{
			kernel_size = model_input->cnn_kernel_size[conv_layer_idx];
			kernel_start_point = initializer->kernel_start_point[conv_layer_idx];
			stride = model_input->stride[conv_layer_idx];
			block_size = kernel_size * kernel_size * before_channel; // before_channel : 두께
			patch_size = kernel_size * kernel_size;
		}
		else        
		{   // pooling layer에는 커널이 없음, 채널은 이전 conv_layer의 채널과 동일
			before_channel = channel;
			stride = model_input->pooling_window_size[pooling_layer_idx];
			pooling_window_size = stride;
			block_size = pooling_window_size * pooling_window_size * channel;
			patch_size = pooling_window_size * pooling_window_size;
			if (layer_type == MaxPooling)
				max_pooling_address_map_start_point = parameters->max_pooling_address_map_start_point[max_pooling_layer_idx];
		}

		block  = (Output*)realloc(block,  block_size * sizeof(Output));
		patch  = (Output*)realloc(patch,  patch_size * sizeof(Output));
		kernel = (Weight*)realloc(kernel, block_size * sizeof(Weight));
		// 이전 블록을 추출, 현재의 커널과 컨볼루션
		for (int y = 0; y < before_height; y += stride)
			for (int x = 0; x < before_width; x += stride)
			{
				if (layer_type == Convolution)
					get_block(before_feature_map_start_point, before_channel, before_width, before_height, x, y, kernel_size, 0, (layer == 0) ? input_data : parameters->feature_map, block);    // featuremap block
				else
					get_patch(before_feature_map_start_point, channel, before_width, before_height, 0, pooling_window_size, x, y, (layer == 0) ? input_data : parameters->feature_map, block, patch, True); //featuremap patch

				for (int c = 0; c < channel; c++)
				{
					switch (layer_type)
					{
					case Convolution :
						pixel_location = get_feature_map_address(feature_map_start_point, c, width, height, x, y);
						get_kernel(kernel_start_point, c, kernel_size, before_channel, parameters->kernels, kernel);
						parameters->feature_map[pixel_location] = activation_function(convolution(kernel, block, block_size), model_input->activation[layer]);
						break;
					case MaxPooling : 
						pixel_location = get_feature_map_address(feature_map_start_point, c, width, height,  x / stride, y / stride);
						max_pooling_address_location = get_feature_map_address(max_pooling_address_map_start_point, c, width, height, x / stride, y / stride);
						get_patch(before_feature_map_start_point, before_channel, before_width, before_height, c, pooling_window_size, x, y, (layer == 0) ? input_data : parameters->feature_map, block, patch, False);
						parameters->feature_map[pixel_location] = max_pooling(patch, patch_size, parameters->max_pooling_address_map, max_pooling_address_location);
						break;
					case AvgPooling :
						pixel_location = get_feature_map_address(feature_map_start_point, c, width, height, x / stride, y / stride);
						get_patch(before_feature_map_start_point, before_channel, before_width, before_height, c, pooling_window_size, x, y, (layer == 0) ? input_data : parameters->feature_map, block, patch, False);
						parameters->feature_map[pixel_location] = avg_pooling(patch, patch_size);
						break;
					}
				}
			}
	}

	free(kernel);
	free(patch);
	free(block);

	return 0;
}

int cnn_backward_propagation(Model* model, Float64* input_data, LR learning_rate, Output* loss_func_derv_fcn)
{
	Model_Initializer* initializer = &(model->initializer);
	Model_Input* model_input = &(model->input);
	CNN_Parameters* parameters = &(model->parameters.CNN);

	Weight* weight_update = (Weight*)calloc(model->initializer.cnn_weights_count, sizeof(Weight));
	Output* buffer_feature_map = (Output*)calloc(model->initializer.cnn_feature_map_size, sizeof(Output));
	
	UInt32 conv_plus_pulling = initializer->Conv_layer_count + initializer->Pooling_layer_count;
	Layer layer_type;

	UInt32 before_width, before_height, before_channel, before_feature_map_start_point, width, height, channel;
	UInt32 feature_map_size, feature_map_start_point, kernel_size, pooling_window_size, stride, kernel_start_point, buffer_block_size, kernel_block_size, patch_size;
	UInt32 pixel_location, max_pooling_address_location, max_pooling_address_map_start_point, max_pooling_address, patch_start_address, gradient_address;
	Int32 window_left, window_right, window_bottom, window_top, padding_size;
	Output activation_derv;

	Int32 conv_layer_idx	    = initializer->Conv_layer_count - 1;
	Int32 pooling_layer_idx     = initializer->Pooling_layer_count;
	Int32 max_pooling_layer_idx = initializer->Max_Pooling_layer_count;

	// 불러올 kernel과 block을 저장할 변수
	Output* feature_block = (Output*)calloc(1, sizeof(Output));
	Output* feature_patch = (Output*)calloc(1, sizeof(Output));
	Output* buffer_block = (Output*)calloc(1, sizeof(Output));
	Output* patch = (Output*)calloc(1, sizeof(Output));
	Output* gradient_patch = (Output*)calloc(1, sizeof(Output));
	Weight* kernel = (Weight*)calloc(1, sizeof(Weight));
	Weight* inverse_kernel = (Weight*)calloc(1, sizeof(Weight));


	for (int layer = conv_plus_pulling - 1; layer >= 0; layer--)
	{
		layer_type = model_input->layer[layer];

		switch (layer_type)
		{
		case MaxPooling:
			max_pooling_layer_idx--;
		case AvgPooling:
			pooling_layer_idx--;
			break;
		}

		before_width = (layer == 0) ? WIDTH : parameters->feature_map_width[layer - 1];
		before_height = (layer == 0) ? HEIGHT : parameters->feature_map_height[layer - 1];
		before_channel = (conv_layer_idx == 0) ? CHANNEL : model_input->channel_count[conv_layer_idx - 1];
		before_feature_map_start_point = (layer == 0) ? 0 : initializer->feature_map_start_point[layer - 1];

		width = parameters->feature_map_width[layer];
		height = parameters->feature_map_height[layer];
		channel = model_input->channel_count[conv_layer_idx];
		feature_map_size = width * height * channel;
		feature_map_start_point = initializer->feature_map_start_point[layer];

		// 현재 커널/컨볼루션 정보
		if (layer_type == Convolution)
		{
			kernel_size = model_input->cnn_kernel_size[conv_layer_idx];
			kernel_start_point = initializer->kernel_start_point[conv_layer_idx];
			stride = model_input->stride[conv_layer_idx];

			buffer_block_size = kernel_size * kernel_size * channel;
			kernel_block_size = kernel_size * kernel_size * before_channel;
			patch_size = kernel_size * kernel_size;
			padding_size = (kernel_size - 1) / 2;

			feature_block =  (Output*)realloc(feature_block, buffer_block_size * sizeof(Output));
			feature_patch =  (Output*)realloc(feature_patch, patch_size * sizeof(Output));
			buffer_block =   (Output*)realloc(buffer_block, buffer_block_size * sizeof(Output));							    // buffer_block   : 현재 feature_map에서 block을 가져오는 것 
			kernel =	     (Weight*)realloc(kernel, kernel_block_size * sizeof(Weight));										// kernel		  : 현재 kernel을 block으로, 두께는 before_channel
			inverse_kernel = (Weight*)realloc(inverse_kernel, kernel_block_size * sizeof(Weight));								// inverse_kernel : 크기는 kernel과 동일
			patch =			 (Output*)realloc(patch, patch_size * sizeof(Output));												// patch	      : 커널을 2차원으로 slice
			gradient_patch = (Output*)realloc(gradient_patch, patch_size * sizeof(Output));								        // gradient_patch : buffer_block을 2차원으로 slice
		}
		else
		{   // pooling layer에는 커널이 없음, 채널은 이전 conv_layer의 채널과 동일
			before_channel = channel;
			stride = model_input->pooling_window_size[pooling_layer_idx];
			pooling_window_size = stride;
			patch_size = pooling_window_size * pooling_window_size;
			buffer_block_size = 1;
			kernel_block_size = 1;
			if (layer_type == MaxPooling)
				max_pooling_address_map_start_point = parameters->max_pooling_address_map_start_point[max_pooling_layer_idx];
		}


		// local gradient 저장
		if (layer == conv_plus_pulling - 1)
			for (int i = 0; i < feature_map_size; i++)
			{
					buffer_feature_map[feature_map_start_point + i] = loss_func_derv_fcn[i] /** derivative_activation_function(parameters->feature_map[feature_map_start_point + i], model_input->activation[layer])*/;
				//else
					//buffer_feature_map[feature_map_start_point + i] *= derivative_activation_function(parameters->feature_map[feature_map_start_point + i], model_input->activation[layer]);
			}
		// 역전파
		
		// buffer update
		for (int y = 0; y < before_height; y += stride)     // 
			for (int x = 0; x < before_width; x += stride)	// x, y는 이전 feature map의 좌표이며 현재 구하는 기울기의 위치
				for (int c = 0; c < before_channel; c++)    // c : 이전 feature_map의 채널 축 좌표, 현재 구하는 기울기의 위치 
				{
					switch (layer_type)
					{
					case MaxPooling:
					{
						max_pooling_address_location = get_feature_map_address(max_pooling_address_map_start_point, c, width, height, x / stride, y / stride);
						max_pooling_address = parameters->max_pooling_address_map[max_pooling_address_location];
						// gradient address : 흘러들어온 미분값을 불러올 주소
						patch_start_address = get_feature_map_address(before_feature_map_start_point, c, before_width, before_height, x, y);
						gradient_address = get_feature_map_address(feature_map_start_point, c, width, height, x / stride, y / stride);
						buffer_feature_map[patch_start_address + max_pooling_address] = buffer_feature_map[gradient_address];

						break;
					}
					case AvgPooling:
					{
						patch_start_address = get_feature_map_address(before_feature_map_start_point, c, before_width, before_height, x, y);
						gradient_address = get_feature_map_address(feature_map_start_point, c, width, height, x / stride, y / stride);

						for (int i = 0; i < patch_size; i++)
							buffer_feature_map[patch_start_address + i] = (buffer_feature_map[gradient_address + i] / (Float64)patch_size);

						break;
					}
					case Convolution:
						if (layer != 0)
						{
							get_block(feature_map_start_point, channel, width, height, x, y, kernel_size, 0, buffer_feature_map, buffer_block);
							get_block(feature_map_start_point, channel, width, height, x, y, kernel_size, 0, parameters->feature_map, feature_block);
							for (int kernel_idx = 0; kernel_idx < channel; kernel_idx++)
							{
								get_patch(feature_map_start_point, channel, width, height, kernel_idx, kernel_size, x, y, buffer_feature_map, buffer_block, gradient_patch, False);
								get_patch(feature_map_start_point, channel, width, height, kernel_idx, kernel_size, x, y, parameters->feature_map, feature_block, feature_patch, False);

								get_kernel(kernel_start_point, kernel_idx, kernel_size, before_channel, parameters->kernels, kernel);
								get_inverse_kernel(kernel, inverse_kernel, kernel_size, before_channel);
								get_patch(kernel_start_point, before_channel, kernel_size, kernel_size, c, kernel_size, 0, 0, parameters->kernels, inverse_kernel, patch, False);
								// gradient address : 저장할 feature map 주소
								gradient_address = get_feature_map_address(before_feature_map_start_point, c, before_width, before_height, x, y);
								buffer_feature_map[gradient_address] += get_gradient(patch, gradient_patch, feature_patch,  patch_size, model_input->activation[layer]);

								for (int i = 0; i < kernel_size; i++)
									for (int j = 0; j < kernel_size; j++)
									{
										Int32 weight_location = kernel_start_point + kernel_idx * kernel_size * kernel_size * before_channel + c * kernel_size * kernel_size + i * kernel_size + j;
										Int32 kernel_center = padding_size;
										Int32 relative_i = i - padding_size;
										Int32 relative_j = j - padding_size;

										Int32 now_feature_map_address = get_feature_map_address(feature_map_start_point, kernel_idx, width, height, x, y);
										Int32 before_feature_map_address = get_feature_map_address(before_feature_map_start_point, c, before_width, before_height, x + relative_j, y + relative_i);

										Output before_feature;
										if (x + relative_j < 0 || y + relative_i < 0 || x + relative_j > width - 1 || y + relative_i > height - 1)
											before_feature = 0.0;
										else
											before_feature = (layer != 0) ? parameters->feature_map[before_feature_map_address] : input_data[before_feature_map_address];

										Output now_gradient = buffer_feature_map[now_feature_map_address];
										activation_derv = derivative_activation_function(parameters->feature_map[now_feature_map_address], model_input->activation[layer]);

										if (activation_derv != 0.0 && before_feature != 0.0)
											weight_update[weight_location] += now_gradient * before_feature * activation_derv;
									}
							}
						}
						break;
					}
				}

		// get delta weight
		if (layer_type == Convolution)
		{
			//UInt32 now_feature_map_address, before_feature_map_address, weight_location, kernel_center;
			//Int32 relative_i, relative_j;
			//Float64 before_feature, now_gradient;
			//for (int kernel_index = 0; kernel_index < channel; kernel_index++) // 커널의 개수 현재 feature map의 idx, 커널 블록 자체의 idx
			//	for (int c = 0; c < before_channel; c++)					   // 커널의 두께, 이전 feature map의 idx, 커널 패치의 idx
			//		for (int i = 0; i < kernel_size; i++)					   // 커널의 y축
			//			for (int j = 0; j < kernel_size; j++)				   // 커널의 x축
			//			{
			//				weight_location = kernel_start_point + kernel_index * kernel_size * kernel_size * before_channel + c * kernel_size * kernel_size + i * kernel_size + j;
			//				kernel_center = padding_size;
			//				relative_i = i - padding_size;
			//				relative_j = j - padding_size;
			//				for (int y = 0; y < height; y++)
			//					for (int x = 0; x < width; x++)
			//					{
			//						now_feature_map_address    = get_feature_map_address(feature_map_start_point, kernel_index, width, height, x, y);
			//						before_feature_map_address = get_feature_map_address(before_feature_map_start_point, c, before_width, before_height, x + relative_j, y + relative_i);
			//						
			//						if (x + relative_j < 0 || y + relative_i < 0 || x + relative_j > width - 1 || y + relative_i > height - 1)
			//							before_feature = 0.0;
			//						else
			//							before_feature = (layer != 0) ? parameters->feature_map[before_feature_map_address] : input_data[before_feature_map_address];
			//						
			//						now_gradient = buffer_feature_map[now_feature_map_address];
			//						activation_derv = derivative_activation_function(parameters->feature_map[now_feature_map_address], model_input->activation[layer]);
	
			//						if (activation_derv != 0.0 && before_feature != 0.0)
			//							weight_update[weight_location] += now_gradient * before_feature * activation_derv;
			//					}
			//			}
			conv_layer_idx--;
		}
	}

	// weight update
	for (int i = 0; i < initializer->cnn_weights_count; i++)
		parameters->kernels[i] += - learning_rate * weight_update[i];

	free(feature_block);
	free(feature_patch);
	free(buffer_feature_map);
	free(weight_update);
	free(kernel);
	free(inverse_kernel);
	free(gradient_patch);
	free(buffer_block);
	free(patch);

	return 0;
}

int forward_propagation(Float64* input_data, Model* model, Output* fcn_input)
{
	UInt32 cnn_plus_pooling = model->initializer.Conv_layer_count + model->initializer.Pooling_layer_count;

	if (cnn_plus_pooling > 0)
	{
		cnn_forward_propagation(input_data, model);

		UInt32 width = model->parameters.CNN.feature_map_width[cnn_plus_pooling - 1];
		UInt32 height = model->parameters.CNN.feature_map_height[cnn_plus_pooling - 1];
		UInt32 channel = model->input.channel_count[model->initializer.Conv_layer_count - 1];
		UInt32 feature_map_start_point = model->initializer.feature_map_start_point[cnn_plus_pooling - 1];

		UInt32 block_size = width * height * channel;
		
		get_result_feature_map(feature_map_start_point, width, height, channel, model->parameters.CNN.feature_map, fcn_input);
		fcn_forward_propagation(fcn_input, model);
	}
	else
	{
		fcn_forward_propagation(input_data, model);
	}

	return 0;
}

int backward_propagation(Model* model, Float64* input_data, UInt8 label, LR learning_rate, Output* fcn_input)
{
	UInt32 cnn_plus_pooling = model->initializer.Conv_layer_count + model->initializer.Pooling_layer_count;

	if (cnn_plus_pooling > 0)
	{
		Output* loss_func_derv_fcn = (Output*)calloc(model->initializer.input_dimension, sizeof(Output));
		fcn_backward_propagation(model, fcn_input, label, learning_rate, loss_func_derv_fcn, True);
		cnn_backward_propagation(model, input_data, learning_rate, loss_func_derv_fcn);
		free(loss_func_derv_fcn);
	}
	else
	{
		Output* output = { 0 };
		fcn_backward_propagation(model, input_data, label, learning_rate, output, False);
	}
	return 0;
}

int save_model(Model* model, Weight* weights, Str* filename)
{
	Model_Input* model_input = &(model->input);
	
	FILE* fp;
	UInt32 buffer = 0;

	fopen_s(&fp, filename, "wb");

	// 1. 총 레이어 수 저장
	buffer = model_input->layer_count;
	fwrite(&buffer, sizeof(UInt32), 1, fp);

	// 2. cnn layer 수 저장
	buffer = model->initializer.Conv_layer_count;
	fwrite(&buffer, sizeof(UInt32), 1, fp);

	// 3. pooling layer 수 저장
	buffer = model->initializer.Pooling_layer_count;
	fwrite(&buffer, sizeof(UInt32), 1, fp);

	// 4. fcn layer 수 저장
	buffer = model->initializer.FCN_layer_count;
	fwrite(&buffer, sizeof(UInt32), 1, fp);

	// 5. 레이어 종류 / 활성화 함수 순서대로 저장
	for (int i = 0; i < model_input->layer_count; i++)
	{
		buffer = model_input->layer[i];
		fwrite(&buffer, sizeof(UInt32), 1, fp);
		buffer = model_input->activation[i];
		fwrite(&buffer, sizeof(UInt32), 1, fp);
	}

	// 6. padding 여부 저장
	buffer = model_input->padding;
	fwrite(&buffer, sizeof(UInt32), 1, fp);

	// 7. channel_count / kernel_size / stride 저장 배열 저장
	for (int i = 0; i < model->initializer.Conv_layer_count; i++)
	{
		buffer = model_input->channel_count[i];
		fwrite(&buffer, sizeof(UInt32), 1, fp);
		buffer = model_input->cnn_kernel_size[i];
		fwrite(&buffer, sizeof(UInt32), 1, fp);
		buffer = model_input->stride[i];
		fwrite(&buffer, sizeof(UInt32), 1, fp);
	}

	// 8. pooling_window_size 저장
	for (int i = 0; i < model->initializer.Pooling_layer_count; i++)
	{
		buffer = model_input->pooling_window_size[i];
		fwrite(&buffer, sizeof(UInt32), 1, fp);
	}

	// 9. node 개수 저장
	for (int i = 0; i < model->initializer.FCN_layer_count; i++)
	{
		buffer = model_input->node_count[i];
		fwrite(&buffer, sizeof(UInt32), 1, fp);
	}

	// 10. 손실 함수 저장
	buffer = model_input->metric;
	fwrite(&buffer, sizeof(UInt32), 1, fp);

	Float64 fBuffer = 0.0;

	// 11. 학습률 저장
	fBuffer = model_input->learning_rate;
	fwrite(&fBuffer, sizeof(Float64), 1, fp);

	// 12. 가중치 저장
	fwrite(weights, sizeof(Weight), model->initializer.cnn_weights_count + model->initializer.fcn_weights_count, fp);

	fclose(fp);
	return 0;
}

Model load_model(Str* filename)
{
	UInt32 buffer = 0;
	Float64 lr = 0;
	Model_Input model_input;
	UInt32 Conv_layer_count, Pooling_layer_count, FCN_layer_count;
	
	FILE* fp;
	fopen_s(&fp, filename, "rb");

	// 1. 총 레이어 수 읽기
	fread(&buffer, sizeof(UInt32), 1, fp);
	model_input.layer_count = buffer;

	// 2. cnn 레이어 수 읽기
	fread(&Conv_layer_count, sizeof(UInt32), 1, fp);

	// 3. pooling 레이어 수 읽기
	fread(&Pooling_layer_count, sizeof(UInt32), 1, fp);

	// 4. fcn 레이어 수 읽기
	fread(&FCN_layer_count, sizeof(UInt32), 1, fp);

	// 5. 레이어 종류 / 활성화 함수 읽기
	model_input.layer = (Layer*)calloc(model_input.layer_count, sizeof(Layer));
	model_input.activation = (Activation_Function*)calloc(model_input.layer_count, sizeof(Activation_Function));
	for (int i = 0; i < model_input.layer_count; i++)
	{
		fread(&buffer, sizeof(UInt32), 1, fp);
		model_input.layer[i] = (Layer) buffer;
		fread(&buffer, sizeof(UInt32), 1, fp);
		model_input.activation[i] = (Activation_Function) buffer;
	}

	// 6. 패딩 여부 읽기
	fread(&buffer, sizeof(UInt32), 1, fp);
	model_input.padding = (Bool) buffer;
	
	// 7. channel_count / kernel_size / stride 배열 읽기
	model_input.channel_count = (UInt32*)calloc(Conv_layer_count, sizeof(UInt32));
	model_input.cnn_kernel_size = (UInt32*)calloc(Conv_layer_count, sizeof(UInt32));
	model_input.stride = (UInt32*)calloc(Conv_layer_count, sizeof(UInt32));
	for (int i = 0; i < Conv_layer_count; i++)
	{
		fread(&buffer, sizeof(UInt32), 1, fp);
		model_input.channel_count[i] = buffer;
		fread(&buffer, sizeof(UInt32), 1, fp);
		model_input.cnn_kernel_size[i] = buffer;
		fread(&buffer, sizeof(UInt32), 1, fp);
		model_input.stride[i] = buffer;
	}

	// 8. pooling window size 읽기
	model_input.pooling_window_size = (UInt32*)calloc(Pooling_layer_count, sizeof(UInt32));
	for (int i = 0; i < Pooling_layer_count; i++)
	{
		fread(&buffer, sizeof(UInt32), 1, fp);
		model_input.pooling_window_size[i] = buffer;
	}

	// 9. node 개수 읽기
	model_input.node_count = (UInt32*)calloc(FCN_layer_count, sizeof(UInt32));
	for (int i = 0; i < FCN_layer_count; i++)
	{
		fread(&buffer, sizeof(UInt32), 1, fp);
		model_input.node_count[i] = buffer;
	}

	// 10. 손실함수 읽기
	fread(&buffer, sizeof(UInt32), 1, fp);
	model_input.metric = (Metric) buffer;

	// 11. 학습률 읽기
	fread(&lr, sizeof(Float64), 1, fp);
	model_input.learning_rate = lr;

	// 12. 가중치 읽기
	Model model = create_Model(model_input);
	Weight* weights_buffer = (Weight*)calloc(model.initializer.cnn_weights_count + model.initializer.fcn_weights_count, sizeof(Weight));
	fread(weights_buffer, sizeof(Weight), model.initializer.cnn_weights_count + model.initializer.fcn_weights_count, fp);

	if (model.initializer.Conv_layer_count + model.initializer.Pooling_layer_count > 0)
	{
		for (int i = 0; i < model.initializer.cnn_weights_count; i++)
			model.parameters.CNN.kernels[i] = weights_buffer[i];

		for (int i = 0; i < model.initializer.fcn_weights_count; i++)
			model.parameters.FCN.weights[i] = weights_buffer[model.initializer.cnn_weights_count + i];
	}
	else
	{
		for (int i = 0; i < model.initializer.fcn_weights_count; i++)
			model.parameters.FCN.weights[i] = weights_buffer[i];
	}

	fclose(fp);
	free(weights_buffer);

	return model;
}

int save_weights(Model* model, Weight* weights, Float64 score)
{
	Float64 maxScore = 0.0;
	
	if (score <= maxScore)
		return 0;

	if (model->initializer.Conv_layer_count + model->initializer.Pooling_layer_count > 0)
	{
		for (int i = 0; i < model->initializer.cnn_weights_count; i++)
			weights[i] = model->parameters.CNN.kernels[i];

		for (int i = 0; i < model->initializer.fcn_weights_count; i++)
			weights[model->initializer.cnn_weights_count + i] = model->parameters.FCN.weights[i];
	}
	else
	{
		for (int i = 0; i < model->initializer.fcn_weights_count; i++)
			weights[i] = model->parameters.FCN.weights[i];
	}

	return 0;
}

UInt8 predict(Float64* image, Model* model)
{
	//printf("예측 수행 중 \n");
	Output* fcn_input = (Output*)calloc(model->initializer.input_dimension, sizeof(Output));

	forward_propagation(image, model, fcn_input);
	Output* label = (Output*)calloc(LABEL, sizeof(Output));
	UInt32 ouput_layer_start = model->initializer.node_start_point[(model->initializer.FCN_layer_count) - 1];

	for (int i = 0; i < LABEL; i++)
		label[i] = model->parameters.FCN.nodes[ouput_layer_start + i];

	UInt8 result = 255;
	Output result_value = -DBL_MAX + 1;

	for (int i = 0; i < LABEL; i++)
	{
		//printf("label %d value is %f \n", i, label[i]);
		if (label[i] > result_value)
		{
			result_value = label[i];
			result = i;
		}
	}

	free(label);
	free(fcn_input);

	return result;
}

Float64 accuracy_score(Data* testdata, Model* model)
{
	printf("모델의 정확도 계산 중\n");
	Float64 score;

	UInt32 all_data_count = 0;
	UInt32 correct_data = 0;

	for (int label = 0; label < LABEL; label++)
		all_data_count += testdata->label_count[label];

	for (int idx = 0; idx < testdata->label_count[testdata->label_max]; idx++)
		for (int label = 0; label < LABEL; label++)
		{
			if (idx >= testdata->label_count[label])
				continue;

			Float64* label_data = testdata->data[label];
			Float64* input_image = (Float64*)calloc(WIDTH * HEIGHT, sizeof(Float64));

			for (int pixel = 0; pixel < WIDTH * HEIGHT; pixel++)
				input_image[pixel] = label_data[idx * WIDTH * HEIGHT + pixel];

			UInt8 output = predict(input_image, model);

			if (output == label)
				correct_data += 1;

			free(input_image);
		}

	score = ((double)correct_data) / ((double)all_data_count);
	//printf("정확도 : %d / %d = %f \n", correct_data, all_data_count, score);

	return score;
}

int model_destroy(Model* model)
{
	printf("model destroy \n");

	if (model->initializer.FCN_layer_count > 0)
	{
		free(model->parameters.FCN.weights);
		free(model->parameters.FCN.nodes);
		free(model->initializer.weight_start_point);
		free(model->initializer.node_start_point);
	}

	if (model->initializer.Conv_layer_count > 0)
	{
		free(model->parameters.CNN.kernels);
		free(model->parameters.CNN.feature_map);
		free(model->parameters.CNN.feature_map_width);
		free(model->parameters.CNN.feature_map_height);
		free(model->initializer.kernel_start_point);
		free(model->initializer.feature_map_start_point);
		free(model->initializer.layer_index);
	}


	if (model->initializer.Max_Pooling_layer_count > 0)
	{
		free(model->parameters.CNN.max_pooling_address_map);
		free(model->parameters.CNN.max_pooling_address_map_start_point);
	}

	return 0;
}

int model_input_destroy(Model* model)
{
	Model_Input* model_input = &(model->input);

	if (model->initializer.FCN_layer_count > 0)
		free(model_input->node_count);

	
	if (model->initializer.Conv_layer_count > 0)
	{
		free(model_input->channel_count);
		free(model_input->cnn_kernel_size);
		free(model_input->stride);
	}

	if (model->initializer.Pooling_layer_count > 0)
		free(model_input->pooling_window_size);
	
	free(model_input->layer);
	free(model_input->activation);

	return 0;
}

void DoProgress(const char label[], int step, int total)
{
	const int pwidth = 72;
	int width = pwidth - strlen(label);
	int pos = (step * width) / total;
	int percent = (step * 100) / total;
	printf("%s[", label);

	for (int i = 0; i < pos; i++)  printf("%c", '=');

	printf("% *c", width - pos + 1, ']');
	printf(" %3d%%\r", percent);
}