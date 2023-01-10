#include<string>
#include<stdio.h>

#include"type.h"
#include"loadData.h"

int PNG2RAW(char* path)
{
	printf("convert png to raw");
	const char* labels[10] = {"0\\*.png", "1\\*.png", "2\\*.png", "3\\*.png", "4\\*.png", "5\\*.png", "6\\*.png", "7\\*.png", "8\\*.png", "9\\*.png"};


	for (int label = 0; label < 10; label++) 
	{
		printf("Start convert Data label %d \n", label);

		char directory_path[100];
		strcpy(directory_path, path);
		strcat(directory_path, labels[label]);

		std::vector<cv::String> filenames;
		cv::glob(directory_path, filenames, true);
		
		for (int idx = 0; idx < filenames.size(); idx++)
		{
			// 파일 읽기
			cv::String filename = filenames[idx];
			cv::Mat input = cv::imread(filename, cv::IMREAD_GRAYSCALE);

			// 저장할 raw 파일 이름 정하기
			Str raw_name[50] = "iamge_";
			Str num[10];
			strcat(raw_name, itoa(label, num, 10));
			strcat(raw_name, itoa(idx, num, 10));
			strcat(raw_name, ".raw");

			UInt8* data = (UInt8*)malloc(sizeof(UInt8) * WIDTH * HEIGHT);

			for (int i = 0; i < WIDTH * HEIGHT; i++)
				data[i] = input.data[i];

			FILE* fp;

			fopen_s(&fp, raw_name, "wb");
			fwrite(data, sizeof(UInt8), WIDTH * HEIGHT, fp);
			fclose(fp);

			free(data);
		}
	}

	return 0;
}

int read_PNG(Data* data, char* path)
{
	const char* labels[LABEL] = { "0\\*.png", "1\\*.png", "2\\*.png", "3\\*.png", "4\\*.png", "5\\*.png", "6\\*.png", "7\\*.png", "8\\*.png", "9\\*.png" };
	//TrainData trainData;
	for (int label = 0; label < 10; label++)
	{
		printf("Start read Data label %d \n", label);

		char directory_path[100];
		strcpy(directory_path, path);
		strcat(directory_path, labels[label]);

		std::vector<cv::String> filenames;
		cv::glob(directory_path, filenames, true);

		data->data[label] = (Float64*)malloc(sizeof(Float64) * WIDTH * HEIGHT * filenames.size());
		data->label_count[label] = filenames.size();

		Float64* label_data = data->data[label];
		for (int idx = 0; idx < filenames.size(); idx++)
		{
			// 파일 읽기
			cv::String filename = filenames[idx];
			cv::Mat input = cv::imread(filename, cv::IMREAD_GRAYSCALE);

			for (int i = 0; i < WIDTH * HEIGHT; i++)
				label_data[idx * WIDTH * WIDTH + i] = input.data[i] / 255.0;
		}
	}

	Int32 max = 0;
	Int32 label_max_index = -1;
	for (int i = 0; i < LABEL; i++)
	{
		if (data->label_count[i] > max)
		{
			max = data->label_count[i];
			label_max_index = i;
		}
	}
	data->label_max = label_max_index;
	return 0;
}

int confirm_Data(Data* data, char* save_name, Int32 label, Int32 num)
{
	printf("Confirm Data %d data of label %d \n", num, label);
	// 저장할 raw 파일 이름 정하기

	UInt8* file_data = (UInt8*)malloc(sizeof(UInt8) * WIDTH * HEIGHT);
	Float64* confirm_data = data->data[label];

	for (int i = 0; i < WIDTH * HEIGHT; i++)
		file_data[i] = (UInt8)round(confirm_data[num * WIDTH * HEIGHT +i] * 255.0);

	FILE* fp;

	fopen_s(&fp, save_name, "wb");
	fwrite(file_data, sizeof(UInt8), WIDTH * HEIGHT, fp);
	fclose(fp);

	free(file_data);
	return 0;
}

int data_destroy(Data* data)
{
	printf("destroy memory \n");

	for (int i = 0; i < LABEL; i++)
		free(data->data[i]);

	return 0;
}