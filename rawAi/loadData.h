#pragma once
#pragma warning(disable:4996)

#include<iostream>
#include<opencv2/opencv.hpp>
#include"type.h"


int PNG2RAW(char* path); // raw ���Ϸ� ���� ��ġ�� ���� ��ǻ� ��� x
int read_PNG(Data* data, char* path); // trainData�� ������ ��� ����
int confirm_Data(Data* data, char* save_name, Int32 label, Int32 num); // label�� num��° ������ raw���Ϸ� ���, save_name path+filename
int data_destroy(Data* data); // �޸� ����