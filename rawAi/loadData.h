#pragma once
#pragma warning(disable:4996)

#include<iostream>
#include<opencv2/opencv.hpp>
#include"type.h"


int PNG2RAW(char* path); // raw 파일로 현재 위치에 저장 사실상 사용 x
int read_PNG(Data* data, char* path); // trainData에 데이터 모두 저장
int confirm_Data(Data* data, char* save_name, Int32 label, Int32 num); // label의 num번째 데이터 raw파일로 출력, save_name path+filename
int data_destroy(Data* data); // 메모리 해제