#pragma once


typedef unsigned char UInt8;
typedef unsigned int  UInt32;
typedef char          Int8;
typedef int           Int32;
typedef double        Float64;

typedef double        Weight;
typedef double        Output;
typedef double        LR;
typedef double        Loss;

typedef char          Str;

const UInt8 WIDTH   = 28;
const UInt8 HEIGHT  = 28;
const UInt8 CHANNEL = 1;
const UInt8 LABEL   = 10;

typedef enum Boolean
{
	False, True
} Bool;

typedef struct data_
{
	Float64* data[LABEL];          // data 배열의 각 LABEL 인덱스에 데이터가 입력되어 있음
	Int32 label_count[LABEL];      // 각 label 별 마다 데이터의 개수
	Int32 label_max;               // 데이터가 가장 많은 label
}Data;

