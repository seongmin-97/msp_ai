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
	Float64* data[LABEL];          // data �迭�� �� LABEL �ε����� �����Ͱ� �ԷµǾ� ����
	Int32 label_count[LABEL];      // �� label �� ���� �������� ����
	Int32 label_max;               // �����Ͱ� ���� ���� label
}Data;

