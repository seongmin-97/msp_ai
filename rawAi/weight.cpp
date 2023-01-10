#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include"weight.h"

#define MEAN 0.0
#define STD 0.01

// ����ġ ���� �Ҵ�
int random_weights(Weight* weights, UInt32 count_weight, Float64 range)
{
	//printf("����ġ ���� �ʱ�ȭ \n");
	srand(0);
	for (int i = 0; i < count_weight; i++)
		weights[i] = ((((Weight)rand() / (Weight)(RAND_MAX)) * 2) - 1) * range; // -0.2 to 0.2 ����

	weights_distribution(weights, count_weight);

	return 0;
}

int random_gaussian_weights(Weight* weights, UInt32 count_weight)
{
	srand(0);
	for (int i = 0; i < count_weight; i++)
		weights[i] = gaussianRandom(MEAN, STD); // -0.2 to 0.2 ����

	weights_distribution(weights, count_weight);

	return 0;
}

Float64 gaussianRandom(Float64 average, Float64 std)
{
	double v1, v2, s, result;

	do
	{
		v1 = 2 * ((Float64)rand() / RAND_MAX) - 1;
		v2 = 2 * ((Float64)rand() / RAND_MAX) - 1;

		s = v1 * v1 + v2 * v2;
	} while (s >= 1 || s == 0);

	s = sqrt((-2 * log(s)) / s);

	result = v1 * s;
	result = (std * result) + average;

	return result;
}

int weights_distribution(Weight* weights, UInt32 count_weight)
{
	Float64 mean = 0;
	Float64 std = 0;

	for (int i = 0; i < count_weight; i++)
		mean += weights[i];

	mean = mean / count_weight;

	for (int i = 0; i < count_weight; i++)
		std += (mean - weights[i]) * (mean - weights[i]);

	std = sqrt(std / count_weight);

	printf("����ġ ��� : %f, ����ġ ǥ������ : %f \n\n", mean, std);
	return 0;
}