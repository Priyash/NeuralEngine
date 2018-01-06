#include<iostream>
#include"ImageManager.h"
#include<cudnn.h>
#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include"ConvNet.h"


int main()
{

	ConvNet* net = new ConvNet();
	net->process_Data_Batch();
	net->start_training();

	return 0;
}