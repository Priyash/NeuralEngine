#ifndef CONVNET_H
#define CONVNET_H
#include"AbstractLayerFactory.h"


class ConvNet
{
	AbstractLayerFactory* inputLayerfactory;

public:
	ConvNet();
	void start(float* img_data);
	~ConvNet();
};

#endif