#ifndef CONVNET_H
#define CONVNET_H
#include"AbstractLayerFactory.h"

class ConvNet
{
	AbstractLayerFactory* inputfactory;

public:
	ConvNet();
	~ConvNet();
};

#endif