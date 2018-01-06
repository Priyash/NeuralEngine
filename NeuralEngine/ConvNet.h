#ifndef CONVNET_H
#define CONVNET_H
#include"AbstractLayerFactory.h"


class ConvNet
{
	AbstractLayerFactory* inputLayerfactory;
public:
	ConvNet();
	~ConvNet();
	void process_Data_Batch();
	void process_image_data(float* img_data);
	void start_training();

};

#endif