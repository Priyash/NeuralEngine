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
	
	void start_training();

private:
	void process_image_data(float* img_data, int img_data_size);

};

#endif