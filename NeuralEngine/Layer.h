#ifndef LAYER_H
#define LAYER_H
#include<cudnn.h>
#include<iostream>
#include"CudaException.h"
#include"ImageManager.h"

using namespace std;

struct InputShape
{
	int batch_size;
	int channels;
	int rows;
	int cols;
};

class AbstractLayer
{

public:
	AbstractLayer(){}
	~AbstractLayer(){}
	virtual void allocateInputData(Mat data) = 0;
	virtual void createTensorDescriptor() = 0;
	virtual void setTensorDescriptor() = 0;
	void check_cuda_status(cudnnStatus_t status, string error_module);
};



class InputLayer : public AbstractLayer
{
	InputShape shape;
	cudnnTensorDescriptor_t input_descriptor;
	cudnnStatus_t status;
	ImageManager* imageManager;
public:
	InputLayer(const InputShape& shape);
	~InputLayer();
	void allocateInputData(Mat data);
	void createTensorDescriptor();
	void setTensorDescriptor();
};











#endif