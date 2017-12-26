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
	int feature_map;
	int rows;
	int cols;
};

struct FilterShape
{
	int n_input_feature_map;
	int n_output_feature_map;
	int filter_width;
	int filter_height;
};

struct BiasShape
{
	int batch_size;
	int feature_map;
	int rows;
	int cols;
};


struct ConvShape
{
	int pad_height;
	int pad_width;
	int vertical_stride;
	int horizontal_stride;
	int dilation_height;
	int dilation_width;
};

class AbstractLayer
{

public:
	AbstractLayer(){}
	~AbstractLayer(){}
	virtual void createTensorDescriptor() = 0;
	virtual void setTensorDescriptor() = 0;
	virtual cudnnTensorDescriptor_t getTensorDescriptor() = 0;
	virtual cudnnFilterDescriptor_t getFilterDescriptor() = 0;
	virtual cudnnConvolutionDescriptor_t getConvDescriptor() = 0;

	void check_cuda_status(cudnnStatus_t status, string error_module);
};



class InputLayer : public AbstractLayer
{
	InputShape shape;
	cudnnTensorDescriptor_t input_descriptor;
	cudnnFilterDescriptor_t  filter_desc;
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnStatus_t status;
public:
	InputLayer(const InputShape& shape);
	~InputLayer();
	void createTensorDescriptor();
	void setTensorDescriptor();
	cudnnTensorDescriptor_t getTensorDescriptor();
	cudnnFilterDescriptor_t getFilterDescriptor(){ return filter_desc; }
	cudnnConvolutionDescriptor_t getConvDescriptor(){ return conv_desc; }
};




class FilterLayer : public AbstractLayer
{
	cudnnFilterDescriptor_t  filter_descriptor;
	cudnnTensorDescriptor_t tensor_desc;
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnStatus_t status;
	FilterShape shape;
public:
	FilterLayer(const FilterShape& shape);
	~FilterLayer();
	void createTensorDescriptor();
	void setTensorDescriptor();
	cudnnTensorDescriptor_t getTensorDescriptor(){ return tensor_desc; }
	cudnnFilterDescriptor_t getFilterDescriptor();
	cudnnConvolutionDescriptor_t getConvDescriptor(){ return conv_desc; }
	
};



class Bias : public AbstractLayer
{

	cudnnTensorDescriptor_t  bias_descriptor;
	cudnnFilterDescriptor_t  filter_des;
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnStatus_t status;
	BiasShape bias_shape;
public:
	Bias(const BiasShape& bias_shape);
	~Bias();
	void createTensorDescriptor();
	void setTensorDescriptor();
	cudnnTensorDescriptor_t getTensorDescriptor();
	cudnnFilterDescriptor_t getFilterDescriptor(){ return filter_des; }
	cudnnConvolutionDescriptor_t getConvDescriptor(){ return conv_desc; }
};


class ConvLayer : public AbstractLayer
{
	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnFilterDescriptor_t  filter_desc;
	cudnnTensorDescriptor_t tensor_desc;
	cudnnStatus_t status;
	ConvShape shape;
public:
	ConvLayer(const ConvShape& shape);
	~ConvLayer();
	void createTensorDescriptor();
	void setTensorDescriptor();
	cudnnTensorDescriptor_t getTensorDescriptor(){ return tensor_desc; }
	cudnnFilterDescriptor_t getFilterDescriptor(){ return filter_desc; }
	cudnnConvolutionDescriptor_t getConvDescriptor();
};



#endif