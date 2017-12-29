#ifndef LAYER_H
#define LAYER_H
#include<cudnn.h>
#include<iostream>
#include"CudaException.h"
#include"ImageManager.h"

using namespace std;

struct TensorShape
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

class AbstractTensorLayer
{
protected:
	cudnnStatus_t status;
public:
	AbstractTensorLayer(){}
	~AbstractTensorLayer(){}
	virtual void createTensorDescriptor() = 0;
	virtual void setTensorDescriptor() = 0;
	virtual cudnnTensorDescriptor_t getTensorDescriptor() = 0;
	virtual cudnnFilterDescriptor_t getFilterDescriptor() = 0;
	virtual cudnnConvolutionDescriptor_t getConvDescriptor() = 0;
	virtual TensorShape getConvolutedImagedOutDim(cudnnConvolutionDescriptor_t conDesc, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filterDesc) = 0;

	void check_cuda_status(cudnnStatus_t status, string error_module);


	virtual cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes,
		cudnnConvolutionFwdAlgo_t conv_fwd_algo) = 0;
	virtual size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdAlgo_t conv_algo) = 0;
};



class TensorLayer : public AbstractTensorLayer
{
	TensorShape shape;
	cudnnTensorDescriptor_t descriptor;
public:
	TensorLayer(const TensorShape& shape);
	~TensorLayer();
	void createTensorDescriptor();
	void setTensorDescriptor();
	cudnnTensorDescriptor_t getTensorDescriptor();
	cudnnFilterDescriptor_t getFilterDescriptor(){ cudnnFilterDescriptor_t  filter_desc;  return filter_desc; }
	cudnnConvolutionDescriptor_t getConvDescriptor(){ cudnnConvolutionDescriptor_t conv_desc ; return conv_desc; }

	TensorShape getConvolutedImagedOutDim(cudnnConvolutionDescriptor_t conDesc, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filterDesc){ TensorShape sh; return sh; }

	cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes,
		cudnnConvolutionFwdAlgo_t conv_fwd_algo){
		cudnnConvolutionFwdAlgo_t convolution_algorithm; return convolution_algorithm;
	}
	size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdAlgo_t conv_algo){
		size_t s; return s;
	}
};


class FilterLayer : public AbstractTensorLayer
{
	cudnnFilterDescriptor_t  filter_descriptor;
	FilterShape shape;
public:
	FilterLayer(const FilterShape& shape);
	~FilterLayer();
	void createTensorDescriptor();
	void setTensorDescriptor();
	cudnnTensorDescriptor_t getTensorDescriptor(){ cudnnTensorDescriptor_t tensor_desc; return tensor_desc; }
	cudnnFilterDescriptor_t getFilterDescriptor();
	cudnnConvolutionDescriptor_t getConvDescriptor(){ cudnnConvolutionDescriptor_t conv_desc ; return conv_desc; }


	TensorShape getConvolutedImagedOutDim(cudnnConvolutionDescriptor_t conDesc, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filterDesc){ TensorShape sh; return sh; }
	
	cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes,
		cudnnConvolutionFwdAlgo_t conv_fwd_algo){
		cudnnConvolutionFwdAlgo_t convolution_algorithm; return convolution_algorithm;
	}
	size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdAlgo_t conv_algo){
		size_t s; return s;
	}
};



class Bias : public AbstractTensorLayer
{
	cudnnTensorDescriptor_t  bias_descriptor;
	BiasShape bias_shape;
public:
	Bias(const BiasShape& bias_shape);
	~Bias();
	void createTensorDescriptor();
	void setTensorDescriptor();
	cudnnTensorDescriptor_t getTensorDescriptor();
	cudnnFilterDescriptor_t getFilterDescriptor(){ cudnnFilterDescriptor_t  filter_des; return filter_des; }
	cudnnConvolutionDescriptor_t getConvDescriptor(){ cudnnConvolutionDescriptor_t conv_desc; return conv_desc; }

	TensorShape getConvolutedImagedOutDim(cudnnConvolutionDescriptor_t conDesc, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filterDesc){ TensorShape sh; return sh; }

	cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes,
		cudnnConvolutionFwdAlgo_t conv_fwd_algo){
		cudnnConvolutionFwdAlgo_t convolution_algorithm; return convolution_algorithm;
	}
	size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdAlgo_t conv_algo){
		size_t s; return s;
	}

};




class ConvTensorLayer : public AbstractTensorLayer
{
	cudnnConvolutionDescriptor_t convolution_descriptor;
	ConvShape shape;
public:
	ConvTensorLayer(const ConvShape& shape);
	~ConvTensorLayer();
	void createTensorDescriptor();
	void setTensorDescriptor();
	cudnnTensorDescriptor_t getTensorDescriptor(){ cudnnTensorDescriptor_t tensor_desc; return tensor_desc; }
	cudnnFilterDescriptor_t getFilterDescriptor(){ cudnnFilterDescriptor_t  filter_desc; return filter_desc; }
	cudnnConvolutionDescriptor_t getConvDescriptor();

	//CONVOLUTION RELATED METHODS

	TensorShape getConvolutedImagedOutDim(cudnnConvolutionDescriptor_t conDesc, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filterDesc);

	cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,cudnnConvolutionFwdPreference_t convFwdPref,
		size_t mem_limit_bytes,cudnnConvolutionFwdAlgo_t conv_fwd_algo);

	size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc, cudnnConvolutionDescriptor_t convDesc,
		cudnnTensorDescriptor_t outDesc, cudnnConvolutionFwdAlgo_t conv_algo);

};

#endif