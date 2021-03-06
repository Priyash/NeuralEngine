#ifndef LAYER_H
#define LAYER_H
#include<cudnn.h>
#include<iostream>
#include"Util.h"
#include<cublas_v2.h>

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
	int conv_fwd_pref;
};

class AbstractTensorLayer
{
protected:
	cudnnStatus_t status;
	cublasStatus_t cublas_status;
	cudnnHandle_t cudnnHandler;
	cublasHandle_t cublasHandler;
public:
	AbstractTensorLayer(){}
	~AbstractTensorLayer(){}
	virtual void createTensorDescriptor() = 0;
	virtual void setTensorDescriptor() = 0;
	virtual cudnnTensorDescriptor_t getTensorDescriptor() = 0;
	virtual cudnnFilterDescriptor_t getFilterDescriptor() = 0;
	virtual cudnnConvolutionDescriptor_t getConvDescriptor() = 0;
	virtual TensorShape getConvolutedImagedOutDim(cudnnConvolutionDescriptor_t conDesc, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filterDesc) = 0;

	virtual void createCudnnHandler() = 0;
	virtual void createCublasHandler() = 0;
	virtual cudnnHandle_t getCudnnHandler() = 0;
	virtual cublasHandle_t getCublasHandler() = 0;

	virtual cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes
		) = 0;
	virtual size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdAlgo_t conv_algo) = 0;

	//CONVOLUTION PROCESS
	virtual void conv_forward(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t inDesc, float* d_input, cudnnFilterDescriptor_t filDesc,
		float* d_filter, cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t conv_algo, void* workspace,
		size_t workspace_byte, const float beta, cudnnTensorDescriptor_t outDesc, float* d_output) = 0;
	//BIAS DATA AFTER CONVOLUTION
	virtual void addBias(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t biasDesc, float* bias_d,
		const float beta, cudnnTensorDescriptor_t convDstDesc, float* conDstDesc_d) = 0;

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


	void createCudnnHandler(){}
	void createCublasHandler(){}
	cudnnHandle_t getCudnnHandler(){ return cudnnHandler; }
	cublasHandle_t getCublasHandler(){ return cublasHandler; }

	cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes){cudnnConvolutionFwdAlgo_t convolution_algorithm; return convolution_algorithm;}

	size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdAlgo_t conv_algo){
		size_t s; return s;
	}

	//CONVOLUTION PROCESS
	void conv_forward(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t inDesc, float* d_input, cudnnFilterDescriptor_t filDesc,
		float* d_filter, cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t conv_algo, void* workspace,
		size_t workspace_byte, const float beta, cudnnTensorDescriptor_t outDesc, float* d_output){}
	//BIAS DATA AFTER CONVOLUTION
	void addBias(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t biasDesc, float* bias_d,
		const float beta, cudnnTensorDescriptor_t convDstDesc, float* conDstDesc_d){}

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
	

	void createCudnnHandler(){}
	void createCublasHandler(){}

	cudnnHandle_t getCudnnHandler(){ return cudnnHandler; }
	cublasHandle_t getCublasHandler(){ return cublasHandler; }
	
	cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes
		){
		cudnnConvolutionFwdAlgo_t convolution_algorithm; return convolution_algorithm;
	}
	size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdAlgo_t conv_algo){
		size_t s; return s;
	}

	//CONVOLUTION PROCESS
	void conv_forward(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t inDesc, float* d_input, cudnnFilterDescriptor_t filDesc,
		float* d_filter, cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t conv_algo, void* workspace,
		size_t workspace_byte, const float beta, cudnnTensorDescriptor_t outDesc, float* d_output){}
	//BIAS DATA AFTER CONVOLUTION
	void addBias(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t biasDesc,
		float* bias_d, const float beta, cudnnTensorDescriptor_t convDstDesc, float* conDstDesc_d){}

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

	void createCudnnHandler(){}
	void createCublasHandler(){}

	cudnnHandle_t getCudnnHandler(){ return cudnnHandler; }
	cublasHandle_t getCublasHandler(){ return cublasHandler; }
	
	cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes
		){
		cudnnConvolutionFwdAlgo_t convolution_algorithm; return convolution_algorithm;
	}
	size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdAlgo_t conv_algo){
		size_t s; return s;
	}
	//CONVOLUTION PROCESS
	void conv_forward(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t inDesc, float* d_input, cudnnFilterDescriptor_t filDesc,
		float* d_filter, cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t conv_algo, void* workspace,
		size_t workspace_byte, const float beta, cudnnTensorDescriptor_t outDesc, float* d_output){}
	//BIAS DATA AFTER CONVOLUTION
	void addBias(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t biasDesc, float* bias_d,
		const float beta, cudnnTensorDescriptor_t convDstDesc, float* conDstDesc_d){}


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
	void createCudnnHandler(){}
	void createCublasHandler(){}
	
	cudnnHandle_t getCudnnHandler(){ return cudnnHandler; }
	cublasHandle_t getCublasHandler(){ return cublasHandler; }

	TensorShape getConvolutedImagedOutDim(cudnnConvolutionDescriptor_t conDesc, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filterDesc);

	//FUNTION FOR ALGORITHM AVAILABLE AT THE TIME CONVOLUTION BASED ON HARDWARE AND MEMORY LIMIT
	cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,cudnnConvolutionFwdPreference_t convFwdPref,
		size_t mem_limit_bytes);

	//REQUIRED SIZES FOR CONVOLUTION TO RUN ON DATASET
	size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc, cudnnConvolutionDescriptor_t convDesc,
		cudnnTensorDescriptor_t outDesc, cudnnConvolutionFwdAlgo_t conv_algo);

	//CONVOLUTION PROCESS
	void conv_forward(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t inDesc, float* d_input, cudnnFilterDescriptor_t filDesc,
		float* d_filter, cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t conv_algo,void* workspace,
		size_t workspace_byte, const float beta, cudnnTensorDescriptor_t outDesc,float* d_output);
	//BIAS DATA AFTER CONVOLUTION
	void addBias(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t biasDesc, float* bias_d, const float beta, cudnnTensorDescriptor_t convDstDesc,float* conDstDesc_d);
	

};



class CudnnHandler : public AbstractTensorLayer
{
	
public:
	CudnnHandler();
	~CudnnHandler();
	void createTensorDescriptor(){}
	void setTensorDescriptor(){}
	cudnnTensorDescriptor_t getTensorDescriptor(){ cudnnTensorDescriptor_t tensor_desc; return tensor_desc; }
	cudnnFilterDescriptor_t getFilterDescriptor(){ cudnnFilterDescriptor_t  filter_des; return filter_des; }
	cudnnConvolutionDescriptor_t getConvDescriptor(){ cudnnConvolutionDescriptor_t conv_desc; return conv_desc; }
	TensorShape getConvolutedImagedOutDim(cudnnConvolutionDescriptor_t conDesc, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filterDesc){ TensorShape sh; return sh; }


	void createCudnnHandler();
	void createCublasHandler();

	cudnnHandle_t getCudnnHandler();
	cublasHandle_t getCublasHandler();

	cudnnConvolutionFwdAlgo_t getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes
		){
		cudnnConvolutionFwdAlgo_t convolution_algorithm; return convolution_algorithm;
	}
	size_t getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc,
		cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc,
		cudnnConvolutionFwdAlgo_t conv_algo){
		size_t s; return s;
	}


	//CONVOLUTION PROCESS
	void conv_forward(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t inDesc, float* d_input, cudnnFilterDescriptor_t filDesc,
		float* d_filter, cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t conv_algo, void* workspace,
		size_t workspace_byte, const float beta, cudnnTensorDescriptor_t outDesc, float* d_output){}
	//BIAS DATA AFTER CONVOLUTION
	void addBias(cudnnHandle_t cudnn, const float alpha, cudnnTensorDescriptor_t biasDesc, float* bias_d, const float beta, 
		cudnnTensorDescriptor_t convDstDesc, float* conDstDesc_d){}

	
};

#endif