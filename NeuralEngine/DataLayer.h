#ifndef DATALAYER_H
#define DATALAYER_H

#include<random>
#include"Util.h"
#include<cuda.h>
#include "CudaException.h"


struct DataLayerResult
{
	float* output_d;
	float* output_h;
};



class AbstractDataLayer
{
protected:
	cudaError_t status;
public:
	AbstractDataLayer(){}
	~AbstractDataLayer(){}
	virtual void init_data() = 0;
	virtual void init_data(float* filter_data, float* bias_data) = 0;
	virtual void alloc_out_data_gpu(int batch, int out_feature_map, int w, int h) = 0;
	virtual vector<float*> getHostData() = 0;
	virtual vector<float*> getDeviceData() = 0;
	virtual void copyDataToDevice() = 0;
	virtual void copyDataToHost() = 0;
	void check_cuda_status(cudaError_t status, string error_module);
	virtual DataLayerResult getResult() = 0;
};


class DataLayer : public AbstractDataLayer
{
	//FILTER DATA
	float* filter_weight_h;
	float* filter_weight_d;
	int w_size;
	int filter_weight_min_value;
	int filter_weight_max_value;

	//BIAS DATA
	float* bias_weight_h;
	float* bias_weight_d;
	int b_size;
	int b_weight_min_value;
	int b_weight_max_value;

	//OUPUT DATA
	DataLayerResult result;

public:
	DataLayer();
	~DataLayer();
	void init_data();
	void init_data(float* filter_data, float* bias_data);
	void alloc_out_data_gpu(int batch, int out_feature_map, int w, int h);
	vector<float*> getHostData();
	vector<float*> getDeviceData();
	void copyDataToDevice();
	void copyDataToHost(){}
	DataLayerResult getResult();
private:
	void compute_filter_size();
	void compute_bias_size();
	double gen_random_number();
	void allocate_gpu_filter_weight_memory(int w_size);
	void allocate_gpu_bias_weight_memory(int b_size);
	void allocate_gpu_output_data_memory(int o_size);
};







#endif