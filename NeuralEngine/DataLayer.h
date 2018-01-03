#ifndef DATALAYER_H
#define DATALAYER_H

#include<random>
#include"Util.h"
#include<cuda.h>

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
	virtual void set_src_data(float* src_data) = 0;
	virtual void init_data(float* filter_data, float* bias_data) = 0;
	virtual void alloc_out_data_gpu(int batch, int out_feature_map, int w, int h) = 0;
	virtual float* getSrcDataHost() = 0;
	virtual float* getSrcDataDevice() = 0;
	virtual vector<float*> getHostData() = 0;
	virtual vector<float*> getDeviceData() = 0;

	virtual void copySrcsDataToDevice() = 0;
	virtual void copyFilterDataToDevice() = 0;
	virtual void copyBiasDataToDevice() = 0;

	virtual void copySrcDataToHost() = 0;
	virtual void copyFilterDataToHost() = 0;
	virtual void copyBiasDataToHost() = 0;
	virtual DataLayerResult getResult() = 0;
};


class DataLayer : public AbstractDataLayer
{
	//SRC DATA
	float* src_data;
	float* src_data_d;
	int src_data_size;

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
	void set_src_data(float* src_data);
	void init_data(float* filter_data, float* bias_data);
	void alloc_out_data_gpu(int batch, int out_feature_map, int w, int h);
	float* getSrcDataHost();
	float* getSrcDataDevice();
	vector<float*> getHostData();
	vector<float*> getDeviceData();
	void copySrcsDataToDevice();
	void copyFilterDataToDevice();
	void copyBiasDataToDevice();

	void copySrcDataToHost(){}
	void copyFilterDataToHost(){}
	void copyBiasDataToHost(){}
	DataLayerResult getResult();
private:
	void compute_src_data_size();
	void compute_filter_size();
	void compute_bias_size();
	double gen_random_number();
	void allocate_gpu_src_data_memory(int src_size);
	void allocate_gpu_filter_weight_memory(int w_size);
	void allocate_gpu_bias_weight_memory(int b_size);
};







#endif