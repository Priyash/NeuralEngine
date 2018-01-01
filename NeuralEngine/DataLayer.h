#ifndef DATALAYER_H
#define DATALAYER_H

#include<random>
#include"Util.h"
#include<cuda.h>
#include "CudaException.h"

class AbstractDataLayer
{
protected:
	cudaError_t status;
public:
	AbstractDataLayer();
	~AbstractDataLayer();
	virtual void init_data() = 0;
	virtual float* getHostData() = 0;
	virtual float* getDeviceData() = 0;
	virtual float* copyDataToDevice() = 0;
	void check_cuda_status(cudaError_t status, string error_module);
};



class FilterDataLayer : public AbstractDataLayer
{

	float* filter_weight_h;
	float* filter_weight_d;
	int w_size;
	int filter_weight_min_value;
	int filter_weight_max_value;
public:
	FilterDataLayer();
	~FilterDataLayer();
	void init_data();
	float* getHostData();
	float* getDeviceData();
	float* copyDataToDevice();


private:
	void compute_filter_size();
	double gen_random_number();
	void allocate_gpu_memory(int w_size);
};






#endif