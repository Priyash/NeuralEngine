#ifndef DATALAYER_H
#define DATALAYER_H

#include<random>
#include"Util.h"
#include<cuda.h>


class AbstractDataLayer
{
protected:
	cudaError_t status;
public:
	AbstractDataLayer(){}
	~AbstractDataLayer(){}

	//COMPUTE DATA SIZES FOR HOST AND DEVICES,READ ATTRIBUTES FROM JSON CONFIG FILE
	virtual void compute_src_data_size(int src_data_size) = 0;
	virtual void compute_filter_data_size() = 0;
	virtual void compute_bias_data_size() = 0;
	virtual void compute_dst_data_size(int batch,int out_feature_map,int width,int height) = 0;
	virtual void compute_workspace_size(size_t workspace_byte) = 0;



	//DATA ALLOCATION TO GPU[CUDAMALLOC]
	virtual void alloc_src_data_to_device() = 0;
	virtual void alloc_filter_data_to_device() = 0;
	virtual void alloc_bias_data_to_device() = 0;
	virtual void alloc_dst_data_to_device() = 0;
	virtual void alloc_workspace_data_to_device() = 0;

	//DATA ALLOCATION TO CPU[NEW]
	virtual void alloc_src_data_to_host() = 0;
	virtual void alloc_filter_data_to_host() = 0;
	virtual void alloc_bias_data_to_host() = 0;
	virtual void alloc_dst_data_to_host() = 0;
	virtual void alloc_workspace_data_to_host() = 0;


	//INITIALIZE DATAs
	virtual void init_filter_data() = 0;
	virtual void init_bias_data() = 0;
	virtual void init_dst_data(float* dst_data) = 0;
	virtual void init_workspace_data(void* workspace_data) = 0;

	//DEVICE DATA POINTERS
	virtual float* get_src_data_d() = 0;
	virtual float* get_filter_data_d() = 0;
	virtual float* get_bias_data_d() = 0;
	virtual float* get_dst_data_d() = 0;
	virtual void* get_workspace_data_d() = 0;

	//HOST DATA POINTERS
	virtual float* get_src_data_h() = 0;
	virtual float* get_filter_data_h() = 0;
	virtual float* get_bias_data_h() = 0;
	virtual float* get_dst_data_h() = 0;
	virtual void* get_workspace_data_h() = 0;

	//COPY DATA FROM HOST TO DEVICE
	virtual void copySrcDataToDevice() = 0;
	virtual void copyFilterDataToDevice() = 0;
	virtual void copyBiasDataToDevice() = 0;
	virtual void copyDstDataToDevice() = 0;
	virtual void copyWorkspaceDataToDevice() = 0;


	//COPY DATA BACK FROM DEVICE TO HOST
	virtual void copySrcDataToHost() = 0;
	virtual void copyFilterDataToHost() = 0;
	virtual void copyBiasDataToHost() = 0;
	virtual void copyDstDataToHost() = 0;
	virtual void copyWorkspaceDataToHost() = 0;

	//DATA SIZE GETTER FUNCTION
	virtual int getSrcDataSize() = 0;
	virtual int getFilterDataSize() = 0;
	virtual int getBiasDataSize() = 0;
	virtual size_t getWorkspaceDataSize() = 0;
	virtual int getDstDataSize() = 0;
};


class DataLayer : public AbstractDataLayer
{

	// SRC DATA VARIABLES
	float* src_data_h;
	float* src_data_d;

	//FILTER DATA VARIABLES
	float* filter_data_h;
	float* filter_data_d;
	int filter_weight_min_value;
	int filter_weight_max_value;
	int filter_in_feature_map;
	int filter_out_feature_map;
	int filter_width;
	int filter_height;

	//BIAS DATA VARIABLES
	float* bias_data_h;
	float* bias_data_d;
	int bias_out_feature_map;

	//DST DATA VARIABLES
	float* dst_data_h;
	float* dst_data_d;
	int dst_batch;
	int dst_out_feature_map;
	int dst_width;
	int dst_height;

	//DATA SIZES VARIABLE
	int src_data_size;
	int filter_data_size;
	int bias_data_size;
	int dst_data_size;

	//WORKSPACE VARIABLE
	void* workspace_data_d;
	void* workspace_data_h;
	size_t workspace_byte;


	

public:
	DataLayer(float* src_data);
	~DataLayer();

	//COMPUTE DATA SIZES FOR HOST AND DEVICES,READ ATTRIBUTES FROM JSON CONFIG FILE
	void compute_src_data_size(int src_data_size);
	void compute_filter_data_size();
	void compute_bias_data_size();
	void compute_dst_data_size(int batch, int out_feature_map, int width, int height);
	void compute_workspace_size(size_t workspace_byte);



	//DATA ALLOCATION TO GPU[CUDAMALLOC]
	void alloc_src_data_to_device();
	void alloc_filter_data_to_device();
	void alloc_bias_data_to_device();
	void alloc_dst_data_to_device();
	void alloc_workspace_data_to_device();

	//DATA ALLOCATION TO CPU[NEW]
	void alloc_src_data_to_host();
	void alloc_filter_data_to_host();
	void alloc_bias_data_to_host();
	void alloc_dst_data_to_host();
	void alloc_workspace_data_to_host();
	

	//INITIALIZE DATAs
	void init_filter_data();
	void init_bias_data();
	void init_dst_data(float* dst_data);
	void init_workspace_data(void*);

	//DEVICE DATA POINTERS
	float* get_src_data_d();
	float* get_filter_data_d();
	float* get_bias_data_d();
	float* get_dst_data_d();
	void* get_workspace_data_d();

	//HOST DATA POINTERS
	float* get_src_data_h();
	float* get_filter_data_h();
	float* get_bias_data_h();
	float* get_dst_data_h();
	void* get_workspace_data_h();
	

	//COPY DATA FROM HOST TO DEVICE
	void copySrcDataToDevice();
	void copyFilterDataToDevice();
	void copyBiasDataToDevice();
	void copyDstDataToDevice();
	void copyWorkspaceDataToDevice();
	

	//COPY DATA BACK FROM DEVICE TO HOST
	void copySrcDataToHost();
	void copyFilterDataToHost();
	void copyBiasDataToHost();
	void copyDstDataToHost();
	void copyWorkspaceDataToHost();

	//DATA SIZE GETTER FUNCTION
	int getSrcDataSize();
	int getFilterDataSize();
	int getBiasDataSize();
	size_t getWorkspaceDataSize();
	int getDstDataSize();
	

private:
	double gen_random_number();
};







#endif