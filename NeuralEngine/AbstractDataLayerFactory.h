#ifndef ABSTRACTDATALAYERFACTORY
#define ABSTRACTDATALAYERFACTORY
#include"DataLayer.h"

enum DATA_LAYER_ID
{
	SRC,
	FILTER,
	BIAS,
	WORKSPACE,
	DST
};


class AbstractDataLayerFactory
{


public:
	AbstractDataLayerFactory(){}
	~AbstractDataLayerFactory(){}
	virtual void compute_data_size(DATA_LAYER_ID id) = 0;
	virtual void compute_src_data_size(int src_data_size) = 0;
	virtual void compute_workspace_data_size(size_t workspace_byte) = 0;
	virtual void compute_dst_data_size(int batch, int out_feature_map, int width, int height) = 0;
	virtual void allocate_data_to_device(DATA_LAYER_ID id) = 0;
	virtual void allocate_data_to_host(DATA_LAYER_ID id) = 0;
	virtual void Init(DATA_LAYER_ID id) = 0;
	virtual void Init_dst_data(float* dst_data) = 0;
	virtual void Init_workspace_data(void* workspace_data) = 0;
	virtual float* get_data_d(DATA_LAYER_ID id) = 0;
	virtual void* get_workspace_data_d(DATA_LAYER_ID id) = 0;
	virtual void* get_workspace_data_h(DATA_LAYER_ID id) = 0;
	virtual float* get_data_h(DATA_LAYER_ID id) = 0;
	//DATA SIZE GETTER FUNCTION
	virtual int getDataSize(DATA_LAYER_ID id) = 0;
	virtual size_t getWorkspaceDataSize() = 0;

	virtual void copyDataToDevice(DATA_LAYER_ID id) = 0;
	virtual void copyDataToHost(DATA_LAYER_ID id) = 0;
};



class DataLayerFactory : public AbstractDataLayerFactory
{

	AbstractDataLayer* dataLayer;
public:
	DataLayerFactory(float* src_data);
	~DataLayerFactory();
	void compute_src_data_size(int src_data_size);
	void compute_data_size(DATA_LAYER_ID id);
	void compute_workspace_data_size(size_t workspace_byte);
	void compute_dst_data_size(int batch, int out_feature_map, int width, int height);
	void allocate_data_to_device(DATA_LAYER_ID id);
	void allocate_data_to_host(DATA_LAYER_ID id);
	void Init(DATA_LAYER_ID id);
	void Init_dst_data(float* dst_data);
	void Init_workspace_data(void* workspace_data);
	float* get_data_d(DATA_LAYER_ID id);
	void* get_workspace_data_d(DATA_LAYER_ID id);
	void* get_workspace_data_h(DATA_LAYER_ID id);
	float* get_data_h(DATA_LAYER_ID id);
	void copyDataToDevice(DATA_LAYER_ID id);
	void copyDataToHost(DATA_LAYER_ID id);
	//DATA SIZE GETTER FUNCTION
	int getDataSize(DATA_LAYER_ID id);
	size_t getWorkspaceDataSize();
	
	
};



#endif