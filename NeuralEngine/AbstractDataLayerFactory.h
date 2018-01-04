#ifndef ABSTRACTDATALAYERFACTORY
#define ABSTRACTDATALAYERFACTORY
#include"DataLayer.h"

enum DATA_LAYER_ID
{
	SRC,
	FILTER,
	BIAS,
	DST
};


class AbstractDataLayerFactory
{


public:
	AbstractDataLayerFactory(){}
	~AbstractDataLayerFactory(){}
	virtual void compute_size(DATA_LAYER_ID id) = 0;
	virtual void compute_dst_size(int batch, int out_feature_map, int width, int height) = 0;
	virtual void allocate_data_to_device(DATA_LAYER_ID id) = 0;
	virtual void allocate_data_to_host(DATA_LAYER_ID id) = 0;
	virtual void Init(DATA_LAYER_ID id) = 0;
	virtual void Init_dst_data(float* dst_data) = 0;
	virtual float* get_data_d(DATA_LAYER_ID id) = 0;
	virtual float* get_data_h(DATA_LAYER_ID id) = 0;
	virtual void copyDataToDevice(DATA_LAYER_ID id) = 0;
	virtual void copyDataToHost(DATA_LAYER_ID id) = 0;
};



class DataLayerFactory : public AbstractDataLayerFactory
{

	AbstractDataLayer* dataLayer;
public:
	DataLayerFactory(float* src_data);
	~DataLayerFactory();
	void compute_size(DATA_LAYER_ID id);
	void compute_dst_size(int batch, int out_feature_map, int width, int height);
	void allocate_data_to_device(DATA_LAYER_ID id);
	void allocate_data_to_host(DATA_LAYER_ID id);
	void Init(DATA_LAYER_ID id);
	void Init_dst_data(float* dst_data);
	float* get_data_d(DATA_LAYER_ID id);
	float* get_data_h(DATA_LAYER_ID id);
	void copyDataToDevice(DATA_LAYER_ID id);
	void copyDataToHost(DATA_LAYER_ID id);
};






#endif