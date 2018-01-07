#include "AbstractDataLayerFactory.h"


DataLayerFactory::DataLayerFactory(float* src_data, int src_data_len)
{
	dataLayer = new DataLayer(src_data, src_data_len);
}


DataLayerFactory::~DataLayerFactory()
{

}


void DataLayerFactory::compute_size(DATA_LAYER_ID id)
{
	switch (id)
	{
	case DATA_LAYER_ID::SRC:
		dataLayer->compute_src_data_size();
		break;
	case DATA_LAYER_ID::FILTER:
		dataLayer->compute_filter_data_size();
		break;
	case DATA_LAYER_ID::BIAS:
		dataLayer->compute_bias_data_size();
		break;

	default:
		return;
	}
}

void DataLayerFactory::compute_dst_size(int batch, int out_feature_map, int width, int height)
{
	dataLayer->compute_dst_data_size(batch, out_feature_map, width, height);
}


void DataLayerFactory::compute_workspace_data_size(size_t workspace_byte)
{
	dataLayer->compute_workspace_size(workspace_byte);
}

void DataLayerFactory::allocate_data_to_device(DATA_LAYER_ID id)
{
	switch (id)
	{
	case DATA_LAYER_ID::SRC:
		dataLayer->alloc_src_data_to_device();
		break;
	case DATA_LAYER_ID::FILTER:
		dataLayer->alloc_filter_data_to_device();
		break;
	case DATA_LAYER_ID::BIAS:
		dataLayer->alloc_bias_data_to_device();
		break;
	case DATA_LAYER_ID::DST:
		dataLayer->alloc_dst_data_to_device();
		break;
	case DATA_LAYER_ID::WORKSPACE:
		dataLayer->alloc_workspace_data_to_device();
	default:
		return;
	}
}

void DataLayerFactory::allocate_data_to_host(DATA_LAYER_ID id)
{
	switch (id)
	{
		case DATA_LAYER_ID::SRC:
			dataLayer->alloc_src_data_to_host();
			break;
		case DATA_LAYER_ID::FILTER:
			dataLayer->alloc_filter_data_to_host();
			break;
		case DATA_LAYER_ID::BIAS:
			dataLayer->alloc_bias_data_to_host();
			break;
		case DATA_LAYER_ID::DST:
			dataLayer->alloc_dst_data_to_host();
			break;
		
		default:
			return;
	}
}

void DataLayerFactory::Init(DATA_LAYER_ID id)
{
	switch (id)
	{
	case DATA_LAYER_ID::SRC:
		
		break;
	case DATA_LAYER_ID::FILTER:
		dataLayer->init_filter_data();
		break;
	case DATA_LAYER_ID::BIAS:
		dataLayer->init_bias_data();
		break;
	default:
		return;
	}
}

void DataLayerFactory::Init_dst_data(float* dst_data)
{
	dataLayer->init_dst_data(dst_data);
}


void DataLayerFactory::Init_workspace_data(void* workspace_data)
{
	dataLayer->init_workspace_data(workspace_data);
}

float* DataLayerFactory::get_data_d(DATA_LAYER_ID id)
{
	float* device_ptr = nullptr;
	switch (id)
	{
	case DATA_LAYER_ID::SRC:
		device_ptr = dataLayer->get_src_data_d();
		break;
	case DATA_LAYER_ID::FILTER:
		device_ptr = dataLayer->get_filter_data_d();
		break;
	case DATA_LAYER_ID::BIAS:
		device_ptr = dataLayer->get_bias_data_d();
		break;
	case DATA_LAYER_ID::DST:
		device_ptr = dataLayer->get_dst_data_d();
		break;

	default:
		return device_ptr;
	}

	return device_ptr;
}

float* DataLayerFactory::get_data_h(DATA_LAYER_ID id)
{
	float* host_ptr = nullptr;
	switch (id)
	{
	case DATA_LAYER_ID::SRC:
		host_ptr = dataLayer->get_src_data_h();
		break;
	case DATA_LAYER_ID::FILTER:
		host_ptr = dataLayer->get_filter_data_h();
		break;
	case DATA_LAYER_ID::BIAS:
		host_ptr = dataLayer->get_bias_data_h();
		break;
	case DATA_LAYER_ID::DST:
		host_ptr = dataLayer->get_dst_data_h();
		break;
	default:
		return host_ptr;
	}

	return host_ptr;
}

void* DataLayerFactory::get_workspace_data_d(DATA_LAYER_ID id)
{
	void* workspace_ptr_d = nullptr;

	switch (id)
	{
	
	case WORKSPACE:
		workspace_ptr_d = dataLayer->get_workspace_data_d();
		break;
	default:
		break;
	}

	return workspace_ptr_d;
}


void* DataLayerFactory::get_workspace_data_h(DATA_LAYER_ID id)
{
	void* workspace_ptr_host = nullptr;

	switch (id)
	{

	case WORKSPACE:
		workspace_ptr_host = dataLayer->get_workspace_data_h();
		break;
	default:
		break;
	}

	return workspace_ptr_host;
}

void DataLayerFactory::copyDataToDevice(DATA_LAYER_ID id)
{
	switch (id)
	{
	case DATA_LAYER_ID::SRC:
		dataLayer->copySrcDataToDevice();
		break;
	case DATA_LAYER_ID::FILTER:
		dataLayer->copyFilterDataToDevice();
		break;
	case DATA_LAYER_ID::BIAS:
		dataLayer->copyBiasDataToDevice();
		break;
	case DATA_LAYER_ID::DST:
		dataLayer->copyDstDataToDevice();
		break;
	default:
		return;
	}
}

void DataLayerFactory::copyDataToHost(DATA_LAYER_ID id)
{
	switch (id)
	{
	case DATA_LAYER_ID::SRC:
		dataLayer->copySrcDataToHost();
		break;
	case DATA_LAYER_ID::FILTER:
		dataLayer->copyFilterDataToHost();
		break;
	case DATA_LAYER_ID::BIAS:
		dataLayer->copyBiasDataToHost();
		break;
	case DATA_LAYER_ID::DST:
		dataLayer->copyDstDataToHost();
		break;
	default:
		return;
	}
}
