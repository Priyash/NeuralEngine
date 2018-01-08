#include "DataLayer.h"


DataLayer::DataLayer(float* src_data)
{
	this->src_data_h = src_data;
	Util::getInstance()->read_Json();
}

DataLayer::~DataLayer()
{

}


double DataLayer::gen_random_number()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(filter_weight_min_value, filter_weight_max_value);

	return dis(gen);
}

void DataLayer::compute_src_data_size(int src_data_size)
{
	this->src_data_size = src_data_size;
}

void DataLayer::compute_filter_data_size()
{
	//READING DATA FROM CONFIG FILE FOR THE FILTERS
	vector<Value::ConstMemberIterator>conv_layer_list = Util::getInstance()->getObjects(CONFIG::CONVLAYER1);
	vector<JSON_VALUE>filter_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::FILTER_SHAPE]);
	filter_in_feature_map = filter_data_list[FILTER_SHAPE_ID::FILTER_INPUT_FEATURE_MAP].json_int_value;
	filter_out_feature_map = filter_data_list[FILTER_SHAPE_ID::FILTER_OUTPUT_FEATURE_MAP].json_int_value;
	filter_width = filter_data_list[FILTER_SHAPE_ID::FILTER_WIDTH].json_int_value;
	filter_height = filter_data_list[FILTER_SHAPE_ID::FILTER_HEIGHT].json_int_value;
	filter_weight_min_value = filter_data_list[FILTER_SHAPE_ID::FILTER_WEIGHT_MIN_VALUE].json_int_value;
	filter_weight_max_value = filter_data_list[FILTER_SHAPE_ID::FILTER_WEIGHT_MAX_VALUE].json_int_value;
	filter_data_size = filter_in_feature_map*filter_out_feature_map*filter_width*filter_height;
}

void DataLayer::compute_bias_data_size()
{

	//READING DATA FROM CONFIG FILE FOR THE BIAS
	vector<Value::ConstMemberIterator>conv_layer_list = Util::getInstance()->getObjects(CONFIG::CONVLAYER1);
	vector<JSON_VALUE>bias_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::BIAS_SHAPE]);
	bias_out_feature_map = bias_data_list[BIAS_SHAPE_ID::BIAS_OUTPUT_FEATURE_MAP].json_int_value;
	bias_data_size = bias_out_feature_map;
}

void DataLayer::compute_dst_data_size(int batch, int out_feature_map, int width, int height)
{
	this->dst_batch = batch;
	this->dst_out_feature_map = out_feature_map;
	this->dst_width = width;
	this->dst_height = height;

	dst_data_size = dst_batch*dst_out_feature_map*dst_width*dst_height;
}


void DataLayer::compute_workspace_size(size_t workspace_byte)
{
	this->workspace_byte = workspace_byte;
}


void DataLayer::alloc_src_data_to_device()
{
	try
	{
		status = cudaMalloc(&src_data_d, src_data_size*sizeof(float));
		Util::getInstance()->check_cuda_status(status , "cudaMalloc_src_data");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

void DataLayer::alloc_filter_data_to_device()
{
	try
	{
		status = cudaMalloc(&filter_data_d, filter_data_size*sizeof(float));
		Util::getInstance()->check_cuda_status(status, "cudaMalloc_filter_data");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

void DataLayer::alloc_bias_data_to_device()
{
	try
	{
		status = cudaMalloc(&bias_data_d, bias_data_size*sizeof(float));
		Util::getInstance()->check_cuda_status(status, "cudaMalloc_bias_data");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}


void DataLayer::alloc_dst_data_to_device()
{
	try
	{
		status = cudaMalloc(&dst_data_d, dst_data_size*sizeof(float));
		Util::getInstance()->check_cuda_status(status, "cudaMalloc_dst_data");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}


void DataLayer::alloc_workspace_data_to_device()
{
	try
	{
		status = cudaMalloc(&workspace_data_d, workspace_byte);
		Util::getInstance()->check_cuda_status(status, "cudaMalloc_workspace_data");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}



void DataLayer::alloc_src_data_to_host()
{

}

void DataLayer::alloc_filter_data_to_host()
{
	//ALLOCATING CPU MEMORY FOR FILTER WEIGHTS
	filter_data_h = new float[filter_data_size];
}

void DataLayer::alloc_bias_data_to_host()
{
	//ALLOCATING CPU MEMORY FOR BIAS WEIGHTS
	bias_data_h = new float[bias_data_size];
}

void DataLayer::alloc_dst_data_to_host()
{
	//ALLOCATING CPU MEMORY FOR DST DATA 
	dst_data_h = new float[dst_data_size];
}

void DataLayer::alloc_workspace_data_to_host()
{
	//TODO : CHECK THIS LATER
}

//INITIALIZE DATAs
void DataLayer::init_filter_data()
{
	for (int i = 0; i < filter_data_size; i++)
	{
		filter_data_h[i] = gen_random_number();
	}
}

void DataLayer::init_bias_data()
{
	for (int i = 0; i < bias_data_size; i++)
	{
		bias_data_h[i] = gen_random_number();
	}
}


void DataLayer::init_dst_data(float* dst_data)
{
	this->dst_data_d = dst_data;
}


void DataLayer::init_workspace_data(void* workspace_data)
{
	this->workspace_data_d = workspace_data;
}


//DEVICE DATA POINTERS
float* DataLayer::get_src_data_d()
{
	return src_data_d;
}
float* DataLayer::get_filter_data_d()
{
	return filter_data_d;
}
float* DataLayer::get_bias_data_d()
{
	return bias_data_d;
}
float* DataLayer::get_dst_data_d()
{
	return dst_data_d;
}

void* DataLayer::get_workspace_data_d()
{
	return workspace_data_d;
}


//HOST DATA POINTERS
float* DataLayer::get_src_data_h()
{
	return src_data_h;
}
float* DataLayer::get_filter_data_h()
{
	return filter_data_h;
}
float* DataLayer::get_bias_data_h()
{
	return bias_data_h;
}
float* DataLayer::get_dst_data_h()
{
	return dst_data_h;
}

void* DataLayer::get_workspace_data_h()
{
	return workspace_data_h;
}


//COPY DATA FROM HOST TO DEVICE
void DataLayer::copySrcDataToDevice()
{
	if (src_data_h != NULL)
	{
		try
		{
			status = cudaMemcpy(src_data_d, src_data_h, src_data_size*sizeof(float), cudaMemcpyHostToDevice);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_src_data");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}

void DataLayer::copyFilterDataToDevice()
{
	if (filter_data_h != NULL)
	{
		try
		{
			status = cudaMemcpy(filter_data_d, filter_data_h, filter_data_size*sizeof(float), cudaMemcpyHostToDevice);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_filter_data");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}

void DataLayer::copyBiasDataToDevice()
{
	if (bias_data_h != NULL)
	{
		try
		{
			status = cudaMemcpy(bias_data_d, bias_data_h, bias_data_size*sizeof(float), cudaMemcpyHostToDevice);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_bias_data");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}

void DataLayer::copyDstDataToDevice()
{
	if (dst_data_h != NULL)
	{
		try
		{
			status = cudaMemcpy(dst_data_d, dst_data_h, dst_data_size*sizeof(float), cudaMemcpyHostToDevice);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_dst_data");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}

void DataLayer::copyWorkspaceDataToDevice()
{
	if (workspace_data_h != NULL)
	{
		try
		{
			status = cudaMemcpy(workspace_data_d, workspace_data_h, workspace_byte, cudaMemcpyHostToDevice);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_workspace_byte_data");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}



//COPY DATA BACK FROM DEVICE TO HOST
void DataLayer::copySrcDataToHost()
{
	if (src_data_d != NULL)
	{
		try
		{
			status = cudaMemcpy(src_data_h, src_data_d, dst_data_size*sizeof(float), cudaMemcpyDeviceToHost);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_src_data_host");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}
void DataLayer::copyFilterDataToHost()
{
	if (filter_data_d != NULL)
	{
		try
		{
			status = cudaMemcpy(filter_data_h, filter_data_d, filter_data_size*sizeof(float), cudaMemcpyDeviceToHost);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_filter_data_host");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}
void DataLayer::copyBiasDataToHost()
{
	if (bias_data_d != NULL)
	{
		try
		{
			status = cudaMemcpy(bias_data_h, bias_data_d, bias_data_size*sizeof(float), cudaMemcpyDeviceToHost);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_bias_data_host");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}
void DataLayer::copyDstDataToHost()
{
	if (dst_data_d != NULL)
	{
		try
		{
			status = cudaMemcpy(dst_data_h, dst_data_d, dst_data_size*sizeof(float), cudaMemcpyDeviceToHost);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_dst_data_host");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}


void DataLayer::copyWorkspaceDataToHost()
{
	if (workspace_data_d != NULL)
	{
		try
		{
			status = cudaMemcpy(workspace_data_h, workspace_data_d, workspace_byte, cudaMemcpyDeviceToHost);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_workspace_data_host");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}



//DATA SIZE GETTER FUNCTION
int DataLayer::getSrcDataSize()
{
	return src_data_size;
}
int DataLayer::getFilterDataSize()
{
	return filter_data_size;
}
int DataLayer::getBiasDataSize()
{
	return bias_data_size;
}
size_t DataLayer::getWorkspaceDataSize()
{
	return workspace_byte;
}
int DataLayer::getDstDataSize()
{
	return dst_data_size;
}