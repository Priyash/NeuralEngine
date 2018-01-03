#include "DataLayer.h"


DataLayer::DataLayer()
{
	filter_weight_d = NULL;
	bias_weight_d = NULL;
	src_data_d = NULL;

	compute_filter_size();
	compute_bias_size();
}


DataLayer::~DataLayer()
{

}


void DataLayer::init_data()
{
	
	for (int i = 0; i < w_size; i++)
	{
		filter_weight_h[i] = gen_random_number();
	}

	for (int i = 0; i < b_size; i++)
	{
		bias_weight_h[i] = gen_random_number();
	}

}

void DataLayer::init_data(float* filter_data, float* bias_data)
{
	filter_weight_h = filter_data;
	bias_weight_h = bias_data;
}

void DataLayer::set_src_data(float* src_data)
{
	this->src_data = src_data;
	compute_src_data_size();
}


vector<float*> DataLayer::getHostData()
{
	vector<float*>weight_list_h;
	weight_list_h.push_back(filter_weight_h);
	weight_list_h.push_back(bias_weight_h);

	return weight_list_h;
}

vector<float*>  DataLayer::getDeviceData()
{
	vector<float*>weight_list_d;
	weight_list_d.push_back(filter_weight_d);
	weight_list_d.push_back(bias_weight_d);

	return weight_list_d;
}


float* DataLayer::getSrcDataHost()
{
	return src_data;
}

float* DataLayer::getSrcDataDevice()
{
	return src_data_d;
}

double DataLayer::gen_random_number()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(filter_weight_min_value, filter_weight_max_value);

	return dis(gen);
}

DataLayerResult DataLayer::getResult()
{
	return result;
}



void DataLayer::compute_src_data_size()
{
	Util::getInstance()->read_Json();
	vector<Value::ConstMemberIterator>conv_layer_list = Util::getInstance()->getObjects(CONFIG::CONVLAYER1);
	vector<JSON_VALUE>input_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::TENSOR_SHAPE]);
	int batch_size = input_data_list[TENSOR_SHAPE_ID::TENSOR_BATCH_SIZE].json_int_value;
	int input_feature_map = input_data_list[TENSOR_SHAPE_ID::TENSOR_INPUT_FEATURE_MAP].json_int_value;
	int width = input_data_list[TENSOR_SHAPE_ID::TENSOR_IMAGE_WIDTH].json_int_value;
	int height = input_data_list[TENSOR_SHAPE_ID::TENSOR_IMAGE_HEIGHT].json_int_value;

	src_data_size = batch_size*input_feature_map*width*height;
	allocate_gpu_src_data_memory(src_data_size);
}

void DataLayer::compute_filter_size()
{
	//FETCHING THE FILTER DATA FROM CONFIG FILE
	Util::getInstance()->read_Json();
	vector<Value::ConstMemberIterator>conv_layer_list = Util::getInstance()->getObjects(CONFIG::CONVLAYER1);
	vector<JSON_VALUE>filter_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::FILTER_SHAPE]);
	int in_feature_map = filter_data_list[FILTER_SHAPE_ID::FILTER_INPUT_FEATURE_MAP].json_int_value;
	int out_feature_map = filter_data_list[FILTER_SHAPE_ID::FILTER_OUTPUT_FEATURE_MAP].json_int_value;
	int filter_width = filter_data_list[FILTER_SHAPE_ID::FILTER_WIDTH].json_int_value;
	int filter_height = filter_data_list[FILTER_SHAPE_ID::FILTER_HEIGHT].json_int_value;
	filter_weight_min_value = filter_data_list[FILTER_SHAPE_ID::FILTER_WEIGHT_MIN_VALUE].json_int_value;
	filter_weight_max_value = filter_data_list[FILTER_SHAPE_ID::FILTER_WEIGHT_MAX_VALUE].json_int_value;

	//ALLOCATING CPU MEMORY FOR FILTER WEIGHTS
	w_size = in_feature_map*out_feature_map*filter_width*filter_height;
	filter_weight_h = new float[w_size];
	//ALLOCATING GPU MEMORY FOR FILTER WEIGHTS
	allocate_gpu_filter_weight_memory(w_size);
	
}

void DataLayer::compute_bias_size()
{
	//FETCHING THE FILTER DATA FROM CONFIG FILE
	Util::getInstance()->read_Json();
	vector<Value::ConstMemberIterator>conv_layer_list = Util::getInstance()->getObjects(CONFIG::CONVLAYER1);
	vector<JSON_VALUE>bias_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::BIAS_SHAPE]);
	int out_feature_map = bias_data_list[FILTER_SHAPE_ID::FILTER_OUTPUT_FEATURE_MAP].json_int_value;
	int filter_width = bias_data_list[FILTER_SHAPE_ID::FILTER_WIDTH].json_int_value;
	int filter_height = bias_data_list[FILTER_SHAPE_ID::FILTER_HEIGHT].json_int_value;
	

	//ALLOCATING CPU MEMORY FOR FILTER WEIGHTS
	b_size = out_feature_map;
	bias_weight_h = new float[b_size];
	//ALLOCATING GPU MEMORY FOR FILTER WEIGHTS
	allocate_gpu_bias_weight_memory(b_size);

}


void DataLayer::allocate_gpu_src_data_memory(int src_size)
{
	try
	{
		status = cudaMalloc(&src_data, src_size*sizeof(float));
		Util::getInstance()->check_cuda_status(status, "cudaMalloc_src_data");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}

}

void DataLayer::allocate_gpu_filter_weight_memory(int w_size)
{
	try
	{
		status = cudaMalloc(&filter_weight_d, w_size*sizeof(float));
		Util::getInstance()->check_cuda_status(status, "cudaMalloc_filter_weights");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

void DataLayer::allocate_gpu_bias_weight_memory(int b_size)
{
	try
	{
		status = cudaMalloc(&bias_weight_d, b_size*sizeof(float));
		Util::getInstance()->check_cuda_status(status, "cudaMalloc_bias_weights");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

void DataLayer::alloc_out_data_gpu(int batch, int out_feature_map, int w, int h)
{
	int o_size = batch*out_feature_map*w*h;
	DataLayerResult result;
	try
	{
		status = cudaMalloc(&result.output_d, o_size*sizeof(float));
		Util::getInstance()->check_cuda_status(status, "cudaMalloc_out_result");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}


void DataLayer::copySrcsDataToDevice()
{
	if (src_data != NULL)
	{
		try
		{
			status = cudaMemcpy(src_data_d, src_data, src_data_size*sizeof(float), cudaMemcpyHostToDevice);
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
	if (filter_weight_h != NULL)
	{
		try
		{
			status = cudaMemcpy(filter_weight_d, filter_weight_h, w_size*sizeof(float), cudaMemcpyHostToDevice);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_filter_weights");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}


void DataLayer::copyBiasDataToDevice()
{
	if (bias_weight_h != NULL)
	{
		try
		{
			status = cudaMemcpy(bias_weight_d, bias_weight_h, b_size*sizeof(float), cudaMemcpyHostToDevice);
			Util::getInstance()->check_cuda_status(status, "cudaMemcpy_bias_weights");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}

