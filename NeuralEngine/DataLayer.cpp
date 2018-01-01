#include "DataLayer.h"

void AbstractDataLayer::check_cuda_status(cudaError_t status, string error_module)
{
	if (status != cudaSuccess)
	{
		throw CudaException(status, error_module);
	}
	return;
}

FilterDataLayer::FilterDataLayer()
{
	compute_filter_size();
}


FilterDataLayer::~FilterDataLayer()
{

}


void FilterDataLayer::init_data()
{
	
	for (int i = 0; i < w_size; i++)
	{
		filter_weight_h[i] = gen_random_number();
	}

}

float* FilterDataLayer::getHostData()
{
	return filter_weight_h;
}

float* FilterDataLayer::getDeviceData()
{
	return filter_weight_d;
}


double FilterDataLayer::gen_random_number()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(filter_weight_min_value, filter_weight_max_value);

	return dis(gen);
}

void FilterDataLayer::compute_filter_size()
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
	allocate_gpu_memory(w_size);
	
}

void FilterDataLayer::allocate_gpu_memory(int w_size)
{
	try
	{
		status = cudaMalloc(&filter_weight_d, w_size*sizeof(float));
		check_cuda_status(status, "cudaMalloc_filter_weights");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

float* FilterDataLayer::copyDataToDevice()
{
	
	if (filter_weight_h != NULL)
	{
		try
		{
			status = cudaMemcpy(filter_weight_d, filter_weight_h, w_size*sizeof(float), cudaMemcpyHostToDevice);
			check_cuda_status(status, "cudaMemcpy_filter_weights");
		}
		catch (CudaException& ce)
		{
			cout << ce.what() << endl;
		}
	}
}