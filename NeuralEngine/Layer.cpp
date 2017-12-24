#include"Layer.h"



void AbstractLayer::check_cuda_status(cudnnStatus_t status, string error_module)
{
	if (status != CUDNN_STATUS_SUCCESS)
	{
		throw CudaException(status, error_module);
	}
	return;
}

InputLayer::InputLayer(const InputShape& shape)
{
	this->shape = shape;
}

InputLayer::~InputLayer()
{

}


void InputLayer::allocateInputData(Mat data)
{
	int image_bytes = shape.batch_size*shape.channels*shape.rows*shape.cols*sizeof(float);
	float* d_input = nullptr;
	cudaMalloc(&d_input, image_bytes);
	cudaMemcpy(d_input, data.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);
	
}

void InputLayer::createTensorDescriptor()
{
	try
	{
		status = cudnnCreateTensorDescriptor(&input_descriptor);
		check_cuda_status(status, "Input_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

void InputLayer::setTensorDescriptor()
{
	try
	{
		status = cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, shape.batch_size, shape.channels, shape.rows, shape.cols);
		check_cuda_status(status, "Set_Input_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}