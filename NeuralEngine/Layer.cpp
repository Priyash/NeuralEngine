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
		status = cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, shape.batch_size, shape.feature_map, shape.rows, shape.cols);
		check_cuda_status(status, "Set_Input_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

cudnnTensorDescriptor_t InputLayer::getTensorDescriptor()
{
	return input_descriptor;
}


//=========================================================CONVOLUTIONAL_KERNEL================================================================

FilterLayer::FilterLayer(const FilterShape& shape)
{

}

FilterLayer::~FilterLayer()
{

}

void FilterLayer::createTensorDescriptor()
{
	try
	{
		status = cudnnCreateFilterDescriptor(&filter_descriptor);
		check_cuda_status(status , "Filter_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

void FilterLayer::setTensorDescriptor()
{
	try
	{
		status = cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, shape.n_output_feature_map, shape.n_input_feature_map, shape.filter_height, shape.filter_width);
		check_cuda_status(status, "Set_Filter_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

cudnnFilterDescriptor_t FilterLayer::getFilterDescriptor()
{
	return filter_descriptor;
}

//=======================================================================CONVOLUTIONAL_KERNEL_END===============================================


//=============================================================================BIAS_TENSOR============================================================

Bias::Bias(const BiasShape& shape)
{
	this->bias_shape = shape;
}

Bias::~Bias()
{
}


void Bias::createTensorDescriptor()
{
	try
	{
		status = cudnnCreateTensorDescriptor(&bias_descriptor);
		check_cuda_status(status, "Bias_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

void Bias::setTensorDescriptor()
{
	try
	{
		status = cudnnSetTensor4dDescriptor(bias_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, bias_shape.batch_size, bias_shape.feature_map, bias_shape.rows, bias_shape.cols);
		check_cuda_status(status, "Set_Bias_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

cudnnTensorDescriptor_t Bias::getTensorDescriptor()
{
	return bias_descriptor;
}


//==========================================================================BIAS_TENSOR_END============================================================


//===========================================================================CONV_TENSOR============================================================

ConvLayer::ConvLayer(const ConvShape& shape)
{
	this->shape = shape;
}

ConvLayer::~ConvLayer()
{

}

void ConvLayer::createTensorDescriptor()
{
	try
	{
		status = cudnnCreateConvolutionDescriptor(&convolution_descriptor);
		check_cuda_status(status, "Convolution_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

void ConvLayer::setTensorDescriptor()
{
	try
	{
		status = cudnnSetConvolution2dDescriptor_v5(convolution_descriptor, shape.pad_height, shape.pad_width, shape.vertical_stride, shape.horizontal_stride, shape.dilation_height, shape.dilation_width, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
		check_cuda_status(status, "Set_Convolution_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

cudnnConvolutionDescriptor_t ConvLayer::getConvDescriptor()
{
	return convolution_descriptor;
}


//==========================================================================CONV_TENSOR_END============================================================