#include"Layer.h"



void AbstractTensorLayer::check_cuda_status(cudnnStatus_t status, string error_module)
{
	if (status != CUDNN_STATUS_SUCCESS)
	{
		throw CudaException(status, error_module);
	}
	return;
}

TensorLayer::TensorLayer(const TensorShape& shape)
{
	this->shape = shape;
}

TensorLayer::~TensorLayer()
{
}


void TensorLayer::createTensorDescriptor()
{
	try
	{
		status = cudnnCreateTensorDescriptor(&descriptor);
		check_cuda_status(status, "Tensor_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

void TensorLayer::setTensorDescriptor()
{
	try
	{
		status = cudnnSetTensor4dDescriptor(descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, shape.batch_size, shape.feature_map, shape.rows, shape.cols);
		check_cuda_status(status, "Set_Tensor_Descriptor");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}
}

cudnnTensorDescriptor_t TensorLayer::getTensorDescriptor()
{
	return descriptor;
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

ConvTensorLayer::ConvTensorLayer(const ConvShape& shape)
{
	this->shape = shape;
}

ConvTensorLayer::~ConvTensorLayer()
{

}

void ConvTensorLayer::createTensorDescriptor()
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

void ConvTensorLayer::setTensorDescriptor()
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

cudnnConvolutionDescriptor_t ConvTensorLayer::getConvDescriptor()
{
	return convolution_descriptor;
}


TensorShape ConvTensorLayer::getConvolutedImagedOutDim(cudnnConvolutionDescriptor_t conDesc,cudnnTensorDescriptor_t inDesc , cudnnFilterDescriptor_t filterDesc)
{
	TensorShape outImageShape;
	try
	{
		status = cudnnGetConvolution2dForwardOutputDim(conDesc, inDesc, filterDesc, &outImageShape.batch_size, &outImageShape.feature_map, &outImageShape.rows, &outImageShape.cols);
		check_cuda_status(status, "cudnnGetConvolution2dForwardOutputDim");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}

	return outImageShape;
}





cudnnConvolutionFwdAlgo_t ConvTensorLayer::getConvFwdAlgo(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc, cudnnConvolutionFwdPreference_t convFwdPref, size_t mem_limit_bytes, cudnnConvolutionFwdAlgo_t conv_fwd_algo)
{
	cudnnConvolutionFwdAlgo_t convolution_algorithm;

	try
	{
		status = cudnnGetConvolutionForwardAlgorithm(cudnn, inDesc, filDesc, convDesc, outDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm);
		check_cuda_status(status,"cudnnGetConvolutionForwardAlgorithm");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}

	return convolution_algorithm;
}


size_t ConvTensorLayer::getConvForwardWorkSpacesize(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inDesc, cudnnFilterDescriptor_t filDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t outDesc, cudnnConvolutionFwdAlgo_t conv_algo)
{
	size_t workspace_bytes;
	try
	{
		status = cudnnGetConvolutionForwardWorkspaceSize(cudnn, inDesc, filDesc, convDesc, outDesc, conv_algo, &workspace_bytes);
		check_cuda_status(status , "cudnnGetConvolutionForwardWorkspaceSize");
	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}

	return workspace_bytes;
}







//==========================================================================CONV_TENSOR_END=============================================