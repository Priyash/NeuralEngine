#include "AbstractLayerFactory.h"

InputLayerFactory::InputLayerFactory()
{
	imageManager = new ImageManager();
}

InputLayerFactory::~InputLayerFactory()
{

}


void InputLayerFactory::createLayer()
{
	img_data_list = imageManager->getImageMatrices(IMAGE::RESIZE);
}


vector<Mat>InputLayerFactory::getInputLayerData()
{
	return img_data_list;
}

//==================================================INPUT_LAYER_END===============================================================


//=================================================HANDLER========================================================================


HandlerFactory::HandlerFactory()
{
	handle = new CudnnHandler();
}

HandlerFactory::~HandlerFactory()
{

}


void HandlerFactory::createLayer()
{
	handle->createCudnnHandler();
	handle->createCublasHandler();
}


cudnnHandle_t HandlerFactory::getCudnnFactoryHandler()
{
	return handle->getCudnnHandler();
}


cublasHandle_t HandlerFactory::getCublasFactoryHandler()
{
	return handle->getCublasHandler();
}



//================================================CONV_LAYER=======================================================================

ConvLayerFactory::ConvLayerFactory(const HandlerFactory& hfact, const InputLayerFactory& iFact)
{
	*this->handlerFact = (hfact);
	*this->inputLayerFactory = (iFact);

	//TODO: CHANGE THE UTIL FILES AND ADD JSON PARSER TO PARSE CONFIG FILE
	Util::getInstance()->read_Json();
}

ConvLayerFactory::~ConvLayerFactory()
{

}

void ConvLayerFactory::setShapeData(CONFIG con)
{
	if (con == CONFIG::TENSOR_SHAPE)
	{
		vector<JSON_VALUE>conv_input_shape_data_list = Util::getInstance()->getValues(CONFIG::TENSOR_SHAPE);
		convInputShape.batch_size = conv_input_shape_data_list[0].json_int_value;
		convInputShape.feature_map = conv_input_shape_data_list[1].json_int_value;
		convInputShape.cols = conv_input_shape_data_list[2].json_int_value;
		convInputShape.rows = conv_input_shape_data_list[3].json_int_value;
	}
	else if (con == CONFIG::FILTER_SHAPE)
	{

		vector<JSON_VALUE>conv_filter_shape_data_list = Util::getInstance()->getValues(CONFIG::FILTER_SHAPE);
		convfilterShape.n_input_feature_map = conv_filter_shape_data_list[0].json_int_value;
		convfilterShape.n_output_feature_map = conv_filter_shape_data_list[1].json_int_value;
		convfilterShape.filter_width = conv_filter_shape_data_list[2].json_int_value;
		convfilterShape.filter_height = conv_filter_shape_data_list[3].json_int_value;
	}
	else if (con == CONFIG::CONV_SHAPE)
	{
		vector<JSON_VALUE>conv_shape_data_list = Util::getInstance()->getValues(CONFIG::CONV_SHAPE);
		convShape.pad_width = conv_shape_data_list[0].json_int_value;
		convShape.pad_height = conv_shape_data_list[1].json_int_value;
		convShape.vertical_stride = conv_shape_data_list[2].json_int_value;
		convShape.horizontal_stride = conv_shape_data_list[3].json_int_value;
		convShape.dilation_height = conv_shape_data_list[4].json_int_value;
		convShape.dilation_width = conv_shape_data_list[5].json_int_value;
		convShape.conv_fwd_pref = conv_shape_data_list[6].json_int_value;
	}
	else if (con == CONFIG::TENSOR_SHAPE)
	{
		vector<JSON_VALUE>out_shape_data_list = Util::getInstance()->getValues(CONFIG::TENSOR_SHAPE);
		convOutputShape.batch_size = out_shape_data_list[0].json_int_value;
		convOutputShape.feature_map = out_shape_data_list[1].json_int_value;
		convOutputShape.cols = out_shape_data_list[2].json_int_value;
		convOutputShape.rows = out_shape_data_list[3].json_int_value;
	}
}

void ConvLayerFactory::createLayer()
{

	//INPUT TENSOR DATA
	setShapeData(CONFIG::TENSOR_SHAPE);
	inputTensor = new TensorLayer(convInputShape);
	inputTensor->createTensorDescriptor();
	inputTensor->setTensorDescriptor();

	//FILTER TENSOR DATA
	setShapeData(CONFIG::FILTER_SHAPE);
	filterTensor = new FilterLayer(convfilterShape);
	filterTensor->createTensorDescriptor();
	filterTensor->setTensorDescriptor();

	//CONV TENSOR DATA
	setShapeData(CONFIG::CONV_SHAPE);
	convTensor = new ConvTensorLayer(convShape);
	convTensor->createTensorDescriptor();
	convTensor->setTensorDescriptor();

	//OUTPUT TENSOR DATA
	setShapeData(CONFIG::TENSOR_SHAPE);
	outputTensor = new TensorLayer(convOutputShape);
	outputTensor->createTensorDescriptor();
	outputTensor->setTensorDescriptor();

	TensorShape outImgShape = convTensor->getConvolutedImagedOutDim(convTensor->getConvDescriptor(), 
																	inputTensor->getTensorDescriptor(), 
																	filterTensor->getFilterDescriptor());
	
	cudnnConvolutionFwdAlgo_t fwd_algo = convTensor->getConvFwdAlgo(handlerFact->getCudnnFactoryHandler(), inputTensor->getTensorDescriptor(),
																	filterTensor->getFilterDescriptor(), convTensor->getConvDescriptor(),
																	outputTensor->getTensorDescriptor(), getConvFwdPref(convShape.conv_fwd_pref),
																	0);
	size_t workspace_bytes = convTensor->getConvForwardWorkSpacesize(handlerFact->getCudnnFactoryHandler(), inputTensor->getTensorDescriptor(),
																	filterTensor->getFilterDescriptor(), convTensor->getConvDescriptor(), 
																	outputTensor->getTensorDescriptor(), fwd_algo);



}

cudnnConvolutionFwdPreference_t ConvLayerFactory::getConvFwdPref(int n)
{
	cudnnConvolutionFwdPreference_t conv_pref_fwd;
	switch (n)
	{

	case 0:
		conv_pref_fwd = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
		break;
	case 1:
		conv_pref_fwd = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
		break;
	case 2:
		conv_pref_fwd = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
		break;
	default:
		conv_pref_fwd = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
		break;
	}

	return conv_pref_fwd;
}




//================================================CONV_LAYER=======================================================================