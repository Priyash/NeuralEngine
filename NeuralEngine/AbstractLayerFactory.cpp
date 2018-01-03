#include "AbstractLayerFactory.h"

InputLayerFactory::InputLayerFactory()
{
	imageManager = new ImageManager();
	batch_prefix_name = "Batch_";
	Util::getInstance()->read_Json();
	vector<Value::ConstMemberIterator>obj_list = Util::getInstance()->getObjects(CONFIG::CONVLAYER1);
	vector<JSON_VALUE>input_shape_list = Util::getInstance()->getValues(obj_list[TENSOR_SHAPE]);
	batch_size = input_shape_list[TENSOR_SHAPE_ID::TENSOR_BATCH_SIZE].json_int_value;
}

InputLayerFactory::~InputLayerFactory()
{

}


void InputLayerFactory::createLayer()
{
	img_data_list = imageManager->getImageMatrices(IMAGE::RESIZE);
	
	for (int i = 0; i < img_data_list.size(); i++)
	{
		Data_Batch db;
		if (i == batch_size - 1)
		{
			batch_list.push_back(db);
			db.batch_data_list.clear();
		}
		else
		{
			db.batch_ID = batch_prefix_name + to_string(i);
			Mat img_obj = img_data_list[i];
			float* img_data_h = img_obj.ptr<float>(0);
			db.batch_data_list.push_back(img_data_h);
		}
	}
}


vector<Data_Batch>InputLayerFactory::getInputLayerData()
{
	return batch_list;
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

/*ConvLayerFactory::ConvLayerFactory(const AbstractLayerFactory& hfact, const AbstractLayerFactory& iFact ,const AbstractDataLayerFactory& dataLayerFactory)
{
	*this->handlerFact = (hfact);
	*this->inputLayerFactory = (iFact);
	*this->dataLayerFactory = (dataLayerFactory);
	//TODO: CHANGE THE UTIL FILES AND ADD JSON PARSER TO PARSE CONFIG FILE
	Util::getInstance()->read_Json();

}*/

ConvLayerFactory::ConvLayerFactory(AbstractLayerFactory* hfact,AbstractDataLayerFactory* dataLayerFactory)
{
	this->handlerFact = hfact;
	this->dataLayerFactory = dataLayerFactory;
	Util::getInstance()->read_Json();

}

ConvLayerFactory::~ConvLayerFactory()
{

}


//INITIALIZE THE CONV DATA 
void ConvLayerFactory::setConvShapeData(CONFIG con)
{
	//LIST OF ALL THE CONV LAYER OBJECTS
	vector<Value::ConstMemberIterator>conv_layer_list = Util::getInstance()->getObjects(con);

	//SETTING DATA FOR THE CONVOLUTION 
	//TENSOR SHAPE DATA FOR THE INPUT TENSOR
	vector<JSON_VALUE>conv_input_shape_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::TENSOR_SHAPE]);
	convInputShape.batch_size = conv_input_shape_data_list[TENSOR_SHAPE_ID::TENSOR_BATCH_SIZE].json_int_value;
	convInputShape.feature_map = conv_input_shape_data_list[TENSOR_SHAPE_ID::TENSOR_INPUT_FEATURE_MAP].json_int_value;
	convInputShape.cols = conv_input_shape_data_list[TENSOR_SHAPE_ID::TENSOR_IMAGE_WIDTH].json_int_value;
	convInputShape.rows = conv_input_shape_data_list[TENSOR_SHAPE_ID::TENSOR_IMAGE_HEIGHT].json_int_value;

	//BIAS SHAPE DATA FOR THE CONVOLUTION BIAS TENSOR
	vector<JSON_VALUE>bias_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::BIAS_SHAPE]);
	biasShape.batch_size = bias_data_list[BIAS_SHAPE_ID::BIAS_BATCH_SIZE].json_int_value;
	biasShape.feature_map = bias_data_list[BIAS_SHAPE_ID::BIAS_OUTPUT_FEATURE_MAP].json_int_value;
	biasShape.cols = bias_data_list[BIAS_SHAPE_ID::BIAS_WIDTH].json_int_value;
	biasShape.rows = bias_data_list[BIAS_SHAPE_ID::BIAS_HEIGHT].json_int_value;

	//FILTER SHAPE DATA FOR THE CONVOLUTION FILTER TENSOR
	vector<JSON_VALUE>conv_filter_shape_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::FILTER_SHAPE]);
	convfilterShape.n_input_feature_map = conv_filter_shape_data_list[FILTER_SHAPE_ID::FILTER_INPUT_FEATURE_MAP].json_int_value;
	convfilterShape.n_output_feature_map = conv_filter_shape_data_list[FILTER_SHAPE_ID::FILTER_OUTPUT_FEATURE_MAP].json_int_value;
	convfilterShape.filter_width = conv_filter_shape_data_list[FILTER_SHAPE_ID::FILTER_WIDTH].json_int_value;
	convfilterShape.filter_height = conv_filter_shape_data_list[FILTER_SHAPE_ID::FILTER_HEIGHT].json_int_value;
	
	//CONV SHAPE DATA FOR THE CONVOLUTION TENSOR
	vector<JSON_VALUE>conv_shape_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::CONV_SHAPE]);
	convShape.pad_width = conv_shape_data_list[CONV_SHAPE_ID::CONV_PAD_WIDTH].json_int_value;
	convShape.pad_height = conv_shape_data_list[CONV_SHAPE_ID::CONV_DILATION_HEIGHT].json_int_value;
	convShape.vertical_stride = conv_shape_data_list[CONV_SHAPE_ID::CONV_VERTICAL_STRIDE].json_int_value;
	convShape.horizontal_stride = conv_shape_data_list[CONV_SHAPE_ID::CONV_HORIZONTAL_STRIDE].json_int_value;
	convShape.dilation_height = conv_shape_data_list[CONV_SHAPE_ID::CONV_DILATION_HEIGHT].json_int_value;
	convShape.dilation_width = conv_shape_data_list[CONV_SHAPE_ID::CONV_DILATION_WIDTH].json_int_value;
	convShape.conv_fwd_pref = conv_shape_data_list[CONV_SHAPE_ID::CONV_CONV_FWD_PREF].json_int_value;
	
	//TENSOR SHAPE DATA FOR THE CONVOLUTION OUTPUT TENSOR
	vector<JSON_VALUE>out_shape_data_list = Util::getInstance()->getValues(conv_layer_list[LAYER_ID::TENSOR_SHAPE]);
	convOutputShape.batch_size = out_shape_data_list[TENSOR_SHAPE_ID::TENSOR_BATCH_SIZE].json_int_value;
	convOutputShape.feature_map = out_shape_data_list[TENSOR_SHAPE_ID::TENSOR_INPUT_FEATURE_MAP].json_int_value;
	convOutputShape.cols = out_shape_data_list[TENSOR_SHAPE_ID::TENSOR_IMAGE_WIDTH].json_int_value;
	convOutputShape.rows = out_shape_data_list[TENSOR_SHAPE_ID::TENSOR_IMAGE_HEIGHT].json_int_value;
	
}

void ConvLayerFactory::createLayer()
{
	setConvShapeData(CONFIG::CONVLAYER1);

	//INPUT TENSOR DATA
	inputTensor = new TensorLayer(convInputShape);
	inputTensor->createTensorDescriptor();
	inputTensor->setTensorDescriptor();

	//FILTER TENSOR DATA
	filterTensor = new FilterLayer(convfilterShape);
	filterTensor->createTensorDescriptor();
	filterTensor->setTensorDescriptor();

	//BIAS TENSOR DATA
	biasTensor = new Bias(biasShape);
	biasTensor->createTensorDescriptor();
	biasTensor->setTensorDescriptor();

	//CONV TENSOR DATA
	convTensor = new ConvTensorLayer(convShape);
	convTensor->createTensorDescriptor();
	convTensor->setTensorDescriptor();

	//OUTPUT TENSOR DATA
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

	//INITIALIZING THE FILTER WEIGHTS WITH RANDOM DATA INCLUDING BIAS WEIGHTS
	dataLayerFactory->init();
	//AFTER INITIALIZING COPY THE DATA FROM CPU TO GPU 
	dataLayerFactory->copyFilterDataToDevice();
	dataLayerFactory->copyBiasDataToDevice();
	//ALLOCATE THE GPU MEMORY FOR THE OUTPUT DATA FOR CONVOLUTION RESULT[output]
	dataLayerFactory->alloc_out_data_gpu(outImgShape.batch_size, outImgShape.feature_map, outImgShape.cols, outImgShape.rows);

}


void ConvLayerFactory::forward()
{
	
	//ADD CONVOLUTION FORWARD 
	//convTensor->conv_forward(handlerFact->getCudnnFactoryHandler(), 1.0f, inputTensor->getTensorDescriptor(), inputLayerFactory->getInputLayerData(),
		//filterTensor->getFilterDescriptor(),)
	//ADD BIAS
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