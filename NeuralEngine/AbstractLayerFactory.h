#ifndef ABSTRACTLAYERFACTORY_H
#define ABSTRACTLAYERFACTORY_H
#include"ImageManager.h"
#include "Layer.h"
#include"AbstractDataLayerFactory.h"


struct Data_Batch
{
	string batch_ID;
	vector<float*> batch_data_list;
};

class AbstractLayerFactory
{

public:
	AbstractLayerFactory(){}
	~AbstractLayerFactory(){}

	virtual void createLayer() = 0;
	virtual void forward() = 0;
	virtual vector<Data_Batch>getInputLayerData() = 0;
	virtual cudnnHandle_t getCudnnFactoryHandler() = 0;
	virtual cublasHandle_t getCublasFactoryHandler() = 0;
	
};

class InputLayerFactory : public AbstractLayerFactory
{
	ImageManager* imageManager;
	vector<Mat>img_data_list;
	int batch_size;
	vector<Data_Batch>batch_list;
	string batch_prefix_name;
public:
	InputLayerFactory();
	~InputLayerFactory();
	void createLayer();

	//ABSTRACT CLASS METHODS 
	vector<Data_Batch>getInputLayerData();
	void forward(){}
	cudnnHandle_t getCudnnFactoryHandler(){ cudnnHandle_t s; return s; }
	cublasHandle_t getCublasFactoryHandler(){ cublasHandle_t s; return s; }
	
};

class HandlerFactory : public AbstractLayerFactory
{
	CudnnHandler* handle;
public:
	HandlerFactory();
	~HandlerFactory();

	void createLayer();
	cudnnHandle_t getCudnnFactoryHandler();
	cublasHandle_t getCublasFactoryHandler();

	//ABSTRACT CLASS METHODS 
	vector<Data_Batch>getInputLayerData(){ vector<Data_Batch>list; return list; }
	void forward(){}
};

class ConvLayerFactory : public AbstractLayerFactory
{
	//TENSOR CLASS HANDLER
	AbstractTensorLayer* inputTensor;
	AbstractTensorLayer* filterTensor;
	AbstractTensorLayer* convTensor;
	AbstractTensorLayer* outputTensor;
	AbstractTensorLayer* biasTensor;
	
	//LAYER FACTORY HANDLER
	AbstractLayerFactory* handlerFact;
	AbstractDataLayerFactory* dataLayerFactory;
public:
	//ConvLayerFactory(AbstractLayerFactory&,AbstractLayerFactory&, const AbstractDataLayerFactory&);
	ConvLayerFactory(AbstractLayerFactory* hfact, AbstractDataLayerFactory* dataLayerFactory);
	~ConvLayerFactory();
	void createLayer();
	void forward();

	//ABSTRACT CLASS METHODS 
	vector<Data_Batch>getInputLayerData(){ vector<Data_Batch>list; return list; }
	cudnnHandle_t getCudnnFactoryHandler(){ cudnnHandle_t s; return s; }
	cublasHandle_t getCublasFactoryHandler(){ cublasHandle_t s; return s; }

private:
	TensorShape convInputShape;
	FilterShape convfilterShape;
	ConvShape convShape;
	TensorShape convOutputShape;
	BiasShape biasShape;

	void setConvShapeData(CONFIG con);
	cudnnConvolutionFwdPreference_t getConvFwdPref(int);
};



#endif