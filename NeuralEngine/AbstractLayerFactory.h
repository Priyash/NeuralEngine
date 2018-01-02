#ifndef ABSTRACTLAYERFACTORY_H
#define ABSTRACTLAYERFACTORY_H

#include"Layer.h"
#include"Util.h"
#include"AbstractDataLayerFactory.h"


class AbstractLayerFactory
{

public:
	AbstractLayerFactory(){}
	~AbstractLayerFactory(){}

	virtual void createLayer() = 0;
	virtual void forward() = 0;
	virtual vector<Mat>getInputLayerData() = 0;
	virtual cudnnHandle_t getCudnnFactoryHandler() = 0;
	virtual cublasHandle_t getCublasFactoryHandler() = 0;
	
};

class InputLayerFactory : public AbstractLayerFactory
{

	ImageManager* imageManager;
	vector<Mat>img_data_list;
public:
	InputLayerFactory();
	~InputLayerFactory();
	void createLayer();

	//ABSTRACT CLASS METHODS 
	vector<Mat>getInputLayerData();
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
	vector<Mat>getInputLayerData(){ vector<Mat>list; return list; }
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
	AbstractLayerFactory* inputLayerFactory;
	AbstractDataLayerFactory* DataLayer;
public:
	ConvLayerFactory(const AbstractLayerFactory&, const AbstractLayerFactory&, const AbstractDataLayerFactory&);
	~ConvLayerFactory();
	void createLayer();
	void forward(AbstractDataLayerFactory*);

	//ABSTRACT CLASS METHODS 
	vector<Mat>getInputLayerData(){ vector<Mat>list; return list; }
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