#include "ConvNet.h"


ConvNet::ConvNet()
{
	inputLayerfactory = new InputLayerFactory();
	
}

ConvNet::~ConvNet()
{

}

void ConvNet::process_Data_Batch()
{
	inputLayerfactory->createLayer();
	vector<Data_Batch>batch_list = inputLayerfactory->getInputLayerData();
	for (auto i : batch_list)
	{
		Data_Batch batch = i;
		for (int j = 0; j < batch.batch_data_list.size(); j++)
		{
			float* img_data = batch.batch_data_list[j];
			process_image_data(img_data);
		}
	}
}

void ConvNet::process_image_data(float* img_data)
{
	
	AbstractLayerFactory* handler = new HandlerFactory();
	handler->createLayer();
	DataLayerFactory* dataHandler = new DataLayerFactory(img_data);

	AbstractLayerFactory* conv1 = new ConvLayerFactory(handler, dataHandler);
	conv1->createLayer();
	conv1->forward();


	/*DataLayerFactory* dataHandler2 = new DataLayerFactory(dataHandler->get_data_d(DATA_LAYER_ID::DST));
	AbstractLayerFactory* conv2 = new ConvLayerFactory(handler, dataHandler2);
	conv2->createLayer();
	conv2->forward();*/


}

void ConvNet::start_training()
{

}
