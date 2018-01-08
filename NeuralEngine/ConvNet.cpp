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
			process_image_data(img_data, batch.img_data_size);
			break;
		}
		break;
	}
}

void ConvNet::process_image_data(float* img_data ,int img_data_size)
{
	
	AbstractLayerFactory* handler = new HandlerFactory();
	handler->createLayer();

	DataLayerFactory* dataHandler = new DataLayerFactory(img_data);
	dataHandler->compute_src_data_size(img_data_size);
	
	AbstractLayerFactory* conv1 = new ConvLayerFactory(handler, dataHandler);
	conv1->createLayer();
	conv1->forward();


	/*DataLayerFactory* dataHandler2 = new DataLayerFactory(dataHandler->get_data_d(DATA_LAYER_ID::DST));
	dataHandler2->compute_src_data_size(dataHandler->getDataSize(DATA_LAYER_ID::DST));
	AbstractLayerFactory* conv2 = new ConvLayerFactory(handler, dataHandler2);
	conv2->createLayer();
	conv2->forward();*/


}

void ConvNet::start_training()
{

}
