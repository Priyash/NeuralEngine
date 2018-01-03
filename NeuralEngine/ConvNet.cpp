#include "ConvNet.h"


ConvNet::ConvNet()
{
	inputLayerfactory = new InputLayerFactory();
	inputLayerfactory->createLayer();
	vector<Data_Batch>batch_list = inputLayerfactory->getInputLayerData();
	for (auto i : batch_list)
	{
		Data_Batch db = i;
		for (int j = 0; j < db.batch_data_list.size(); j++)
		{
			float* img_data = db.batch_data_list[j];
			//start(img_data);
		}
	}
}


void start(float* img_data)
{
	AbstractDataLayerFactory* dataLayer = new DataLayerFactory();
	dataLayer->set_src_data(img_data);
	
	AbstractLayerFactory* handler = new HandlerFactory();
	handler->createLayer();

	AbstractLayerFactory* conv1 = new ConvLayerFactory(handler, dataLayer);


}

ConvNet::~ConvNet()
{

}
