#include "ConvNet.h"


ConvNet::ConvNet()
{
	inputfactory = new InputLayerFactory();
	inputfactory->createInputLayer();
}


ConvNet::~ConvNet()
{

}
