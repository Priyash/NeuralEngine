#include"ImageManager.h"


vector<unsigned char*> ImageManager::getRawImageData()
{
	return image->readRawImageData();
}

vector<Mat> ImageManager::getImageMatrices()
{
	return image->readImages();
}