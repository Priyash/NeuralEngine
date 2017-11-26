#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H
#include"Image.h"




class ImageManager
{

	
	Image* image;
	const string IMAGE_PATH = "Images_Input\\";
public:
	ImageManager()
	{
		image = new ImageLoader();
		image->loadImages(IMAGE_PATH);
	}

	~ImageManager()
	{
		image = nullptr;
	}

	vector<Mat>getImageMatrices();
	vector<unsigned char*>getRawImageData();
	int getImageWidth(Mat src){ return src.cols; }
	int getImageHeight(Mat src){ return src.rows; }
};









#endif