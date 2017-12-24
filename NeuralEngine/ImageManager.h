#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H

#include"ImageFactory.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include"Util.h"


class ImageManager
{
	AbstractImageFactory* Ifactory;
	
public:
	ImageManager()
	{
		Ifactory = new ImageFactory();
	}

	~ImageManager()
	{
		Ifactory = nullptr;
	}

	vector<Mat>getImageMatrices(IMAGE img);
	vector<string>getImageList();
	int getImageWidth(Mat src){ return src.cols; }
	int getImageHeight(Mat src){ return src.rows; }
};


#endif