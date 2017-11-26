#ifndef IMAGE_H
#define IMAGE_H
#include<vector>
#include<string>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include<Windows.h>
#include<iostream>


using namespace std;
using namespace cv;


class Image
{

public:
	Image()
	{

	}

	~Image()
	{

	}

	virtual void loadImages(const string& image_base_path) = 0;
	virtual vector<Mat> readImages() = 0;
	virtual vector<unsigned char*>readRawImageData() = 0;

};


class ImageLoader : public Image
{

	HANDLE handler;
	WIN32_FIND_DATA fdata;
	vector<string>image_list;
	vector<Mat>image_matrices;
	vector<unsigned char*>raw_image_data_list;
	string image_base_path = "";
public:
	ImageLoader()
	{

	}

	~ImageLoader()
	{

	}

	virtual void loadImages(const string& image_base_path);
	virtual vector<Mat> readImages();
	virtual vector<unsigned char*>readRawImageData();

private:
	string ToString(WCHAR* data);
	vector<string> list_images(const string& image_base_path);
	Mat readImage(string image_file);
};



#endif