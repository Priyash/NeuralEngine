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

	virtual vector<string> loadImages(const string& image_base_path) = 0;
	virtual vector<Mat> readImages(vector<string>image_list) = 0;
	virtual vector<Mat>normalize_images(vector<Mat>img) = 0;


};


class ImageLoader : public Image
{

	HANDLE handler;
	WIN32_FIND_DATA fdata;
	vector<Mat>image_matrices;
	string image_base_path = "";
public:
	ImageLoader()
	{

	}

	~ImageLoader()
	{
		image_matrices.clear();
	}

	virtual vector<string> loadImages(const string& image_base_path);
	virtual vector<Mat>readImages(vector<string>image_list);
	virtual vector<Mat>normalize_images(vector<Mat>img);

private:
	string ToString(WCHAR* data);
	vector<string> list_images(const string& image_base_path);
	Mat readImage(string image_file);
};


#endif