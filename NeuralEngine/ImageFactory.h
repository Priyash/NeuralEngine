#ifndef IMAGEFACTORY_H
#define IMAGEFACTORY_H
#include"Image.h"
#include"Util.h"



using namespace std;

enum IMAGE
{
	NORMAL = 1,
	RESIZE = 0
};


class AbstractImageFactory
{
public:
	AbstractImageFactory(){}
	~AbstractImageFactory(){}

	virtual vector<Mat> loadAndReadImages() = 0;
	virtual vector<Mat> Normalize_images(vector<Mat>img_list) = 0;
	virtual vector<Mat> ResizeImages(vector<Mat>img_list) = 0;
	virtual vector<string> getImageList() = 0;
};


class ImageFactory :public AbstractImageFactory
{
	Image* image;
	string IMAGE_PATH;
	string CONFIG_FILE;
public:
	ImageFactory();
	~ImageFactory();
	vector<Mat> loadAndReadImages();
	vector<Mat> Normalize_images(vector<Mat>img_list);
	vector<Mat> ResizeImages(vector<Mat>img_list);
	vector<string> getImageList();
};

#endif