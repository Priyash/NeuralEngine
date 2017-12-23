#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H
#include"Image.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include"Util.h"

enum IMAGE
{
	NORMAL = 1,
	RESIZE = 0
};

class ImageManager
{

	
	Image* image;
	string IMAGE_PATH;
	vector<Mat>resized_img_mat_list;
	string CONFIG_FILE;
public:
	ImageManager()
	{
		image = new ImageLoader();
		CONFIG_FILE = "config.txt";
		Util::getInstance()->read_config_file(CONFIG_FILE);
		Util::getInstance()->parse_config_file(CONFIG_ID::BASE_IMAGE_PATH);
		Util::getInstance()->parse_config_file(CONFIG_ID::IMAGE_RESIZE);
		IMAGE_PATH = Util::getInstance()->getImage_Base_Path();
		cout << IMAGE_PATH << endl;
		image->loadImages(IMAGE_PATH);
	}

	~ImageManager()
	{
		image = nullptr;
		resized_img_mat_list.clear();
	}

	vector<Mat>getImageMatrices(IMAGE img);
	int getImageWidth(Mat src){ return src.cols; }
	int getImageHeight(Mat src){ return src.rows; }
	
private:
	vector<Mat> resize_images(int width, int height, vector<Mat>img_mat_list);
};









#endif