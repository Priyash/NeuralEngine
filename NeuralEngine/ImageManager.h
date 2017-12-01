#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H
#include"Image.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

struct Pixel
{
	float r;
	float g;
	float b;
};

enum IMAGE
{
	NORMAL = 1,
	RESIZE = 0
};

class ImageManager
{

	
	Image* image;
	const string IMAGE_PATH = "Images_Input\\";
	vector<Mat>resized_img_mat_list;
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

	vector<Mat>getImageMatrices(IMAGE img);
	vector<vector<Pixel>>getRGBValues_list(IMAGE img);
	int getImageWidth(Mat src){ return src.cols; }
	int getImageHeight(Mat src){ return src.rows; }
	
private:
	vector<Pixel> compute_RGB_Values(Mat mat);
	vector<Mat> resize_images(int width, int height, vector<Mat>img_mat_list);
	vector<Mat>Normalize_Image_Mat(vector<Mat>);
};









#endif