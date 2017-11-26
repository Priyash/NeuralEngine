#include"ImageManager.h"


void ImageManager::show_image_list()
{
	image->loadImages(IMAGE_PATH);
	vector<Mat>img_mat = image->readImages();
	vector<unsigned char*>raw_img_data = image->readRawImageData();
}