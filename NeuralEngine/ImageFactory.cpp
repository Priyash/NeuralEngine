#include "ImageFactory.h"

ImageFactory::ImageFactory()
{
	image = new ImageLoader();
	CONFIG_FILE = "config.txt";
	Util::getInstance()->read_config_file(CONFIG_FILE);
	Util::getInstance()->parse_config_file(CONFIG_ID::BASE_IMAGE_PATH);
	Util::getInstance()->parse_config_file(CONFIG_ID::IMAGE_RESIZE);
	IMAGE_PATH = Util::getInstance()->getImage_Base_Path();
	cout << IMAGE_PATH << endl;
	
}

ImageFactory::~ImageFactory()
{
	image = nullptr;
}


vector<Mat> ImageFactory::loadAndReadImages()
{
	image->loadImages(IMAGE_PATH);
	vector<Mat>img_mat_data_list = image->normalize_images(image->readImages());
	return img_mat_data_list;
}

vector<Mat> ImageFactory::Normalize_images(vector<Mat>img_list)
{
	vector<Mat>img_mat_norm_data_list = image->normalize_images(img_list);
	return img_mat_norm_data_list;
}

vector<Mat> ImageFactory::ResizeImages(vector<Mat>img_mat_list)
{
	vector<Mat>resized_img_mat_list;
	for (auto i : img_mat_list)
	{
		Mat resized_img;
		resize(i, i, Size(Util::getInstance()->getResizeValue().width, Util::getInstance()->getResizeValue().height), 0, 0, INTER_CUBIC);
		resized_img_mat_list.push_back(i);
		resized_img.release();
	}

	return resized_img_mat_list;
}
