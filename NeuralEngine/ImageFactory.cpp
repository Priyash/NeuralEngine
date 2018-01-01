#include "ImageFactory.h"

ImageFactory::ImageFactory()
{
	image = new ImageLoader();
	
	Util::getInstance()->read_Json();
	vector<JSON_VALUE>input_list = Util::getInstance()->getValues(CONFIG::INPUT_SHAPE);
	IMAGE_INPUT_PATH = input_list[0].json_str_value;
	IMAGE_RESIZE_WIDTH = input_list[2].json_int_value;
	IMAGE_RESIZE_HEIGHT = input_list[3].json_int_value;

	cout << IMAGE_INPUT_PATH << endl;
}

ImageFactory::~ImageFactory()
{
	image = nullptr;
}

//THIS METHOD IS ONLY REQUIRED WHEN THERE IS A NEED FOR LIST OF IMAGES NAME
vector<string> ImageFactory::getImageList()
{
	return image->loadImages(IMAGE_INPUT_PATH);
}

//THIS METHOD ONLY GET CALLED WHEN IMAGES OBJECT IS LOADED INTO THE MEMORY BY NORMALIZING THE IMAGES
vector<Mat> ImageFactory::loadAndReadImages()
{
	vector<Mat>img_mat_data_list = image->readImages(image->loadImages(IMAGE_INPUT_PATH));
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
		resize(i, i, Size(IMAGE_RESIZE_WIDTH, IMAGE_RESIZE_HEIGHT), 0, 0, INTER_CUBIC);
		resized_img_mat_list.push_back(i);
		resized_img.release();
	}

	return resized_img_mat_list;
}
