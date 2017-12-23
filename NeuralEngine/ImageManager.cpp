#include"ImageManager.h"
#include"Util.h"


vector<Mat> ImageManager::getImageMatrices(IMAGE img)
{
	vector<Mat>img_mat_data_list;
	if (img == IMAGE::NORMAL)
	{
		img_mat_data_list = image->normalize_images(image->readImages());
	}
	else if (img == IMAGE::RESIZE)
	{

		img_mat_data_list = resize_images(Util::getInstance()->getResizeValue().width, Util::getInstance()->getResizeValue().height, image->readImages());
		img_mat_data_list = image->normalize_images(img_mat_data_list);
	}

	return img_mat_data_list;
}


vector<Mat> ImageManager::resize_images(int width, int height, vector<Mat>img_mat_list)
{
	cout << width << endl;
	for (auto i : img_mat_list)
	{
		Mat resized_img;
		resize(i, i, Size(width, height), 0, 0, INTER_CUBIC);
		resized_img_mat_list.push_back(i);
		resized_img.release();
	}

	return resized_img_mat_list;
}
