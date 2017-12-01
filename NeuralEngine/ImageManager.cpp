#include"ImageManager.h"



vector<Mat> ImageManager::getImageMatrices(IMAGE img)
{
	vector<Mat>img_mat_data_list;
	if (img == IMAGE::NORMAL)
	{
		img_mat_data_list = Normalize_Image_Mat(image->readImages());
	}
	else if (img == IMAGE::RESIZE)
	{
		img_mat_data_list = Normalize_Image_Mat(resize_images(128, 128 ,image->readImages()));
		
	}
	return img_mat_data_list;
}

vector<vector<Pixel>> ImageManager::getRGBValues_list(IMAGE img)
{
	vector<Mat>img_mat_list = getImageMatrices(img);
	vector<vector<Pixel>>pixel_mat;
	for (auto i : img_mat_list)
	{
		vector<Pixel>pixel_list = compute_RGB_Values(i);
		pixel_mat.push_back(pixel_list);
		pixel_list.clear();
	}
	
	return pixel_mat;
}

vector<Pixel> ImageManager::compute_RGB_Values(Mat img)
{
	vector<Pixel>pixel_list;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Pixel px;
			Vec3b bgrPixel = img.at<Vec3b>(i, j);
			px.r = bgrPixel[2];
			px.g = bgrPixel[1];
			px.b = bgrPixel[0];
			
			pixel_list.push_back(px);
		}
	}

	return pixel_list;
}


vector<Mat> ImageManager::resize_images(int width, int height, vector<Mat>img_mat_list)
{
	
	for (auto i : img_mat_list)
	{
		Mat resized_img;
		resize(i, resized_img, Size(width, height), 0, 0, INTER_CUBIC);
		resized_img_mat_list.push_back(resized_img);
		resized_img.release();
	}

	return resized_img_mat_list;
}

vector<Mat>ImageManager::Normalize_Image_Mat(vector<Mat>img_mat_list)
{
	vector<Mat>norm_img_mat_list;
	for (auto i : img_mat_list)
	{
		Mat norm_img_mat;
		normalize(i, norm_img_mat,0.0,255,NORM_MINMAX);
		norm_img_mat_list.push_back(norm_img_mat);
		norm_img_mat.release();
	}

	return norm_img_mat_list;
}