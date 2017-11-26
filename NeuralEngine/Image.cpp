#include"Image.h"



string ImageLoader::ToString(WCHAR* data)
{
	wstring wstr(data);
	string temp_str(wstr.begin(), wstr.end());
	return temp_str;
}

vector<string> ImageLoader::list_images(const string& image_base_path)
{
	this->image_base_path = image_base_path;
	vector<string>image_data_list;
	wstring pattern(image_base_path.begin(), image_base_path.end());
	pattern.append(L"*.*");
	if ((handler = FindFirstFile(pattern.c_str(), &fdata)) != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (fdata.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)
			{
				if (wcscmp(fdata.cFileName, L".") != 0 && wcscmp(fdata.cFileName, L"..") != 0)
				{
					//FOUND THE DIRECTORY

				}
			}
			else
			{
				//FILES 
				image_data_list.push_back(ToString(fdata.cFileName));

			}

		} while (FindNextFile(handler, &fdata));

	}
	else
	{
		cout << "Error with handler" << endl;
	}

	return image_data_list;
}

void ImageLoader::loadImages(const string& image_base_path)
{
	image_list = list_images(image_base_path);
}



vector<Mat> ImageLoader::readImages()
{
	for (auto i : image_list)
	{
		string img_full_path = image_base_path + i;
		Mat img_mat = readImage(img_full_path);
		image_matrices.push_back(img_mat);
	}
	
	return image_matrices;
}

Mat ImageLoader::readImage(string image_file)
{
	Mat img_mat = imread(image_file);
	return img_mat;
}

vector<unsigned char*>ImageLoader::readRawImageData()
{
	for (auto i : image_matrices)
	{
		unsigned char* img_data = (unsigned char*)i.data;
		raw_image_data_list.push_back(img_data);
	}

	
	return raw_image_data_list;
}