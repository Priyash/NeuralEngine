#include"ImageManager.h"
#include"Util.h"


vector<Mat> ImageManager::getImageMatrices(IMAGE img)
{
	vector<Mat>img_mat_data_list;
	if (img == IMAGE::NORMAL)
	{
		img_mat_data_list = Ifactory->Normalize_images(Ifactory->loadAndReadImages());
	}
	else if (img == IMAGE::RESIZE)
	{
		img_mat_data_list = Ifactory->Normalize_images(Ifactory->ResizeImages(Ifactory->loadAndReadImages()));
	}

	return img_mat_data_list;
}

