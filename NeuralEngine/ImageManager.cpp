#include"ImageManager.h"
#include"Util.h"


//THIS METHOD READ THE IMAGES FROM THE INPUT PATH[CONFIG FILE] AND LOAD THE IMAGE-OBJECTS INTO THE MEMORY
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

//THIS METHOD GETS ALL THE LIST OF IMAGE NAME FROM THE INPUT PATH SPECIFIED IN CONFIG FILE
vector<string> ImageManager::getImageList()
{
	return Ifactory->getImageList();
}
