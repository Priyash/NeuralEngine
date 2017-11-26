#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H
#include"Image.h"




class ImageManager
{

	
	Image* image;
	const string IMAGE_PATH = "Images_Input\\";
public:
	ImageManager()
	{
		image = new ImageLoader();
	}

	~ImageManager()
	{

	}

	void show_image_list();
};









#endif