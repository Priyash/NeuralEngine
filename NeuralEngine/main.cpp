#include<iostream>
#include"ImageManager.h"



int main()
{

	ImageManager* manager = new ImageManager();
	vector<Mat>data = manager->getImageMatrices(IMAGE::RESIZE); 

	

	return 0;
}