#include<iostream>
#include"ImageManager.h"



int main()
{

	ImageManager* manager = new ImageManager();
	
	vector<Mat>data = manager->getImageMatrices();
	for (auto i : data)
	{
		cout << manager->getImageHeight(i) << endl;
	}

	return 0;
}