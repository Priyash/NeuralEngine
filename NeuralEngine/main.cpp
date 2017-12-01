#include<iostream>
#include"ImageManager.h"



int main()
{

	ImageManager* manager = new ImageManager();
	
	vector<vector<Pixel>>pixel_list = manager->getRGBValues_list(IMAGE::RESIZE);

	for (auto i : pixel_list)
	{
		vector<Pixel>px = i;
		for (auto j : px)
		{
			cout << j.r << "," << j.g << "," << j.b << endl;
			
		}
		break;
		cout << px[0].r << "," << px[5].g << "," << px[5].b << endl;
	}

	

	return 0;
}