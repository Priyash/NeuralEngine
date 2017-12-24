#ifndef UTIL_H
#define UTIL_H
#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<algorithm>


using namespace std;

struct Config
{
	string config_key;
	string config_value;
};

struct Resize
{
	int width;
	int height;
};

enum CONFIG_ID
{
	BASE_IMAGE_PATH,
	IMAGE_RESIZE,
	IMAGE_OUTPUT_PATH,
	IMAGE_BATCH,
	IMAGE_CHANNELS
};

class Util
{
	string CONFIG_PATH;
	static Util* util;
	vector<Config>config_list;
	Resize size;
	string IMAGE_BASE_PATH_VALUE;
	string IMAGE_RESIZE_WIDTH_VALUE;
	string IMAGE_RESIZE_HEIGHT_VALUE;
	string IMAGE_OUTPUT_PATH_VALUE;
	int IMAGE_BATCH_VALUE;
	int IMAGE_CHANNELS_VALUE;

	Util();
public:
	static Util* getInstance();
	void read_config_file(string config_file);

	void parse_config_file(CONFIG_ID con);

	//ADD THE GET METHOD FOR THE VALUE RETRIEVAL
	string getImage_Base_Path();
	Resize getResizeValue();
	string getImageOutputPath();
	int getImageBatch();
	int getImageChannel();

private:
	
	vector<string>split(string data, string delim);
	string find_config_value_by_key(string key);

	string getKey(CONFIG_ID con);
	//ADD THE SET METHOD FOR VALUE SET FROM CONFIG FILE
	void setImage_Base_Path(string value);

	void setResizeValue(int width, int height);
	void setImageOutputPath(string path);

	void setImageBatch(int value);
	void setImageChannel(int value);
};







#endif