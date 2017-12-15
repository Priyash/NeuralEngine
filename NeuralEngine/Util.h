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

enum CONFIG
{
	BASE_IMAGE_PATH
};

class Util
{
	string CONFIG_PATH;
	static Util* util;
	vector<Config>config_list;
	//ADD THE KEY-VALUE PAIR
	string IMAGE_BASE_PATH_KEY;
	string IMAGE_BASE_PATH_VALUE;



	Util();
public:
	static Util* getInstance();
	void read_config_file(string config_file);

	void parse_config_file(CONFIG con);

	//ADD THE GET METHOD FOR THE VALUE RETRIEVAL
	string getImage_Base_Path();

private:
	
	vector<string>split(string data, string delim);
	string find_config_value_by_key(string key);

	//ADD THE SET METHOD FOR VALUE SET FROM CONFIG FILE
	void setImage_Base_Path(string value);
};







#endif