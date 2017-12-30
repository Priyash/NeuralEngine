#ifndef UTIL_H
#define UTIL_H
#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<rapidjson\document.h>
#include "rapidjson\writer.h"
#include "rapidjson\stringbuffer.h"
#include <fstream>
#include <rapidjson/istreamwrapper.h>

using namespace std;
using namespace rapidjson;

struct JSON_VALUE
{
	string json_str_value;
	int json_int_value;
};

enum CONFIG
{
	INPUT_SHAPE,
	TENSOR_SHAPE,
	BIAS_SHAPE,
	FILTER_SHAPE,
	CONV_SHAPE
};


class Util
{
	static Util* util;
	Util();
	Document doc;
	string config_file;
public:
	static Util* getInstance();
	void read_Json();
	JSON_VALUE getValue(CONFIG con, string key);
	vector<JSON_VALUE> getValues(CONFIG con);
private:
	string toStr(CONFIG con);
	

};







#endif