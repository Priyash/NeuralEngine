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
	CONVLAYER1,
	INPUT_SHAPE
};

enum LAYER_ID
{
	TENSOR_SHAPE = 0,
	FILTER_SHAPE = 1,
	BIAS_SHAPE = 2,
	CONV_SHAPE = 3
};

enum TENSOR_SHAPE_ID
{
	TENSOR_BATCH_SIZE = 0,
	TENSOR_INPUT_FEATURE_MAP = 1,
	TENSOR_IMAGE_WIDTH = 2,
	TENSOR_IMAGE_HEIGHT = 3
};

enum FILTER_SHAPE_ID
{
	FILTER_INPUT_FEATURE_MAP = 0,
	FILTER_OUTPUT_FEATURE_MAP = 1,
	FILTER_WIDTH = 2,
	FILTER_HEIGHT = 3
};

enum BIAS_SHAPE_ID
{
	BIAS_BATCH_SIZE = 0,
	BIAS_OUTPUT_FEATURE_MAP = 1,
	BIAS_WIDTH = 2,
	BIAS_HEIGHT = 3
};


enum CONV_SHAPE_ID
{
	CONV_PAD_WIDTH = 0,
	CONV_PAD_HEIGHT = 1,
	CONV_VERTICAL_STRIDE = 2,
	CONV_HORIZONTAL_STRIDE = 3,
	CONV_DILATION_HEIGHT = 4,
	CONV_DILATION_WIDTH = 5,
	CONV_CONV_FWD_PREF = 6
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
	vector<Value::ConstMemberIterator>getObjects(CONFIG con);
	vector<JSON_VALUE> getValues(Value::ConstMemberIterator);
private:
	string toStr(CONFIG con);
	

};







#endif